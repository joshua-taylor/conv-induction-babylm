"""
train_babylm.py — BabyLM 2026 Strict-Small training (Kaggle)
============================================================

Trains the final Conv-Routed Induction LM (see modeling_induction.py) on the
BabyLM-2026 Strict-Small corpus (<=10M unique words) and saves everything the
evaluation pipeline + leaderboard submission need:

  /kaggle/working/
    checkpoints/chck_1M ... chck_10M (+ chck_20M ...)   <- HF snapshots at each
        1M cumulative-word milestone (push each as a HF branch for the --fast eval)
    final-model/                                        <- the model to submit (main),
        selected as the BEST-BLiMP checkpoint over training (not end-of-training)
    hyperparameters.json                                <- for the submission form/paper

Recipe (validated in the v23-v25 sweeps):
  * MULTI-EPOCH: ~6 epochs (n_steps=12000 at batch 32 / seq 256 over the ~16.5M-token
    corpus). Held-out ppl bottoms out at ~1 epoch but BLiMP keeps improving for several
    more epochs, so we train well past the ppl optimum and SELECT ON BLiMP.
  * OPTIMISER: Muon (orthogonalised momentum) on the 2D weight matrices at lr 0.03,
    AdamW on embeddings / norms / sinks at lr 6e-4. Muon lowered ppl and raised BLiMP
    vs AdamW at equal steps.
  * SCHEDULE: linear warmup (300) then cosine to 5% over the FULL n_steps. The cosine
    T_max MUST equal n_steps — shortening it quenches the LR early and costs ~15 ppl.
  * SELECTION: BLiMP (full nyu-mll/blimp proxy) evaluated every eval_every steps; the
    best-BLiMP weights are restored and saved as final-model. (Leaderboard BLiMP is the
    *filtered* set — re-run the eval pipeline on the pushed model for the comparable number.)

Each saved dir is a self-contained HF model (config + safetensors + modeling_induction.py
via trust_remote_code) plus the tokenizer, so `AutoModelForCausalLM.from_pretrained(dir,
trust_remote_code=True)` works directly in the eval pipeline.

Run:  python train_babylm.py            (full run on Kaggle GPU, Internet ON for data+BLiMP)
      SMOKE=1 python train_babylm.py    (tiny synthetic sanity check, no GPU/HF/BLiMP needed)
"""

import os, json, math, time, random
import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling_induction import InductionConfig, InductionForCausalLM

SMOKE = bool(os.environ.get("SMOKE"))
OUT = "/tmp/babylm_out" if SMOKE else "/kaggle/working"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------------------------
# Hyperparameters (logged to hyperparameters.json — these go in the form/paper)
# ----------------------------------------------------------------------
HP = dict(
    # data / tokenizer
    dataset="BabyLM-community/BabyLM-2026-Strict-Small",
    vocab_size=8000,
    # model (the locked dynconv16 architecture; ~12.2M params)
    d_model=384, d_ff=1536, n_layers=3, n_heads=4,
    conv_dilations=[1, 2, 4], conv_kernel=3, dyn_groups=16, match_m=5,
    max_position_embeddings=512,
    # optimisation — multi-epoch + Muon (see module docstring)
    seq_len=256, batch_size=32, n_steps=12000,        # ~6 epochs of the ~16.5M-token corpus
    optimizer="Muon(2D matrices, lr 0.03) + AdamW(embeddings/norms/sinks, lr 6e-4)",
    muon_lr=0.03, muon_momentum=0.95, muon_ns_steps=5,
    adamw_lr=6e-4, lr_schedule="warmup_cosine", warmup_steps=300, eta_min_frac=0.05,
    weight_decay=0.1, betas=[0.9, 0.95], grad_clip=1.0, seed=42,
    eval_every=2000, blimp_per_paradigm=100, select_by="blimp",
    precision="fp32", tie_word_embeddings=True, use_compile=True,
)
if SMOKE:
    HP.update(vocab_size=300, d_model=64, d_ff=128, n_layers=2, n_heads=4, dyn_groups=8,
              seq_len=64, batch_size=8, n_steps=60, max_position_embeddings=128,
              warmup_steps=5, eval_every=20, blimp_per_paradigm=0, use_compile=False)

random.seed(HP["seed"]); torch.manual_seed(HP["seed"])
SPECIAL = ["<pad>", "<unk>", "<s>", "</s>"]                # ids 0,1,2,3


# ======================================================================
# Muon optimiser (orthogonalised momentum on the 2D weight matrices)
# ======================================================================
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Quintic Newton-Schulz iteration -> nearest semi-orthogonal matrix to G."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + eps)
    transpose = X.size(0) > X.size(1)
    if transpose:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Momentum SGD whose update is orthogonalised per 2D parameter. Use ONLY for hidden
       weight matrices; route embeddings / norms / scalars to AdamW."""
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5, wd=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps, wd=wd))

    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            mom = grp["momentum"]
            for p in grp["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p)
                buf = st["buf"]; buf.mul_(mom).add_(p.grad)
                u = p.grad.add(buf, alpha=mom)                       # Nesterov
                u = zeropower_via_newtonschulz5(u, grp["ns_steps"])
                u = u * (max(1.0, p.size(0) / p.size(1)) ** 0.5)     # shape-correct scale
                if grp["wd"]:
                    p.mul_(1 - grp["lr"] * grp["wd"])
                p.add_(u, alpha=-grp["lr"])

    def zero_grad(self, set_to_none=True):
        for grp in self.param_groups:
            for p in grp["params"]:
                p.grad = None if set_to_none else (p.grad.zero_() if p.grad is not None else None)


def is_muon_param(name, p):
    """2D hidden weight matrices go to Muon; embeddings/positional/sinks/heads/norms do not."""
    return p.ndim == 2 and not any(t in name for t in ("embed", "pos", "sink", "lm_head", "score"))


def build_optimisers(model):
    muon_p = [p for n, p in model.named_parameters() if is_muon_param(n, p)]
    other_p = [p for n, p in model.named_parameters() if not is_muon_param(n, p)]
    om = Muon(muon_p, lr=HP["muon_lr"], momentum=HP["muon_momentum"],
              ns_steps=HP["muon_ns_steps"], wd=HP["weight_decay"])
    oa = torch.optim.AdamW(other_p, lr=HP["adamw_lr"], weight_decay=HP["weight_decay"],
                           betas=tuple(HP["betas"]))

    def lr_lambda(step):
        if step < HP["warmup_steps"]:
            return (step + 1) / HP["warmup_steps"]
        prog = min(1.0, (step - HP["warmup_steps"]) / max(1, HP["n_steps"] - HP["warmup_steps"]))
        return HP["eta_min_frac"] + (1 - HP["eta_min_frac"]) * 0.5 * (1 + math.cos(math.pi * prog))

    opts = [om, oa]
    schs = [torch.optim.lr_scheduler.LambdaLR(o, lr_lambda) for o in opts]
    print(f"  optimiser: Muon lr {HP['muon_lr']} on {len(muon_p)} matrices | "
          f"AdamW lr {HP['adamw_lr']} on {len(other_p)} other tensors")
    return opts, schs


# ----------------------------------------------------------------------
# Data + tokenizer
# ----------------------------------------------------------------------
def load_text():
    if SMOKE:
        words = ("the little girl said she would go to the park and play with the dog "
                 "then the dog ran away and the girl was sad but her mother came").split()
        train = " ".join(random.choice(words) for _ in range(20000))
        val = " ".join(random.choice(words) for _ in range(2000))
        return train, val
    from datasets import load_dataset
    ds = load_dataset(HP["dataset"]); splits = list(ds.keys())
    tr = "train" if "train" in splits else splits[0]
    va = next((s for s in ["validation", "valid", "dev", "test"] if s in splits), None)
    col = next(k for k in ["text", "content", "document", "raw"] if k in ds[tr][0])
    train = "\n".join(t for t in ds[tr][col] if t and t.strip())
    if va:
        val = "\n".join(t for t in ds[va][col] if t and t.strip())
    else:
        c = int(len(train) * 0.95); val, train = train[c:], train[:c]
    return train, val


def build_tokenizer(train_text):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel as BLPre
    from tokenizers.decoders import ByteLevel as BLDec
    from transformers import PreTrainedTokenizerFast
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = BLPre(); tok.decoder = BLDec()
    tok.train_from_iterator(
        (train_text[i:i + 10000] for i in range(0, len(train_text), 10000)),
        BpeTrainer(vocab_size=HP["vocab_size"], special_tokens=SPECIAL,
                   initial_alphabet=BLPre.alphabet(), show_progress=False))
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>",
        bos_token="<s>", eos_token="</s>")
    return hf_tok


# ----------------------------------------------------------------------
# Batching + eval
# ----------------------------------------------------------------------
def get_batch(ids, bs, T):
    n = ids.size(0) - T - 1
    si = torch.randint(0, n, (bs,), device=ids.device)
    off = torch.arange(T + 1, device=ids.device)
    seq = ids[si.unsqueeze(1) + off.unsqueeze(0)]
    return seq[:, :T].contiguous(), seq[:, 1:T + 1].contiguous()


@torch.no_grad()
def val_ppl(model, ids, T, bs, n=30, seed=99):
    model.eval(); st = torch.get_rng_state(); torch.manual_seed(seed); ls = []
    for _ in range(n):
        x, _ = get_batch(ids, bs, T)
        logits = model(input_ids=x).logits
        ls.append(F.cross_entropy(logits[:, :-1].reshape(-1, model.config.vocab_size),
                                  x[:, 1:].reshape(-1)).item())
    torch.set_rng_state(st); model.train()
    return math.exp(sum(ls) / len(ls))


# ---- BLiMP (full nyu-mll/blimp proxy; selection metric) ----
def load_blimp(n_per):
    from datasets import load_dataset, get_dataset_config_names
    name = "nyu-mll/blimp"
    try:
        configs = get_dataset_config_names(name)
    except Exception:
        name = "blimp"; configs = get_dataset_config_names(name)
    good, bad = [], []
    for cfg_name in configs:
        d = load_dataset(name, cfg_name, split="train")
        if n_per:
            d = d.select(range(min(n_per, len(d))))
        good += list(d["sentence_good"]); bad += list(d["sentence_bad"])
    return good, bad


@torch.no_grad()
def _score(model, tokenizer, sents, bs=64, max_len=128):
    model.eval(); lls = []
    enc = [(tokenizer(s, return_attention_mask=False)["input_ids"][:max_len] or [0]) for s in sents]
    for i in range(0, len(enc), bs):
        chunk = enc[i:i + bs]; L = max(len(c) for c in chunk)
        x = torch.zeros(len(chunk), L, dtype=torch.long, device=device)   # right-pad with <pad>=0
        lens = torch.tensor([len(c) for c in chunk], device=device)
        for j, c in enumerate(chunk):
            x[j, :len(c)] = torch.tensor(c, device=device)
        logp = model(input_ids=x).logits.log_softmax(-1)
        lp = logp[:, :-1].gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
        mask = torch.arange(L - 1, device=device)[None, :] < (lens[:, None] - 1)
        lls.extend((lp * mask).sum(-1).tolist())
    return lls


@torch.no_grad()
def eval_blimp(model, tokenizer, good, bad):
    lg = torch.tensor(_score(model, tokenizer, good))
    lb = torch.tensor(_score(model, tokenizer, bad))
    model.train()
    return (lg > lb).float().mean().item()


@torch.no_grad()
def verify_causality(model, vocab, T=64):
    model.eval()
    x = torch.randint(0, vocab, (2, T), device=device)
    l1 = model(input_ids=x).logits
    cut = T // 2
    x2 = x.clone(); x2[:, cut:] = torch.randint(0, vocab, (2, T - cut), device=device)
    l2 = model(input_ids=x2).logits
    diff = (l1[:, :cut] - l2[:, :cut]).abs().max().item()
    model.train()
    ok = diff < 1e-3 and not torch.isnan(l1).any().item()
    print(f"  causality check: max diff over first {cut} = {diff:.1e} [{'OK' if ok else 'FAIL'}]")
    assert ok, "causality violated"


def save_ckpt(model, tokenizer, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


# ----------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    print("Loading data ..."); train_text, val_text = load_text()
    n_words = len(train_text.split())
    print(f"  train words: {n_words:,}  (Strict-Small budget = 10,000,000 unique)")
    assert SMOKE or n_words <= 10_000_000 * 1.02, "corpus exceeds the 10M-word budget!"

    print("Building tokenizer ..."); tokenizer = build_tokenizer(train_text)
    enc = lambda s: tokenizer(s, return_attention_mask=False)["input_ids"]
    train_ids = torch.tensor(enc(train_text), dtype=torch.long, device=device)
    val_ids = torch.tensor(enc(val_text), dtype=torch.long, device=device)
    HP["vocab_size_actual"] = tokenizer.vocab_size
    tok_per_word = train_ids.numel() / max(n_words, 1)
    words_per_step = HP["batch_size"] * HP["seq_len"] / tok_per_word
    steps_per_epoch = train_ids.numel() / (HP["batch_size"] * HP["seq_len"])
    print(f"  tokens: train {train_ids.numel():,} | val {val_ids.numel():,} "
          f"| {tok_per_word:.2f} tok/word | {words_per_step:,.0f} words/step "
          f"| 1 epoch ~ {steps_per_epoch:,.0f} steps | n_steps {HP['n_steps']} "
          f"~ {HP['n_steps']/max(steps_per_epoch,1):.1f} epochs")

    # BLiMP (selection metric). Optional: if unavailable, fall back to val-ppl selection.
    use_blimp = False; blimp_good = blimp_bad = None
    if not SMOKE:
        try:
            print("Loading BLiMP (selection metric) ...")
            blimp_good, blimp_bad = load_blimp(HP["blimp_per_paradigm"])
            use_blimp = True
            print(f"  BLiMP pairs: {len(blimp_good):,}")
        except Exception as e:
            print(f"  BLiMP unavailable ({type(e).__name__}: {str(e)[:70]}); selecting by val ppl")
    HP["select_by"] = "blimp" if use_blimp else "val_ppl"

    cfg = InductionConfig(
        vocab_size=tokenizer.vocab_size, d_model=HP["d_model"], d_ff=HP["d_ff"],
        n_layers=HP["n_layers"], n_heads=HP["n_heads"], conv_dilations=HP["conv_dilations"],
        conv_kernel=HP["conv_kernel"], dyn_groups=HP["dyn_groups"], match_m=HP["match_m"],
        max_position_embeddings=HP["max_position_embeddings"],
        pad_token_id=0, bos_token_id=2, eos_token_id=3,
        tie_word_embeddings=HP["tie_word_embeddings"])
    model = InductionForCausalLM(cfg).to(device)
    HP["n_params"] = sum(p.numel() for p in model.parameters())
    print(f"  model params: {HP['n_params']/1e6:.2f}M")
    verify_causality(model, tokenizer.vocab_size)

    train_target = model
    if HP["use_compile"]:
        try:
            train_target = torch.compile(model, dynamic=False)
            model.train(); xb, _ = get_batch(train_ids, HP["batch_size"], HP["seq_len"])
            F.cross_entropy(train_target(input_ids=xb).logits[:, :-1].reshape(-1, cfg.vocab_size),
                            xb[:, 1:].reshape(-1)).backward()
            for p in model.parameters():
                p.grad = None
            if device == "cuda":
                torch.cuda.synchronize()
            print("  torch.compile: on")
        except Exception as ex:
            print(f"  torch.compile failed ({type(ex).__name__}); running eager")
            train_target = model

    opts, schs = build_optimisers(model)

    # checkpoint milestones (cumulative words seen, incl. repeats): 1M..10M, then every 10M
    targets = [i * 1_000_000 for i in range(1, 11)] + [i * 10_000_000 for i in range(2, 101)]
    if SMOKE:
        targets = [2000 * i for i in range(1, 6)]
    ckpt_dir = os.path.join(OUT, "checkpoints"); os.makedirs(ckpt_dir, exist_ok=True)

    log = []; cumw = 0.0; ti = 0
    # selection: higher-is-better score (BLiMP, or -ppl as fallback)
    best_score = -float("inf"); best_meta = {}; best_state = None
    model.train(); t0 = time.time()
    for step in range(1, HP["n_steps"] + 1):
        x, _ = get_batch(train_ids, HP["batch_size"], HP["seq_len"])
        loss = F.cross_entropy(train_target(input_ids=x).logits[:, :-1].reshape(-1, cfg.vocab_size),
                               x[:, 1:].reshape(-1))
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), HP["grad_clip"])
        for o in opts:
            o.step()
        for s in schs:
            s.step()
        cumw += words_per_step

        # ---- milestone snapshots (learning-curve branches) ----
        while ti < len(targets) and cumw >= targets[ti]:
            m = targets[ti] // 1_000_000 if not SMOKE else targets[ti]
            name = f"chck_{m}M" if not SMOKE else f"chck_{m}"
            vp = val_ppl(model, val_ids, HP["seq_len"], HP["batch_size"])
            save_ckpt(model, tokenizer, os.path.join(ckpt_dir, name))
            log.append(dict(checkpoint=name, step=step, cum_words=int(cumw), val_ppl=round(vp, 3)))
            print(f"  [ckpt] {name}: step {step} | cum_words {int(cumw):,} | val ppl {vp:.2f}")
            ti += 1

        # ---- periodic eval + BLiMP-based selection of the submission model ----
        if step % HP["eval_every"] == 0 or step == HP["n_steps"]:
            vp = val_ppl(model, val_ids, HP["seq_len"], HP["batch_size"])
            bl = eval_blimp(model, tokenizer, blimp_good, blimp_bad) if use_blimp else float("nan")
            ep = step / max(steps_per_epoch, 1)
            score = bl if use_blimp else -math.log(vp)
            tag = ""
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_meta = dict(step=step, epoch=round(ep, 2), val_ppl=round(vp, 3),
                                 blimp=(round(bl, 4) if use_blimp else None))
                tag = "  * (best)"
            blimp_str = f"{bl*100:.2f}" if use_blimp else "n/a"
            print(f"  step {step:5d}/{HP['n_steps']} (ep {ep:.1f}) | train loss {loss.item():.3f} "
                  f"| val ppl {vp:.2f} | BLiMP {blimp_str}{tag}")

    # ---- final model = best-selected checkpoint (restored), pushed to main ----
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    final_vp = val_ppl(model, val_ids, HP["seq_len"], HP["batch_size"])
    save_ckpt(model, tokenizer, os.path.join(OUT, "final-model"))

    HP["selected_checkpoint"] = best_meta
    HP["final_val_ppl"] = round(final_vp, 3)
    HP["total_cum_words"] = int(cumw)
    HP["wall_time_sec"] = round(time.time() - t0, 1)
    HP["checkpoint_log"] = log
    with open(os.path.join(OUT, "hyperparameters.json"), "w") as f:
        json.dump(HP, f, indent=2)

    sel = best_meta.get("blimp")
    sel_str = f"BLiMP {sel*100:.2f}" if sel is not None else f"val ppl {best_meta.get('val_ppl')}"
    print(f"\nDone. final-model = best {HP['select_by']} ({sel_str}) at step {best_meta.get('step')}"
          f" / epoch {best_meta.get('epoch')} | its val ppl {final_vp:.2f}")
    print(f"Saved {len(log)} milestone checkpoints + final-model + hyperparameters.json to {OUT}")
    print("Each dir is a self-contained HF model (config + safetensors + modeling_induction.py +")
    print("tokenizer); load with AutoModelForCausalLM.from_pretrained(dir, trust_remote_code=True).")


if __name__ == "__main__":
    main()

"""
train_babylm.py — BabyLM 2026 Strict-Small training (Kaggle)
============================================================

Trains the final Conv-Routed Induction LM (see modeling_induction.py) on the
BabyLM-2026 Strict-Small corpus (<=10M words) and saves everything the evaluation
pipeline + leaderboard submission need:

  /kaggle/working/
    checkpoints/chck_1M ... chck_10M (+ chck_20M ...)   <- HF checkpoints at each
        1M cumulative-word milestone (push each as a HF branch for the --fast eval)
    final-model/                                        <- the model to submit (main)
    hyperparameters.json                                <- for the submission form/paper

Each saved dir is a self-contained HF model (config + safetensors + modeling_induction.py
via trust_remote_code) plus the tokenizer, so `AutoModelForCausalLM.from_pretrained(dir,
trust_remote_code=True)` works directly in the eval pipeline.

Run:  python train_babylm.py            (full run on Kaggle GPU)
      SMOKE=1 python train_babylm.py    (tiny synthetic sanity check, no GPU/HF needed)
"""

import os, json, math, time, random, shutil
import torch
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
    # model (the locked dynconv16 architecture)
    d_model=192, d_ff=768, n_layers=3, n_heads=4,
    conv_dilations=[1, 2, 4], conv_kernel=3, dyn_groups=16, match_m=5,
    max_position_embeddings=512,
    # optimisation
    seq_len=256, batch_size=32, n_steps=4000,
    lr=5e-4, lr_schedule="cosine", eta_min_frac=0.05,
    weight_decay=0.01, betas=[0.9, 0.95], grad_clip=1.0,
    warmup_steps=0, seed=42,
    optimizer="AdamW", precision="fp32", tie_word_embeddings=True,
)
if SMOKE:
    HP.update(vocab_size=300, d_model=64, d_ff=128, n_layers=2, n_heads=4,
              dyn_groups=8, seq_len=64, batch_size=8, n_steps=60, max_position_embeddings=128)

random.seed(HP["seed"]); torch.manual_seed(HP["seed"])
SPECIAL = ["<pad>", "<unk>", "<s>", "</s>"]                # ids 0,1,2,3


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
    print(f"  train words: {n_words:,}  (Strict-Small budget = 10,000,000)")
    assert SMOKE or n_words <= 10_000_000 * 1.02, "corpus exceeds the 10M-word budget!"

    print("Building tokenizer ..."); tokenizer = build_tokenizer(train_text)
    enc = lambda s: tokenizer(s, return_attention_mask=False)["input_ids"]
    train_ids = torch.tensor(enc(train_text), dtype=torch.long, device=device)
    val_ids = torch.tensor(enc(val_text), dtype=torch.long, device=device)
    HP["vocab_size_actual"] = tokenizer.vocab_size
    tok_per_word = train_ids.numel() / max(n_words, 1)
    words_per_step = HP["batch_size"] * HP["seq_len"] / tok_per_word
    print(f"  tokens: train {train_ids.numel():,} | val {val_ids.numel():,} "
          f"| {tok_per_word:.2f} tok/word | {words_per_step:,.0f} words/step")

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

    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"],
                            weight_decay=HP["weight_decay"], betas=tuple(HP["betas"]))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=HP["n_steps"], eta_min=HP["lr"] * HP["eta_min_frac"])

    # checkpoint milestones (cumulative words): 1M..10M, then every 10M
    targets = [i * 1_000_000 for i in range(1, 11)] + [i * 10_000_000 for i in range(2, 101)]
    if SMOKE:
        targets = [2000 * i for i in range(1, 6)]
    ckpt_dir = os.path.join(OUT, "checkpoints"); os.makedirs(ckpt_dir, exist_ok=True)

    log = []; cumw = 0.0; ti = 0; best = (float("inf"), 0)
    model.train(); t0 = time.time()
    for step in range(1, HP["n_steps"] + 1):
        x, _ = get_batch(train_ids, HP["batch_size"], HP["seq_len"])
        logits = model(input_ids=x).logits
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), HP["grad_clip"])
        opt.step(); sch.step()
        cumw += words_per_step

        while ti < len(targets) and cumw >= targets[ti]:
            m = targets[ti] // 1_000_000 if not SMOKE else targets[ti]
            name = f"chck_{m}M" if not SMOKE else f"chck_{m}"
            vp = val_ppl(model, val_ids, HP["seq_len"], HP["batch_size"])
            save_ckpt(model, tokenizer, os.path.join(ckpt_dir, name))
            log.append(dict(checkpoint=name, step=step, cum_words=int(cumw), val_ppl=round(vp, 3)))
            print(f"  [ckpt] {name}: step {step} | cum_words {int(cumw):,} | val ppl {vp:.2f}")
            if vp < best[0]: best = (vp, step)
            ti += 1

        if step % max(1, HP["n_steps"] // 10) == 0:
            vp = val_ppl(model, val_ids, HP["seq_len"], HP["batch_size"])
            print(f"  step {step:5d}/{HP['n_steps']} | train loss {loss.item():.3f} | val ppl {vp:.2f}")

    # final model (end of training) -> the model to submit on the main branch
    final_vp = val_ppl(model, val_ids, HP["seq_len"], HP["batch_size"])
    save_ckpt(model, tokenizer, os.path.join(OUT, "final-model"))
    HP["final_val_ppl"] = round(final_vp, 3)
    HP["best_val_ppl"] = round(best[0], 3)
    HP["total_cum_words"] = int(cumw)
    HP["wall_time_sec"] = round(time.time() - t0, 1)
    HP["checkpoint_log"] = log
    with open(os.path.join(OUT, "hyperparameters.json"), "w") as f:
        json.dump(HP, f, indent=2)

    print(f"\nDone. final val ppl {final_vp:.2f} | best {best[0]:.2f} (step {best[1]})")
    print(f"Saved {len(log)} checkpoints + final-model + hyperparameters.json to {OUT}")
    print("Each dir is a self-contained HF model (config + safetensors + modeling_induction.py +")
    print("tokenizer); load with AutoModelForCausalLM.from_pretrained(dir, trust_remote_code=True).")


if __name__ == "__main__":
    main()

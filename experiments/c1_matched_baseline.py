"""
c1_matched_baseline.py — the controlled head-to-head (C1)
=========================================================

The paper's whole "why" half rests on one claim: at equal budget, the attention-FREE
induction model matches or beats a learned-attention model. C1 makes that a single-variable
control. We hold EVERYTHING fixed — DynamicConv, SwiGLU FFN, learned positions, tokenizer,
data, optimiser, step budget, seeds — and swap ONLY the mixer:

    arm "induction"  : InductionMixer   (exact-token look-back, copies raw continuation)
    arm "attention"  : AttentionMixer   (causal multi-head self-attention + learnable sink)

The AttentionMixer is parameter-matched to the InductionMixer to the digit (same q/k/v/o and
sink shapes), so any difference is the mechanism, not capacity. Both arms compute the same
`cand` index (attention ignores it) and receive the same learned absolute positions, so the
positional treatment is matched too.

Outputs per (arm, seed): best val ppl + a saved HF checkpoint, ready for the official
babylm-eval BLiMP run. Reports mean±std per arm. (In-loop selection is by val ppl for
self-containment; for the headline BLiMP number, run the saved checkpoints through the
pipeline, or pass a `blimp_score` hook — see BLIMP_HOOK below.)

Run:
    python c1_matched_baseline.py                         # AdamW recipe, 3 seeds
    HF_MODEL_ID=you/repo python c1_matched_baseline.py    # reuse the production tokenizer
    SMOKE=1 python c1_matched_baseline.py                 # tiny CPU harness test

NOTE on matching the leaderboard exactly: production uses the Muon optimiser. The comparison
here is valid with any shared optimiser (both arms use the same code path); to reproduce the
production number, plug your Muon builder into build_optimizer() — it will apply to both arms.
"""

import os, math, time, random
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling_induction import (
    InductionConfig, InductionForCausalLM, RMSNorm,
)

SMOKE = bool(os.environ.get("SMOKE"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO = os.environ.get("HF_MODEL_ID")
SEEDS = [0] if SMOKE else [0, 1]
NEG = float("-inf")

# Optional: set to a callable(model, tokenizer, device) -> float BLiMP score to get BLiMP
# in-loop. Left None by default (use the official pipeline on the saved checkpoints).
BLIMP_HOOK = None


@dataclass
class Cfg:
    vocab_size: int = 8000
    d_model: int = 384
    d_ff: int = 1536
    n_layers: int = 3
    n_heads: int = 4
    max_position_embeddings: int = 512
    seq_len: int = 256
    batch_size: int = 32
    n_steps: int = 12000
    lr: float = 6e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    val_every: int = 1000


def smoke_cfg():
    c = Cfg(vocab_size=64, d_model=64, d_ff=128, n_layers=2, n_heads=4,
            max_position_embeddings=128, seq_len=32, batch_size=8, n_steps=40,
            val_every=20)
    return c


# ======================================================================
# Parameter-matched causal self-attention mixer (drop-in for InductionMixer)
#   same submodules: norm, q, k, v, o (d->d, no bias), sink_k, sink_v (H,dh)
#   forward(x, cand) — cand ignored (interface parity)
# ======================================================================
class AttentionMixer(nn.Module):
    def __init__(self, d, H, M=None):
        super().__init__()
        self.H = H; self.dh = d // H
        self.norm = RMSNorm(d)
        self.q = nn.Linear(d, d, bias=False); self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False); self.o = nn.Linear(d, d, bias=False)
        self.sink_k = nn.Parameter(torch.randn(H, self.dh) * 0.02)
        self.sink_v = nn.Parameter(torch.randn(H, self.dh) * 0.02)

    def forward(self, x, cand=None):
        B, T, d = x.shape; H, dh = self.H, self.dh; dev = x.device
        h = self.norm(x)
        q = self.q(h).view(B, T, H, dh).transpose(1, 2)            # (B,H,T,dh)
        k = self.k(h).view(B, T, H, dh).transpose(1, 2)
        v = self.v(h).view(B, T, H, dh).transpose(1, 2)
        sk = self.sink_k[None, :, None, :].expand(B, H, 1, dh)
        sv = self.sink_v[None, :, None, :].expand(B, H, 1, dh)
        k = torch.cat([sk, k], 2); v = torch.cat([sv, v], 2)       # sink at index 0
        s = (q @ k.transpose(-2, -1)) / math.sqrt(dh)              # (B,H,T,T+1)
        ti = torch.arange(T, device=dev)
        causal = (ti[None, :] <= ti[:, None])                      # (T,T)
        mask = torch.cat([torch.ones(T, 1, dtype=torch.bool, device=dev), causal], 1)
        s = s.masked_fill(~mask[None, None], NEG)
        out = (F.softmax(s, -1) @ v).transpose(1, 2).reshape(B, T, d)
        return self.o(out)


def build_model(cfg: Cfg, arm: str, seed: int):
    torch.manual_seed(seed); random.seed(seed)
    ic = InductionConfig(vocab_size=cfg.vocab_size, d_model=cfg.d_model, d_ff=cfg.d_ff,
                         n_layers=cfg.n_layers, n_heads=cfg.n_heads,
                         max_position_embeddings=cfg.max_position_embeddings)
    model = InductionForCausalLM(ic)
    if arm == "attention":
        for blk in model.model.layers:
            blk.mixer = AttentionMixer(cfg.d_model, cfg.n_heads)
    model.apply(model._init_weights)                               # identical init scheme, both arms
    return model.to(DEVICE)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build_optimizer(model, cfg):
    # Shared recipe across arms. Swap in Muon here to match production exactly.
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                             weight_decay=cfg.weight_decay, betas=(0.9, 0.95))


# ======================================================================
# data
# ======================================================================
def load_data(cfg, tok):
    if SMOKE:
        g = torch.Generator().manual_seed(123)
        ids = torch.randint(2, cfg.vocab_size, (200 * cfg.seq_len,), generator=g)
        n = int(0.9 * ids.numel())
        return ids[:n].to(DEVICE), ids[n:].to(DEVICE)
    from datasets import load_dataset
    ds = load_dataset("BabyLM-community/BabyLM-2026-Strict-Small")
    sp = list(ds.keys()); tr = "train" if "train" in sp else sp[0]
    col = next(k for k in ["text", "content", "document", "raw"] if k in ds[tr][0])
    text = "\n".join(t for t in ds[tr][col] if t and t.strip())
    enc = tok(text, return_attention_mask=False)["input_ids"] if tok else None
    if enc is None:
        raise RuntimeError("no tokenizer; set HF_MODEL_ID to reuse the production tokenizer")
    ids = torch.tensor(enc, dtype=torch.long, device=DEVICE)
    n = int(0.95 * ids.numel())
    return ids[:n], ids[n:]


def get_batch(ids, cfg):
    n = ids.size(0) - cfg.seq_len - 1
    si = torch.randint(0, n, (cfg.batch_size,), device=ids.device)
    off = torch.arange(cfg.seq_len + 1, device=ids.device)
    seq = ids[si[:, None] + off[None, :]]
    return seq[:, :cfg.seq_len].contiguous(), seq[:, 1:].contiguous()


@torch.no_grad()
def val_ppl(model, ids, cfg, n_batches=20):
    model.eval(); losses = []
    for _ in range(n_batches):
        x, y = get_batch(ids, cfg)
        out = model(input_ids=x)
        losses.append(F.cross_entropy(out.logits.reshape(-1, cfg.vocab_size), y.reshape(-1)).item())
    model.train(); return math.exp(sum(losses) / len(losses))


@torch.no_grad()
def verify_causality(model, cfg):
    model.eval(); T = cfg.seq_len; cut = T // 2
    x = torch.randint(2, cfg.vocab_size, (2, T), device=DEVICE)
    l1 = model(input_ids=x).logits
    x2 = x.clone(); x2[:, cut:] = torch.randint(2, cfg.vocab_size, (2, T - cut), device=DEVICE)
    l2 = model(input_ids=x2).logits
    diff = (l1[:, :cut] - l2[:, :cut]).abs().max().item()
    model.train(); return diff


def train_one(cfg, arm, seed, train_ids, val_ids, tok, outdir):
    model = build_model(cfg, arm, seed)
    npar = count_params(model)
    cd = verify_causality(model, cfg)
    assert cd < 1e-3, f"{arm} not causal (diff {cd:.2e})"
    opt = build_optimizer(model, cfg)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.n_steps, eta_min=cfg.lr * 0.05)
    best, best_dir = float("inf"), None
    model.train()
    for step in range(1, cfg.n_steps + 1):
        x, y = get_batch(train_ids, cfg)
        out = model(input_ids=x)
        loss = F.cross_entropy(out.logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step(); sch.step()
        if step % cfg.val_every == 0 or step == cfg.n_steps:
            vp = val_ppl(model, val_ids, cfg)
            if vp < best:
                best = vp; best_dir = os.path.join(outdir, f"{arm}_seed{seed}")
                if not SMOKE:
                    model.save_pretrained(best_dir)
                    if tok is not None: tok.save_pretrained(best_dir)
            print(f"    [{arm} s{seed}] step {step:5d}  val ppl {vp:7.2f}  (best {best:7.2f})")
    blimp = None
    if BLIMP_HOOK is not None:
        blimp = BLIMP_HOOK(model, tok, DEVICE)
    return dict(arm=arm, seed=seed, params=npar, best_ppl=best, blimp=blimp, dir=best_dir)


def main():
    cfg = smoke_cfg() if SMOKE else Cfg()
    tok = None
    if REPO and not SMOKE:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
        cfg.vocab_size = tok.vocab_size
    train_ids, val_ids = load_data(cfg, tok)
    outdir = os.environ.get("OUTDIR", "/kaggle/working/c1_runs" if not SMOKE else "/tmp/c1_smoke")
    os.makedirs(outdir, exist_ok=True)

    # param-parity check (the control's credibility)
    pi = count_params(build_model(cfg, "induction", 0))
    pa = count_params(build_model(cfg, "attention", 0))
    print(f"param parity:  induction {pi:,}  vs  attention {pa:,}  (Δ={pi-pa:+,})")

    results = []
    for arm in ["induction", "attention"]:
        for seed in SEEDS:
            print(f"\n>>> arm={arm} seed={seed}")
            results.append(train_one(cfg, arm, seed, train_ids, val_ids, tok, outdir))

    print("\n" + "=" * 64)
    print(f"{'arm':<11}{'seed':>5}{'params':>12}{'best ppl':>11}{'BLiMP':>9}")
    print("-" * 64)
    for r in results:
        bl = f"{r['blimp']:.2f}" if r['blimp'] is not None else "  (run eval)"
        print(f"{r['arm']:<11}{r['seed']:>5}{r['params']:>12,}{r['best_ppl']:>11.2f}{bl:>9}")
    print("-" * 64)
    for arm in ["induction", "attention"]:
        ps = torch.tensor([r["best_ppl"] for r in results if r["arm"] == arm])
        print(f"  {arm:<11} val ppl  mean {ps.mean():.2f}  std {ps.std():.2f}  (n={ps.numel()})")
    if not SMOKE:
        print(f"\nCheckpoints in {outdir}/<arm>_seed<k>/  — run babylm-eval on each for filtered BLiMP.")

    if SMOKE:
        assert pi == pa, f"param mismatch {pi} vs {pa} — AttentionMixer not matched"
        assert all(math.isfinite(r["best_ppl"]) for r in results), "non-finite ppl"
        print("\nSMOKE OK: param parity exact; both arms train, stay causal, produce finite ppl.")


if __name__ == "__main__":
    main()

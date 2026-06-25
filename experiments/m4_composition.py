"""
m4_composition.py — does the circuit COMPOSE across layers?  (H2, part 3)
=========================================================================

The look-back index is computed once from the literal tokens and shared by every layer, so
"composition" cannot mean re-routing edges across layers. It means a deeper layer reading a
source position whose residual a *lower* layer has already enriched. The clean test:

  2-hop chain (one sequence):   ... X B ...... A X ......  [query A] -> target B
    * query A's edge -> the prior A (in "A X"); the copied source is that A's continuation = X2.
    * to answer B, layer 1 must first write B into X2's residual (via X2's OWN edge back to the
      earlier X1, whose continuation is B); layer 2's A-edge then transports that enriched
      residual. => requires >= 2 layers.
  1-hop chain (control):        ... A z ...... [query A] -> target z   (solvable at depth 1)

We train the induction architecture at depth in {1,2,3} on a mix of both, then measure
top-1 accuracy at the query position.

  PREDICTION : 1-hop solved at every depth; 2-hop solved only at depth >= 2
               (and at depth 1 the model falls back to the 1-hop answer X, not B).
  FALSIFIER  : depth 1 already solves 2-hop (no composition needed), OR depth>=2 cannot
               (the architecture cannot compose) -> H2 composition claim fails.

Run:  python m4_composition.py        |        SMOKE=1 python m4_composition.py
"""

import os, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling_induction import InductionConfig, InductionForCausalLM

SMOKE = bool(os.environ.get("SMOKE"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VOCAB = 64 if SMOKE else 128
T = 24 if SMOKE else 48
BATCH = 16 if SMOKE else 64
STEPS = 80 if SMOKE else 4000
DEPTHS = [1, 2] if SMOKE else [1, 2, 3]
N_SYMBOLS = 3                       # A, X, B drawn from the "symbol" band; filler from the rest
SEED = 0


def _pools(vocab, g):
    """Disjoint symbol band (A/X/B/z) and filler pool so structural tokens never collide."""
    perm = torch.randperm(vocab - 2, generator=g) + 2     # skip 0,1 (pad/unk)
    return perm[:8], perm[8:]                              # (symbols, filler)


def gen_2hop(batch, T, vocab, g):
    """... X1 B ... A X2 ... [A] -> B.  Returns x, qpos, target_B, fallback_X."""
    x = torch.zeros(batch, T, dtype=torch.long)
    qpos = torch.full((batch,), T - 1, dtype=torch.long)
    tgtB = torch.zeros(batch, dtype=torch.long)
    fbX = torch.zeros(batch, dtype=torch.long)
    for b in range(batch):
        syms, filler = _pools(vocab, g)
        A, X, B = syms[0].item(), syms[1].item(), syms[2].item()
        x[b] = filler[torch.randint(0, filler.numel(), (T,), generator=g)]
        p1 = torch.randint(1, T // 3, (1,), generator=g).item()           # X1 B
        p2 = torch.randint(T // 2, 2 * T // 3, (1,), generator=g).item()  # A  X2
        x[b, p1] = X; x[b, p1 + 1] = B
        x[b, p2] = A; x[b, p2 + 1] = X
        x[b, T - 1] = A                                                   # query
        tgtB[b] = B; fbX[b] = X
    return x.to(DEVICE), qpos.to(DEVICE), tgtB.to(DEVICE), fbX.to(DEVICE)


def gen_1hop(batch, T, vocab, g):
    """... A z ... [A] -> z."""
    x = torch.zeros(batch, T, dtype=torch.long)
    qpos = torch.full((batch,), T - 1, dtype=torch.long)
    tgt = torch.zeros(batch, dtype=torch.long)
    for b in range(batch):
        syms, filler = _pools(vocab, g)
        A, z = syms[0].item(), syms[1].item()
        x[b] = filler[torch.randint(0, filler.numel(), (T,), generator=g)]
        p = torch.randint(T // 3, 2 * T // 3, (1,), generator=g).item()
        x[b, p] = A; x[b, p + 1] = z
        x[b, T - 1] = A
        tgt[b] = z
    return x.to(DEVICE), qpos.to(DEVICE), tgt.to(DEVICE)


def build(depth):
    torch.manual_seed(SEED)
    cfg = InductionConfig(vocab_size=VOCAB, d_model=64 if SMOKE else 128,
                          d_ff=128 if SMOKE else 256, n_layers=depth, n_heads=4,
                          max_position_embeddings=T, match_m=5,
                          conv_dilations=(1, 2, 4), conv_kernel=3)
    return InductionForCausalLM(cfg).to(DEVICE)


@torch.no_grad()
def acc_at_query(model, x, qpos, target):
    model.eval()
    logits = model(input_ids=x).logits
    pred = logits[torch.arange(x.size(0), device=DEVICE), qpos].argmax(-1)
    model.train()
    return (pred == target).float().mean().item()


def train_depth(depth, g):
    model = build(depth)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4, betas=(0.9, 0.95))
    half = BATCH // 2
    for step in range(1, STEPS + 1):
        x1, q1, t1 = gen_1hop(half, T, VOCAB, g)
        x2, q2, t2, _ = gen_2hop(half, T, VOCAB, g)
        x = torch.cat([x1, x2], 0)
        # next-token LM loss over all positions (the query supervision is the last position)
        logits = model(input_ids=x[:, :-1]).logits
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    # eval on fresh chains
    x1, q1, t1 = gen_1hop(256 if not SMOKE else 32, T, VOCAB, g)
    x2, q2, tB, fX = gen_2hop(256 if not SMOKE else 32, T, VOCAB, g)
    a1 = acc_at_query(model, x1, q1, t1)
    aB = acc_at_query(model, x2, q2, tB)        # 2-hop success (answers B)
    aX = acc_at_query(model, x2, q2, fX)        # 1-hop fallback (answers X)
    return dict(depth=depth, params=sum(p.numel() for p in model.parameters()),
                onehop=a1, twohop_B=aB, twohop_Xfallback=aX)


def main():
    g = torch.Generator().manual_seed(SEED)
    print(f"VOCAB={VOCAB} T={T} STEPS={STEPS} depths={DEPTHS} device={DEVICE}")
    print(f"\n{'depth':>5} {'params':>9} {'1-hop acc':>10} {'2-hop->B':>9} {'2-hop->X(fallback)':>19}")
    print("-" * 60)
    res = []
    for d in DEPTHS:
        r = train_depth(d, g)
        res.append(r)
        print(f"{r['depth']:>5} {r['params']:>9,} {r['onehop']:>10.3f} "
              f"{r['twohop_B']:>9.3f} {r['twohop_Xfallback']:>19.3f}")
    print("-" * 60)
    print("expect: 1-hop solved at all depths; 2-hop->B jumps at depth>=2; depth-1 falls back to X.")

    if SMOKE:
        assert all(0 <= r['onehop'] <= 1 and 0 <= r['twohop_B'] <= 1 for r in res), "acc out of range"
        assert all(math.isfinite(r['twohop_B']) for r in res), "non-finite acc"
        print("\nSMOKE OK: task generators + depth sweep run; accuracies finite and in range.")


if __name__ == "__main__":
    main()

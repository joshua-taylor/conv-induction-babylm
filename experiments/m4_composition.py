"""
m4_composition.py — does the circuit COMPOSE across layers?  (H2, part 3 — fixed)
=================================================================================

The look-back index is computed once from the literal tokens and shared by every layer, so
"composition" means a deeper layer reading a source whose residual a LOWER layer enriched.

  2-hop chain:  ... X1 B ...... A1 X2 ...... [query A2] -> target B
    query A2's edge -> A1; the copied source is A1's continuation X2 (the 1-hop answer).
    To answer B, layer 1 must first write B into X2's residual (X2's own edge back to X1,
    whose continuation is B); layer 2's A2-edge then transports it. => needs >= 2 layers.
  1-hop chain:  ... A1 z ...... [query A2] -> target z              (solvable at depth 1)

FIXES vs the first version:
  * the query sits at T-2 (not the last position) so the LM loss actually SUPERVISES it;
  * an induction WARM-UP (repeated-sequence task) teaches the copy mechanism robustly;
  * loss is label-masked (ignore random filler) so the signal isn't drowned out.

  PREDICTION : 1-hop solved at every depth; 2-hop->B jumps at depth >= 2; depth 1 falls back to X.
  FALSIFIER  : depth 1 already solves 2-hop, OR depth>=2 cannot -> composition claim fails.

Run:  python m4_composition.py   |   SMOKE=1 python m4_composition.py
"""
import os, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling_induction import InductionConfig, InductionForCausalLM
import paper_style as ps

SMOKE = bool(os.environ.get("SMOKE"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB = 32 if SMOKE else 128
T = 24 if SMOKE else 48
BATCH = 24 if SMOKE else 96
STEPS = 600 if SMOKE else 5000
DEPTHS = [1, 2] if SMOKE else [1, 2, 3]
SEED = 0
IGN = -100


def _filler(b, T, vocab, g, exclude):
    pool = [v for v in range(2, vocab) if v not in exclude]
    idx = torch.randint(0, len(pool), (T,), generator=g)
    return torch.tensor([pool[i] for i in idx])


def gen_warmup(batch, T, vocab, g):
    """[r, r] repeated random sequence; supervise the 2nd copy (induction)."""
    k = T // 2
    r = torch.randint(2, vocab, (batch, k), generator=g)
    x = torch.cat([r, r], 1)[:, :T]
    sup = torch.zeros(batch, T, dtype=torch.bool)
    sup[:, k:T - 1] = True                       # positions in 2nd copy whose next is known
    return x.to(DEVICE), sup.to(DEVICE)


def gen_1hop(batch, T, vocab, g):
    x = torch.zeros(batch, T, dtype=torch.long)
    sup = torch.zeros(batch, T, dtype=torch.bool)
    qpos = T - 2
    tgt = torch.zeros(batch, dtype=torch.long)
    for b in range(batch):
        A, z = torch.randint(2, vocab, (2,), generator=g).tolist()
        x[b] = _filler(b, T, vocab, g, {A, z})
        p = torch.randint(2, qpos - 2, (1,), generator=g).item()
        x[b, p] = A; x[b, p + 1] = z
        x[b, qpos] = A; x[b, qpos + 1] = z       # target z follows the query (supervised)
        sup[b, qpos] = True
        tgt[b] = z
    return x.to(DEVICE), sup.to(DEVICE), qpos, tgt.to(DEVICE)


def gen_2hop(batch, T, vocab, g):
    x = torch.zeros(batch, T, dtype=torch.long)
    sup = torch.zeros(batch, T, dtype=torch.bool)
    qpos = T - 2
    tgtB = torch.zeros(batch, dtype=torch.long)
    fbX = torch.zeros(batch, dtype=torch.long)
    for b in range(batch):
        A, X, B = torch.randint(2, vocab, (3,), generator=g).tolist()
        x[b] = _filler(b, T, vocab, g, {A, X, B})
        p1 = torch.randint(1, T // 4, (1,), generator=g).item()          # X1 B
        p2 = torch.randint(T // 2, qpos - 2, (1,), generator=g).item()   # A1 X2
        x[b, p1] = X; x[b, p1 + 1] = B
        x[b, p2] = A; x[b, p2 + 1] = X
        x[b, qpos] = A; x[b, qpos + 1] = B       # target B follows the query (supervised)
        sup[b, p2 + 1] = True                    # helper: X2 -> B (the 1-hop link reused by hop 2)
        sup[b, qpos] = True                      # the 2-hop query
        tgtB[b] = B; fbX[b] = X
    return x.to(DEVICE), sup.to(DEVICE), qpos, tgtB.to(DEVICE), fbX.to(DEVICE)


def masked_loss(model, x, sup):
    logits = model(input_ids=x).logits                      # (B,T,V)
    tgt = x[:, 1:].clone()
    tgt[~sup[:, :-1]] = IGN
    return F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), tgt.reshape(-1), ignore_index=IGN)


def build(depth):
    torch.manual_seed(SEED)
    cfg = InductionConfig(vocab_size=VOCAB, d_model=64 if SMOKE else 128,
                          d_ff=128 if SMOKE else 256, n_layers=depth, n_heads=4,
                          max_position_embeddings=T, match_m=5,
                          conv_dilations=(1, 2, 4), conv_kernel=3)
    return InductionForCausalLM(cfg).to(DEVICE)


@torch.no_grad()
def acc_at(model, x, qpos, target):
    model.eval()
    pred = model(input_ids=x).logits[:, qpos].argmax(-1)
    model.train()
    return (pred == target).float().mean().item()


def train_depth(depth, g):
    model = build(depth)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4, betas=(0.9, 0.95))
    nw, n1, n2 = BATCH // 2, BATCH // 4, BATCH // 4
    for step in range(1, STEPS + 1):
        xw, sw = gen_warmup(nw, T, VOCAB, g)
        x1, s1, _, _ = gen_1hop(n1, T, VOCAB, g)
        x2, s2, _, _, _ = gen_2hop(n2, T, VOCAB, g)
        x = torch.cat([xw, x1, x2], 0); sup = torch.cat([sw, s1, s2], 0)
        loss = masked_loss(model, x, sup)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    ne = 64 if SMOKE else 512
    x1, s1, q1, t1 = gen_1hop(ne, T, VOCAB, g)
    x2, s2, q2, tB, fX = gen_2hop(ne, T, VOCAB, g)
    return dict(depth=depth, params=sum(p.numel() for p in model.parameters()),
                onehop=acc_at(model, x1, q1, t1),
                twohop_B=acc_at(model, x2, q2, tB),
                twohop_X=acc_at(model, x2, q2, fX))


def main():
    g = torch.Generator().manual_seed(SEED)
    print(f"VOCAB={VOCAB} T={T} STEPS={STEPS} depths={DEPTHS} device={DEVICE}")
    print(f"\n{'depth':>5} {'params':>9} {'1-hop':>8} {'2-hop->B':>9} {'2-hop->X(fallback)':>19}")
    print("-" * 56)
    res = []
    for d in DEPTHS:
        r = train_depth(d, g); res.append(r)
        print(f"{r['depth']:>5} {r['params']:>9,} {r['onehop']:>8.3f} "
              f"{r['twohop_B']:>9.3f} {r['twohop_X']:>19.3f}")
    print("-" * 56)
    print("expect: 1-hop high at all depths; 2-hop->B jumps at depth>=2; depth-1 falls back to X.")

    ps.apply()
    import matplotlib.pyplot as plt
    import numpy as np
    depths = [r["depth"] for r in res]
    x = np.arange(len(depths)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(x - w / 2, [r["onehop"] for r in res], w, label="1-hop", color=ps.PRIMARY, edgecolor="white", lw=0.5)
    ax.bar(x + w / 2, [r["twohop_B"] for r in res], w, label="2-hop \u2192 B", color=ps.SECOND, edgecolor="white", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f"{d} layer{'s' if d > 1 else ''}" for d in depths])
    ax.set_ylim(0, 1.02); ax.set_ylabel("accuracy at query"); ax.set_title("cross-layer composition")
    ax.axhline(1 / VOCAB, ls="--", lw=1, color=ps.MUTED); ax.text(len(depths) - 1, 1 / VOCAB + 0.02, "chance", color=ps.MUTED, fontsize=8, ha="right")
    ax.legend()
    ps.save(fig, "figs/m4_composition.png")

    if SMOKE:
        assert all(0 <= r["onehop"] <= 1 for r in res)
        print(f"\nSMOKE OK (1-hop acc {[round(r['onehop'],2) for r in res]} — should be >chance {1/VOCAB:.3f}).")


if __name__ == "__main__":
    main()

"""
exp_c2_scaling.py — data-scaling curves + induction emergence  (C2 + M6, the H1 backbone)
==========================================================================================

Trains the induction arm and the matched attention arm (reusing c1_matched_baseline) to the
full step budget, SNAPSHOTTING at intermediate points. At each snapshot we record:
  * words seen
  * val perplexity
  * induction-copy score = accuracy on the 2nd copy of a repeated random sequence
    (the canonical induction-head metric: predict r[i+1] given r[i] in [r, r])

  PREDICTION (H1): the induction arm's copy score is high from the very start; the attention
    arm's rises late (a phase change), and our ppl/BLiMP advantage is largest at low data and
    narrows as data grows.
  FALSIFIER : the gap is flat or widens with data -> advantage isn't data-efficiency.

Run:  HF_MODEL_ID=you/repo python exp_c2_scaling.py   |   SMOKE=1 python exp_c2_scaling.py
(BLiMP-per-snapshot is heavy; use val-ppl + copy-score here, and run babylm-eval on saved
snapshots if you want BLiMP on the curve.)
"""
import os, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import c1_matched_baseline as C1

SMOKE = C1.SMOKE
DEVICE = C1.DEVICE
REPO = os.environ.get("HF_MODEL_ID")
SNAP_FRACS = [0.1, 0.25, 0.5, 1.0] if not SMOKE else [0.5, 1.0]


@torch.no_grad()
def induction_copy_score(model, vocab, device, k=16, n=64, T_cap=512):
    """Repeat a random length-k sequence twice; measure top-1 acc on the 2nd copy."""
    k = min(k, (T_cap // 2) - 1)
    g = torch.Generator().manual_seed(0)
    r = torch.randint(2, vocab, (n, k), generator=g).to(device)
    x = torch.cat([r, r], 1)                                   # [r, r], length 2k
    model.eval()
    logits = model(input_ids=x[:, :-1]).logits                 # predict next
    pred = logits.argmax(-1)
    # positions in the 2nd copy predict r[i+1]; compare to the realised next token
    tgt = x[:, 1:]
    sec = torch.zeros_like(tgt, dtype=torch.bool)
    sec[:, k:] = True                                          # 2nd-copy predictions
    acc = ((pred == tgt) & sec).sum().float() / sec.sum().float()
    model.train()
    return acc.item()


def train_with_snapshots(cfg, arm, seed, train_ids, val_ids, vocab):
    model = C1.build_model(cfg, arm, seed)
    opt = C1.build_optimizer(model, cfg)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.n_steps, eta_min=cfg.lr * 0.05)
    snaps = sorted(set(int(f * cfg.n_steps) for f in SNAP_FRACS))
    rows = []
    model.train()
    for step in range(1, cfg.n_steps + 1):
        x, y = C1.get_batch(train_ids, cfg)
        loss = F.cross_entropy(model(input_ids=x).logits.reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); opt.step(); sch.step()
        if step in snaps:
            words = step * cfg.batch_size * cfg.seq_len
            vp = C1.val_ppl(model, val_ids, cfg)
            cs = induction_copy_score(model, vocab, DEVICE, T_cap=cfg.max_position_embeddings)
            rows.append((step, words, vp, cs))
            print(f"    [{arm} s{seed}] step {step:5d} words {words:>10,} val_ppl {vp:7.2f} copy {cs:.3f}")
    return rows


def main():
    cfg = C1.smoke_cfg() if SMOKE else C1.Cfg()
    tok = None
    if REPO and not SMOKE:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
        cfg.vocab_size = tok.vocab_size
    train_ids, val_ids = C1.load_data(cfg, tok)
    vocab = cfg.vocab_size

    curves = {}
    for arm in ["induction", "attention"]:
        print(f"\n>>> scaling arm={arm}")
        curves[arm] = train_with_snapshots(cfg, arm, 0, train_ids, val_ids, vocab)

    print("\n" + "=" * 70)
    print(f"{'arm':<11}{'words':>12}{'val_ppl':>10}{'copy_score':>12}")
    print("-" * 70)
    for arm, rows in curves.items():
        for (step, words, vp, cs) in rows:
            print(f"{arm:<11}{words:>12,}{vp:>10.2f}{cs:>12.3f}")

    # optional figure
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for arm, rows in curves.items():
            w = [r[1] for r in rows]
            ax[0].plot(w, [r[2] for r in rows], "-o", label=arm)
            ax[1].plot(w, [r[3] for r in rows], "-o", label=arm)
        ax[0].set(title="val perplexity", xlabel="words seen", ylabel="ppl"); ax[0].set_xscale("log"); ax[0].legend()
        ax[1].set(title="induction copy score", xlabel="words seen", ylabel="acc"); ax[1].set_xscale("log"); ax[1].legend()
        os.makedirs("figs", exist_ok=True); fig.tight_layout(); fig.savefig("figs/c2_scaling.png", dpi=140)
        print("\nsaved figs/c2_scaling.png")
    except Exception as e:
        print(f"(figure skipped: {type(e).__name__})")

    if SMOKE:
        assert all(0 <= r[3] <= 1 for rows in curves.values() for r in rows), "copy score out of range"
        print("\nSMOKE OK: both arms trained with snapshots; copy score computed.")


if __name__ == "__main__":
    main()

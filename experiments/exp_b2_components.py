"""
exp_b2_components.py — division of labour  (B2)
===============================================

Trains the architecture with each component knocked out (replaced by a zero sub-layer, so the
residual passes through) to show none is redundant:
    full | -conv | -induction | -ffn
Same recipe/seed as c1_matched_baseline; reports val ppl per arm.

  PREDICTION : removing conv OR induction each hurts clearly (local order and exact recall are
               complementary); removing ffn hurts too.
  FALSIFIER  : a component can be removed at no cost -> it isn't pulling its weight.

Run:  HF_MODEL_ID=you/repo python exp_b2_components.py   |   SMOKE=1 python exp_b2_components.py
"""
import os, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import c1_matched_baseline as C1

SMOKE = C1.SMOKE
DEVICE = C1.DEVICE
REPO = os.environ.get("HF_MODEL_ID")
ARMS = ["full", "-conv", "-induction", "-ffn"]


class Zero(nn.Module):
    """Returns zeros shaped like the residual input (accepts the extra `cand` arg too)."""
    def forward(self, x, *args, **kw):
        return torch.zeros_like(x)


def build_ablated(cfg, arm, seed):
    model = C1.build_model(cfg, "induction", seed)            # full induction model
    if arm == "full":
        return model
    for blk in model.model.layers:
        if arm == "-conv":
            blk.conv = Zero()
        elif arm == "-induction":
            blk.mixer = Zero()
        elif arm == "-ffn":
            blk.ffn = Zero()
    return model.to(DEVICE)


def train_arm(cfg, arm, seed, train_ids, val_ids, vocab):
    model = build_ablated(cfg, arm, seed)
    opt = C1.build_optimizer(model, cfg)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.n_steps, eta_min=cfg.lr * 0.05)
    model.train()
    best = float("inf")
    for step in range(1, cfg.n_steps + 1):
        x, y = C1.get_batch(train_ids, cfg)
        loss = F.cross_entropy(model(input_ids=x).logits.reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); opt.step(); sch.step()
        if step % cfg.val_every == 0 or step == cfg.n_steps:
            best = min(best, C1.val_ppl(model, val_ids, cfg))
    return dict(arm=arm, best_ppl=best,
                params=sum(p.numel() for p in model.parameters() if p.requires_grad))


def main():
    cfg = C1.smoke_cfg() if SMOKE else C1.Cfg()
    tok = None
    if REPO and not SMOKE:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
        cfg.vocab_size = tok.vocab_size
    train_ids, val_ids = C1.load_data(cfg, tok)
    res = []
    for arm in ARMS:
        print(f">>> training {arm}")
        res.append(train_arm(cfg, arm, 0, train_ids, val_ids, cfg.vocab_size))
    print("\n" + "=" * 48)
    print(f"{'arm':<14}{'params':>12}{'best val ppl':>14}")
    print("-" * 48)
    base = next(r["best_ppl"] for r in res if r["arm"] == "full")
    for r in res:
        delta = "" if r["arm"] == "full" else f"  ({r['best_ppl']-base:+.2f})"
        print(f"{r['arm']:<14}{r['params']:>12,}{r['best_ppl']:>14.2f}{delta}")
    if SMOKE:
        assert all(math.isfinite(r["best_ppl"]) for r in res)
        print("\nSMOKE OK: all ablation arms train and produce finite ppl.")


if __name__ == "__main__":
    main()

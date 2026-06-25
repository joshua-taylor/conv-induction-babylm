"""
exp_m1_edge_patching.py — does information flow ALONG the look-back edges?  (H2, part 1)
=======================================================================================

Design (single-variable, pre-registered):
  For query t whose strongest long-range edge points to source s (continuation s+1, copied
  token y* = x[s+1]), corrupt the mixer's input residual at s+1 (mean-ablation) and measure
  how much the query's prediction of y* drops. Compare against corrupting a *distance-matched
  control* position that is NOT one of t's sources.

  Because the edge is long-range (>= MIN_DIST, beyond the conv's reach), the only path from
  s+1 to t is the induction edge, so any effect on t is edge-mediated.

  PREDICTION : Delta logp(y*) under edge-patch >> under control-patch  (flow along the edge).
  FALSIFIER  : edge effect ~ control effect ~ 0                        (no flow; H2 wrong).

Metrics per selected (b,t):
  * d_logp_edge / d_logp_ctrl  = logp_clean(y*) - logp_patched(y*)     (drop in copied-token logprob)
  * l2_edge / l2_ctrl          = ||logits_patched[t] - logits_clean[t]||  (total shift; harness sanity)

Run:  python exp_m1_edge_patching.py            (needs HF_MODEL_ID env or edit REPO)
      SMOKE=1 python exp_m1_edge_patching.py    (tiny CPU harness test)
"""

import os, math
import torch
import torch.nn.functional as F
import mech_common as MC

REPO = os.environ.get("HF_MODEL_ID")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER = int(os.environ.get("LAYER", "-1"))       # which layer's mixer input to patch (-1 = last)
MIN_DIST = int(os.environ.get("MIN_DIST", "6" if MC.SMOKE else "48"))
T = 32 if MC.SMOKE else 256
B = 8 if MC.SMOKE else 16
N_BATCHES = 2 if MC.SMOKE else 40


def logits_at(model, input_ids, capture=None, patch=None):
    ctx_cap = capture if capture is not None else _nullctx()
    ctx_patch = patch if patch is not None else _nullctx()
    with torch.no_grad(), ctx_cap, ctx_patch:
        out = model(input_ids=input_ids)
    return out.logits


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def run():
    model, tok, cfg = MC.load_model(REPO, DEVICE)
    n_layers = len(MC.get_layers(model))
    layer = LAYER if LAYER >= 0 else n_layers - 1
    print(f"model: {n_layers} layers, d={cfg.d_model}, vocab={cfg.vocab_size} | patch layer {layer} "
          f"| MIN_DIST={MIN_DIST} | device={DEVICE}")

    # data source
    if REPO and tok is not None and not MC.SMOKE:
        from datasets import load_dataset
        ds = load_dataset("BabyLM-community/BabyLM-2026-Strict-Small", split="train", streaming=True)
        texts = [r.get("text") or r.get("content") or "" for r, _ in zip(ds, range(2000))]
        data = MC.tokenize_corpus(tok, texts, T, max_seqs=B * N_BATCHES, device=DEVICE)
    else:
        data = torch.cat([MC.synthetic_induction_batch(B, T, cfg.vocab_size, DEVICE, seed=i)
                          for i in range(N_BATCHES)], 0)

    rows = []  # (d_logp_edge, d_logp_ctrl, l2_edge, l2_ctrl)
    for bi in range(0, data.size(0), B):
        x = data[bi:bi + B]
        if x.size(0) < 2:
            continue
        cap = MC.MixerCapture(model)
        clean = logits_at(model, x, capture=cap)
        xin = cap.x[layer]
        cand = cap.cand
        mixer = MC.get_layers(model)[layer].mixer
        _, mass, ok, _ = MC.recompute_mixer_S(mixer, xin, cand)
        sel = MC.select_top_edges(mass, cand, ok, MIN_DIST)
        if sel["b"].numel() == 0:
            continue
        b, t, cont = sel["b"], sel["t"], sel["cont"]
        cont = cont.clamp(0, T - 1)
        y_star = x[b, cont]                                       # copied token at the query
        ctrl = MC.distance_matched_control(x, b, t, cont, cand)
        repl = xin.reshape(-1, xin.size(-1)).mean(0)             # mean residual (position-agnostic)

        with MC.PatchMixerInput(model, layer, b, cont, repl):
            edge = logits_at(model, x)
        with MC.PatchMixerInput(model, layer, b, ctrl, repl):
            ctl = logits_at(model, x)

        lp_clean = F.log_softmax(clean[b, t], -1).gather(1, y_star[:, None]).squeeze(1)
        lp_edge = F.log_softmax(edge[b, t], -1).gather(1, y_star[:, None]).squeeze(1)
        lp_ctrl = F.log_softmax(ctl[b, t], -1).gather(1, y_star[:, None]).squeeze(1)
        l2_e = (edge[b, t] - clean[b, t]).norm(dim=-1)
        l2_c = (ctl[b, t] - clean[b, t]).norm(dim=-1)
        for i in range(b.numel()):
            rows.append((float(lp_clean[i] - lp_edge[i]), float(lp_clean[i] - lp_ctrl[i]),
                         float(l2_e[i]), float(l2_c[i])))

    if not rows:
        print("No qualifying long-range edges found (try lowering MIN_DIST)."); return
    r = torch.tensor(rows)
    de, dc, le, lc = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
    print(f"\nselected edges: {len(rows)}")
    print(f"  Delta logp(y*)   edge  = {de.mean():+.4f} +/- {de.std():.4f}")
    print(f"  Delta logp(y*)   ctrl  = {dc.mean():+.4f} +/- {dc.std():.4f}")
    print(f"  edge - ctrl            = {(de - dc).mean():+.4f}   (>0 => flow along edge)")
    print(f"  P(edge drop > ctrl)    = {(de > dc).float().mean():.3f}")
    print(f"  ||Δlogits@t||  edge={le.mean():.4f}  ctrl={lc.mean():.4f}  ratio={le.mean()/lc.mean().clamp(min=1e-6):.2f}x")

    if MC.SMOKE:
        # harness sanity: corrupting the true source shifts the query MORE than a far control
        assert torch.isfinite(r).all(), "non-finite metrics"
        assert le.mean() > lc.mean(), f"edge L2 {le.mean():.4f} should exceed control {lc.mean():.4f}"
        print("\nSMOKE OK: edge patch shifts the query strictly more than the control.")


if __name__ == "__main__":
    run()

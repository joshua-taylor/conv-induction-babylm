"""
viz_v2_edge_arcs.py — the look-back graph, by layer  (V2, refined)
==================================================================

Per layer, an arc from each token to its top induction source. Refinements:
  * mass mapped to opacity AND thickness AND colour, normalised per layer for contrast
  * clean token labels (byte-level BPE artifacts like the leading-space marker stripped)
  * consistent paper styling
Renders figs/v2_edge_arcs.png. Uses the trained model if HF_MODEL_ID is set.

Run:  HF_MODEL_ID=you/repo python viz_v2_edge_arcs.py   |   SMOKE=1 python viz_v2_edge_arcs.py
"""
import os
import torch
import mech_common as MC
import paper_style as ps

REPO = os.environ.get("HF_MODEL_ID")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PASSAGE = ("the little girl saw the old boat and the girl wanted the old boat by the river "
           "so the girl sat in the old boat near the river")
MAXTOK = 40


def clean_labels(tok, ids):
    out = []
    for i in ids:
        s = tok.decode([int(i)])
        s = s.replace("\u0120", "").replace("\u010a", "").replace("\u2581", "").strip()
        out.append(s if s else "\u00b7")
    return out


def run():
    model, tok, cfg = MC.load_model(REPO, DEVICE)
    n_layers = len(MC.get_layers(model))

    if tok is not None and not MC.SMOKE:
        ids = tok(PASSAGE, return_attention_mask=False)["input_ids"][:MAXTOK]
        labels = clean_labels(tok, ids)
        x = torch.tensor([ids], device=DEVICE)
    else:
        x = MC.synthetic_induction_batch(1, 28, cfg.vocab_size, DEVICE, seed=3)
        labels = [str(int(t)) for t in x[0]]

    cap = MC.MixerCapture(model)
    with torch.no_grad(), cap:
        model(input_ids=x)

    per_layer = []
    for L in range(n_layers):
        with torch.no_grad():
            S, mass, ok, nxt = MC.recompute_mixer_S(MC.get_layers(model)[L].mixer, cap.x[L], cap.cand)
        m = mass[0].masked_fill(~ok[0], -1.0)
        top_c = m.argmax(-1)
        top_m = m.gather(1, top_c[:, None]).squeeze(1).clamp(min=0)
        src = cap.cand[0].gather(1, top_c[:, None]).squeeze(1)
        per_layer.append((src.cpu(), top_m.cpu(), ok[0].any(-1).cpu()))

    ps.apply()
    import matplotlib.pyplot as plt
    import numpy as np
    T = x.size(1)
    fig, axes = plt.subplots(n_layers, 1, figsize=(min(15, 0.42 * T + 2), 1.9 * n_layers), squeeze=False)
    cmap = plt.cm.YlOrRd
    for L in range(n_layers):
        ax = axes[L][0]
        src, mval, has = per_layer[L]
        mmax = max(float(mval[has].max()) if has.any() else 1.0, 1e-6)
        order = sorted(range(T), key=lambda t: float(mval[t]))   # weak first, strong on top
        for t in order:
            if has[t] and src[t] >= 0 and mval[t] > 0:
                s = int(src[t]); norm = float(mval[t]) / mmax
                cx = (s + t) / 2.0; rad = (t - s) / 2.0
                th = np.linspace(0, np.pi, 60)
                ax.plot(cx + rad * np.cos(th), np.sin(th) * (0.55 + 0.1 * norm),
                        color=cmap(0.35 + 0.6 * norm),
                        alpha=0.30 + 0.70 * norm, lw=0.7 + 3.8 * norm,
                        solid_capstyle="round", zorder=2 + norm)
        ax.scatter(range(T), [0] * T, s=14, color=ps.INK, zorder=5)
        ax.set_xlim(-0.8, T - 0.2); ax.set_ylim(-0.08, 0.8)
        ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"layer {L}", fontsize=11)
        if L == n_layers - 1:
            ax.set_xticks(range(T)); ax.set_xticklabels(labels, rotation=90, fontsize=7.5)
            ax.tick_params(length=0)
        else:
            ax.set_xticks([])
    fig.suptitle("top look-back edge per token  (thickness & shade = attention mass)",
                 x=0.012, ha="left", fontsize=13, weight="600")
    ps.save(fig, "figs/v2_edge_arcs.png")
    if MC.SMOKE:
        print("SMOKE OK: refined edge-arc figure produced.")


if __name__ == "__main__":
    run()

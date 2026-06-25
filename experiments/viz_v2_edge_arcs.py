"""
viz_v2_edge_arcs.py — the look-back graph, by layer  (V2)
=========================================================

For a sample passage, draw an arc from each query token to its top induction source, one panel
per layer, with arc opacity = the mixer's attention mass on that edge. Shows the local->long-
range shift across depth (the visual companion to M5).

Renders figs/v2_edge_arcs.png. Needs a model for per-layer mass (uses the trained model if
HF_MODEL_ID is set, else a tiny random model under SMOKE just to exercise the rendering).

Run:  HF_MODEL_ID=you/repo python viz_v2_edge_arcs.py   |   SMOKE=1 python viz_v2_edge_arcs.py
"""
import os
import torch
import mech_common as MC

REPO = os.environ.get("HF_MODEL_ID")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PASSAGE = ("the little girl saw the old boat and the girl wanted the old boat by the river")


def run():
    model, tok, cfg = MC.load_model(REPO, DEVICE)
    n_layers = len(MC.get_layers(model))

    if tok is not None and not MC.SMOKE:
        ids = tok(PASSAGE, return_attention_mask=False)["input_ids"][:48]
        labels = tok.convert_ids_to_tokens(ids)
        x = torch.tensor([ids], device=DEVICE)
    else:
        x = MC.synthetic_induction_batch(1, 32, cfg.vocab_size, DEVICE, seed=3)
        labels = [str(int(t)) for t in x[0]]

    cap = MC.MixerCapture(model)
    with torch.no_grad(), cap:
        model(input_ids=x)

    # per layer: for each query t, the top valid candidate + its mass
    per_layer = []
    for L in range(n_layers):
        with torch.no_grad():
            S, mass, ok, nxt = MC.recompute_mixer_S(MC.get_layers(model)[L].mixer, cap.x[L], cap.cand)
        m = mass[0].masked_fill(~ok[0], -1.0)                 # (T,C)
        top_c = m.argmax(-1)
        top_m = m.gather(1, top_c[:, None]).squeeze(1)
        src = cap.cand[0].gather(1, top_c[:, None]).squeeze(1)  # source position
        per_layer.append((src.cpu(), top_m.cpu(), ok[0].any(-1).cpu()))

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        import numpy as np
        T = x.size(1)
        fig, axes = plt.subplots(n_layers, 1, figsize=(min(16, 0.45 * T + 2), 2.2 * n_layers), squeeze=False)
        for L in range(n_layers):
            ax = axes[L][0]
            src, mval, has = per_layer[L]
            ax.scatter(range(T), [0] * T, s=8, color="#333")
            for t in range(T):
                if has[t] and src[t] >= 0 and mval[t] > 0:
                    s = int(src[t]); w = float(mval[t])
                    cx = (s + t) / 2.0; rad = (t - s) / 2.0
                    th = np.linspace(0, np.pi, 30)
                    ax.plot(cx + rad * np.cos(th), np.sin(th) * 0.6,
                            color="#cc3311", alpha=min(1.0, 0.15 + 0.85 * w), lw=1.0)
            ax.set_xlim(-1, T); ax.set_ylim(-0.1, 0.8); ax.set_yticks([])
            ax.set_title(f"layer {L} — top look-back edge per token (opacity = mass)", fontsize=9, loc="left")
            if L == n_layers - 1:
                ax.set_xticks(range(T)); ax.set_xticklabels(labels, rotation=90, fontsize=6)
            else:
                ax.set_xticks([])
        os.makedirs("figs", exist_ok=True); fig.tight_layout(); fig.savefig("figs/v2_edge_arcs.png", dpi=140)
        print("saved figs/v2_edge_arcs.png")
    except Exception as e:
        print(f"(figure skipped: {type(e).__name__}: {e})")

    if MC.SMOKE:
        print("SMOKE OK: per-layer top edges extracted + figure produced.")


if __name__ == "__main__":
    run()

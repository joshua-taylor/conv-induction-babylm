"""
exp_m1_edge_patching.py — does information flow ALONG the look-back edges?  (H2, part 1)
=======================================================================================

For a query t whose strongest long-range edge points to source s (continuation s+1, copied
token y* = x[s+1]), corrupt the mixer's input residual at s+1 (mean-ablation) and measure how
much the query's prediction of y* drops, vs corrupting a distance-matched control position
that is NOT one of t's sources. The edge is long-range (>= MIN_DIST, beyond the conv's reach),
so any effect on t is edge-mediated.

  PREDICTION : Delta logp(y*) under edge-patch >> under control-patch (flow along the edge);
               and (from the layer sweep) the effect concentrates in the upper layers, where
               the copy is assembled.
  FALSIFIER  : edge effect ~ control effect ~ 0.

Sweeps every layer in one run and writes figs/m1_edge_patching.png.

Run:  HF_MODEL_ID=you/repo python exp_m1_edge_patching.py   |   SMOKE=1 python exp_m1_edge_patching.py
"""
import os
import torch
import torch.nn.functional as F
import mech_common as MC

REPO = os.environ.get("HF_MODEL_ID")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIN_DIST = int(os.environ.get("MIN_DIST", "6" if MC.SMOKE else "32"))
T = 32 if MC.SMOKE else 256
B = 8 if MC.SMOKE else 16
N_BATCHES = 2 if MC.SMOKE else 80
K_PER_SEQ = int(os.environ.get("K_PER_SEQ", "2" if MC.SMOKE else "8"))


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def logits_at(model, x, capture=None, patch=None):
    cc = capture if capture is not None else _null()
    cp = patch if patch is not None else _null()
    with torch.no_grad(), cc, cp:
        return model(input_ids=x).logits


def select_multi(mass, cand, ok, min_dist, t_min, k):
    """Up to k strongest qualifying long-range edges per sequence (distinct query positions)."""
    Bn, Tn, C = mass.shape; dev = mass.device
    ti = torch.arange(Tn, device=dev)[None, :, None]
    good = ok & ((ti - (cand + 1)) >= min_dist) & (ti >= t_min)
    m = mass.masked_fill(~good, -1.0)
    best_c = m.argmax(-1)                                    # (B,T)
    best_m = m.gather(2, best_c[..., None]).squeeze(-1)      # (B,T)
    bs, ts, cts = [], [], []
    for b in range(Bn):
        vt = (best_m[b] > 0).nonzero(as_tuple=True)[0]
        if vt.numel() == 0:
            continue
        order = best_m[b, vt].argsort(descending=True)[:k]
        for t in vt[order].tolist():
            s = int(cand[b, t, int(best_c[b, t])])
            bs.append(b); ts.append(t); cts.append(s + 1)
    if not bs:
        z = torch.zeros(0, dtype=torch.long, device=dev)
        return z, z, z
    return (torch.tensor(bs, device=dev), torch.tensor(ts, device=dev),
            torch.tensor(cts, device=dev))


def measure_layer(model, data, layer, T):
    rows = []
    for bi in range(0, data.size(0), B):
        x = data[bi:bi + B]
        if x.size(0) < 2:
            continue
        cap = MC.MixerCapture(model)
        clean = logits_at(model, x, capture=cap)
        xin, cand = cap.x[layer], cap.cand
        mixer = MC.get_layers(model)[layer].mixer
        with torch.no_grad():
            _, mass, ok, _ = MC.recompute_mixer_S(mixer, xin, cand)
        sel_b, sel_t, sel_cont = select_multi(mass, cand, ok, MIN_DIST, t_min=8, k=K_PER_SEQ)
        if sel_b.numel() == 0:
            continue
        b, t, cont = sel_b, sel_t, sel_cont.clamp(0, T - 1)
        y = x[b, cont]
        ctrl = MC.distance_matched_control(x, b, t, cont, cand)
        repl = xin.reshape(-1, xin.size(-1)).mean(0)
        with MC.PatchMixerInput(model, layer, b, cont, repl):
            edge = logits_at(model, x)
        with MC.PatchMixerInput(model, layer, b, ctrl, repl):
            ctl = logits_at(model, x)
        lc = F.log_softmax(clean[b, t], -1).gather(1, y[:, None]).squeeze(1)
        le = F.log_softmax(edge[b, t], -1).gather(1, y[:, None]).squeeze(1)
        lk = F.log_softmax(ctl[b, t], -1).gather(1, y[:, None]).squeeze(1)
        l2e = (edge[b, t] - clean[b, t]).norm(dim=-1)
        l2k = (ctl[b, t] - clean[b, t]).norm(dim=-1)
        for i in range(b.numel()):
            rows.append((float(lc[i] - le[i]), float(lc[i] - lk[i]), float(l2e[i]), float(l2k[i])))
    return torch.tensor(rows) if rows else torch.empty(0, 4)


def run():
    model, tok, cfg = MC.load_model(REPO, DEVICE)
    n_layers = len(MC.get_layers(model))
    print(f"model: {n_layers} layers d={cfg.d_model} vocab={cfg.vocab_size} | MIN_DIST={MIN_DIST} | {DEVICE}")

    if REPO and tok is not None and not MC.SMOKE:
        from datasets import load_dataset
        ds = load_dataset("BabyLM-community/BabyLM-2026-Strict-Small", split="train", streaming=True)
        texts = [r.get("text") or r.get("content") or "" for r, _ in zip(ds, range(3000))]
        data = MC.tokenize_corpus(tok, texts, T, max_seqs=B * N_BATCHES, device=DEVICE)
    else:
        data = torch.cat([MC.synthetic_induction_batch(B, T, cfg.vocab_size, DEVICE, seed=i)
                          for i in range(N_BATCHES)], 0)

    print(f"\n{'layer':>5} {'n':>5} {'Δlogp edge':>12} {'Δlogp ctrl':>12} {'edge-ctrl':>10} {'‖Δ‖edge':>9} {'‖Δ‖ctrl':>9}")
    print("-" * 72)
    res = []
    for L in range(n_layers):
        r = measure_layer(model, data, L, T)
        if r.numel() == 0:
            print(f"{L:>5}     0   (no qualifying edges)"); res.append((L, 0, 0.0, 0.0, 0.0, 0.0, 0.0)); continue
        de, dc = float(r[:, 0].mean()), float(r[:, 1].mean())
        le, lk = float(r[:, 2].mean()), float(r[:, 3].mean())
        res.append((L, r.size(0), de, dc, float(r[:, 0].std()), le, lk))
        note = "" if lk > 1e-3 else "   (ctrl=0: no path downstream of this layer)"
        print(f"{L:>5} {r.size(0):>5} {de:>12.4f} {dc:>12.4f} {de-dc:>10.4f} {le:>9.2f} {lk:>9.2f}{note}")

    try:
        import paper_style as ps; ps.apply()
        import matplotlib.pyplot as plt
        import numpy as np
        L = [r[0] for r in res]; x = np.arange(len(L)); w = 0.38
        edge = [r[2] for r in res]; ctrl = [r[3] for r in res]; err = [r[4] for r in res]
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        ax.bar(x - w / 2, edge, w, yerr=err, capsize=3, label="patch edge source",
               color=ps.PRIMARY, edgecolor="white", lw=0.5)
        ax.bar(x + w / 2, ctrl, w, label="patch control", color=ps.SECOND, edgecolor="white", lw=0.5)
        ax.axhline(0, color=ps.MUTED, lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels([f"layer {l}" for l in L])
        ax.set_ylabel("drop in logp(copied token)"); ax.set_title("information flow along the edge, by layer")
        ax.legend()
        ps.save(fig, "figs/m1_edge_patching.png")
    except Exception as e:
        print(f"(figure skipped: {type(e).__name__}: {e})")

    if MC.SMOKE:
        nonzero = [r for r in res if r[1] > 0]
        assert nonzero, "no edges found in smoke"
        assert all(r[5] >= r[6] - 1e-6 for r in nonzero), "edge should shift the query at least as much as control"
        print("\nSMOKE OK: per-layer edge-patching swept; edge shift >= control at every layer.")


if __name__ == "__main__":
    run()

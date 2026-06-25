"""
exp_m2_edge_severing.py — are the look-back edges NECESSARY?  (H2, necessity)
=============================================================================

Complement to M1 (which showed information flows along edges). Here we SEVER the strongest
long-range edge for a query (set that candidate to invalid in `cand`, applied at every layer)
and measure the drop in the query's log-prob of the copied token. Control: sever the
LOWEST-mass valid candidate for the same query instead.

  PREDICTION : severing the top edge drops logp(y*) far more than severing a low-mass edge.
  FALSIFIER  : top and control drops are comparable -> the specific edge isn't load-bearing.

Run:  python exp_m2_edge_severing.py   |   SMOKE=1 python exp_m2_edge_severing.py
"""
import os, math
import torch
import torch.nn.functional as F
import mech_common as MC

REPO = os.environ.get("HF_MODEL_ID")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER = int(os.environ.get("LAYER", "-1"))
MIN_DIST = int(os.environ.get("MIN_DIST", "6" if MC.SMOKE else "48"))
T = 32 if MC.SMOKE else 256
B = 8 if MC.SMOKE else 16
N_BATCHES = 2 if MC.SMOKE else 40


class SeverEdges:
    """Set cand[b,t,c] = -1 (invalid) at every layer's mixer -> the edge is globally removed."""
    def __init__(self, model, b, t, c):
        self.layers = MC.get_layers(model); self.b, self.t, self.c = b, t, c; self._h = []

    def __enter__(self):
        def pre(mod, args):
            x, cand = args
            cand = cand.clone()
            cand[self.b, self.t, self.c] = -1
            return (x, cand)
        for l in self.layers:
            self._h.append(l.mixer.register_forward_pre_hook(pre))
        return self

    def __exit__(self, *a):
        for h in self._h:
            h.remove()


def run():
    model, tok, cfg = MC.load_model(REPO, DEVICE)
    n_layers = len(MC.get_layers(model))
    layer = LAYER if LAYER >= 0 else n_layers - 1
    print(f"model: {n_layers} layers d={cfg.d_model} | sever@all-layers, rank@layer {layer} | MIN_DIST={MIN_DIST}")

    if REPO and tok is not None and not MC.SMOKE:
        from datasets import load_dataset
        ds = load_dataset("BabyLM-community/BabyLM-2026-Strict-Small", split="train", streaming=True)
        texts = [r.get("text") or r.get("content") or "" for r, _ in zip(ds, range(2000))]
        data = MC.tokenize_corpus(tok, texts, T, max_seqs=B * N_BATCHES, device=DEVICE)
    else:
        data = torch.cat([MC.synthetic_induction_batch(B, T, cfg.vocab_size, DEVICE, seed=i)
                          for i in range(N_BATCHES)], 0)

    rows = []
    for bi in range(0, data.size(0), B):
        x = data[bi:bi + B]
        if x.size(0) < 2:
            continue
        cap = MC.MixerCapture(model)
        with torch.no_grad(), cap:
            clean = model(input_ids=x).logits
        mixer = MC.get_layers(model)[layer].mixer
        with torch.no_grad():
            _, mass, ok, _ = MC.recompute_mixer_S(mixer, cap.x[layer], cap.cand)
        cand = cap.cand
        Bn, Tn, C = mass.shape
        ti = torch.arange(Tn, device=DEVICE)[None, :, None]
        dist = ti - (cand + 1)
        good = ok & (dist >= MIN_DIST) & (ti >= 8)
        m = mass.masked_fill(~good, -1.0)
        flat = m.view(Bn, -1); best = flat.argmax(-1); bestval = flat.gather(1, best[:, None]).squeeze(1)
        keep = bestval > 0
        if keep.sum() == 0:
            continue
        bidx = torch.arange(Bn, device=DEVICE)[keep]
        t_sel = (best[keep] // C); top_c = (best[keep] % C)
        # control = lowest-mass *valid* candidate for the same (b,t), excluding top
        mt = mass[bidx, t_sel].clone()                       # (n,C)
        mt = mt.masked_fill(~ok[bidx, t_sel], float("inf"))
        mt[torch.arange(bidx.numel()), top_c] = float("inf")
        ctrl_c = mt.argmin(-1)                                # low-mass valid edge, or a no-op slot
        cont = (cand[bidx, t_sel, top_c] + 1).clamp(0, Tn - 1)
        y_star = x[bidx, cont]

        with torch.no_grad():
            with SeverEdges(model, bidx, t_sel, top_c):
                sev = model(input_ids=x).logits
            with SeverEdges(model, bidx, t_sel, ctrl_c):
                ctl = model(input_ids=x).logits
        lp_clean = F.log_softmax(clean[bidx, t_sel], -1).gather(1, y_star[:, None]).squeeze(1)
        lp_sev = F.log_softmax(sev[bidx, t_sel], -1).gather(1, y_star[:, None]).squeeze(1)
        lp_ctl = F.log_softmax(ctl[bidx, t_sel], -1).gather(1, y_star[:, None]).squeeze(1)
        l2_s = (sev[bidx, t_sel] - clean[bidx, t_sel]).norm(dim=-1)
        l2_c = (ctl[bidx, t_sel] - clean[bidx, t_sel]).norm(dim=-1)
        for i in range(bidx.numel()):
            rows.append((float(lp_clean[i] - lp_sev[i]), float(lp_clean[i] - lp_ctl[i]),
                         float(l2_s[i]), float(l2_c[i])))

    if not rows:
        print("No qualifying edges (lower MIN_DIST)."); return
    r = torch.tensor(rows)
    print(f"\nselected edges: {len(rows)}")
    print(f"  Delta logp(y*)  sever-TOP  = {r[:,0].mean():+.4f} +/- {r[:,0].std():.4f}")
    print(f"  Delta logp(y*)  sever-ctrl = {r[:,1].mean():+.4f} +/- {r[:,1].std():.4f}")
    print(f"  top - ctrl                 = {(r[:,0]-r[:,1]).mean():+.4f}   (>0 => edge is load-bearing)")
    print(f"  ||Δlogits@t||  top={r[:,2].mean():.4f}  ctrl={r[:,3].mean():.4f}")
    if MC.SMOKE:
        assert torch.isfinite(r).all()
        assert r[:, 2].mean() >= r[:, 3].mean() - 1e-6, "top-edge severing should shift query >= control"
        print("\nSMOKE OK: severing the top edge shifts the query at least as much as a low-mass edge.")


if __name__ == "__main__":
    run()

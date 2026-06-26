"""
exp_m2_edge_severing.py — are the look-back edges NECESSARY?  (H2, necessity)
=============================================================================

Complement to M1 (which showed information flows along edges). Here we SEVER the strongest
long-range edge for a query (set that candidate to invalid in `cand`, applied at every layer)
and measure the drop in the query's log-prob of the copied token (NECESSITY). For SPECIFICITY
we sever, on the same query, the lowest-mass valid candidate whose continuation differs from y*.
On repetitive spans no such contrastive candidate exists; there the control is a true no-op and
the query is excluded from the specificity comparison (we report how many queries had a control).

  PREDICTION : severing the top edge drops logp(y*) sharply; where a contrastive control exists,
               severing it drops logp(y*) far less.
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

    top_d, top_l2 = [], []                         # necessity: clean -> sever-top (all queries)
    ctrl_d, top_d_sub = [], []                     # specificity: paired, on queries with a real control
    n_kept = 0; n_ctrl = 0
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
        good = ok & ((ti - (cand + 1)) >= MIN_DIST) & (ti >= 8)
        m = mass.masked_fill(~good, -1.0)
        flat = m.view(Bn, -1); best = flat.argmax(-1)
        keep = flat.gather(1, best[:, None]).squeeze(1) > 0
        if keep.sum() == 0:
            continue
        bidx = torch.arange(Bn, device=DEVICE)[keep]
        t_sel = (best[keep] // C); top_c = (best[keep] % C)
        cont = (cand[bidx, t_sel, top_c] + 1).clamp(0, Tn - 1)
        y_star = x[bidx, cont]
        n_kept += bidx.numel()

        # --- necessity: sever the top edge (every kept query) ---
        with torch.no_grad(), SeverEdges(model, bidx, t_sel, top_c):
            sev = model(input_ids=x).logits
        lp_clean = F.log_softmax(clean[bidx, t_sel], -1).gather(1, y_star[:, None]).squeeze(1)
        lp_sev = F.log_softmax(sev[bidx, t_sel], -1).gather(1, y_star[:, None]).squeeze(1)
        drop_top = lp_clean - lp_sev
        top_d += drop_top.tolist()
        top_l2 += (sev[bidx, t_sel] - clean[bidx, t_sel]).norm(dim=-1).tolist()

        # --- specificity control: lowest-mass valid candidate whose continuation != y* ---
        # repetitive spans have no such candidate -> control is a no-op there (query skipped).
        cont_tok = torch.gather(x[bidx], 1, (cand[bidx, t_sel] + 1).clamp(0, Tn - 1))
        mt = mass[bidx, t_sel].clone()
        mt = mt.masked_fill(~ok[bidx, t_sel], float("inf"))
        mt = mt.masked_fill(cont_tok == y_star[:, None], float("inf"))
        mt[torch.arange(bidx.numel()), top_c] = float("inf")
        has_ctrl = torch.isfinite(mt).any(-1)
        n_ctrl += int(has_ctrl.sum())
        if has_ctrl.any():
            sub = has_ctrl.nonzero(as_tuple=True)[0]
            cc = mt[sub].argmin(-1)
            bc, tc, yc = bidx[sub], t_sel[sub], y_star[sub]
            with torch.no_grad(), SeverEdges(model, bc, tc, cc):
                ctl = model(input_ids=x).logits
            lp_cln = F.log_softmax(clean[bc, tc], -1).gather(1, yc[:, None]).squeeze(1)
            lp_ctl = F.log_softmax(ctl[bc, tc], -1).gather(1, yc[:, None]).squeeze(1)
            ctrl_d += (lp_cln - lp_ctl).tolist()
            top_d_sub += drop_top[sub].tolist()

    if n_kept == 0:
        print("No qualifying edges (lower MIN_DIST)."); return
    td = torch.tensor(top_d)
    print(f"\nqueries with a long-range top edge: {n_kept}")
    print(f"  NECESSITY   clean -> sever-TOP   Δlogp(y*) = {td.mean():+.4f} +/- {td.std():.4f}")
    print(f"              ||Δlogits@t|| (top)            = {torch.tensor(top_l2).mean():.4f}")
    print(f"\n  contrastive control available for {n_ctrl}/{n_kept} queries "
          f"({100 * n_ctrl / max(n_kept, 1):.0f}%); the rest sit on repetitive spans where every")
    print(f"  alternative shares the same continuation -> the control is a no-op there.")
    if ctrl_d:
        cd = torch.tensor(ctrl_d); ts = torch.tensor(top_d_sub)
        print(f"\n  SPECIFICITY (paired, on those {len(ctrl_d)} queries):")
        print(f"     sever-TOP  Δlogp(y*) = {ts.mean():+.4f} +/- {ts.std():.4f}")
        print(f"     sever-CTRL Δlogp(y*) = {cd.mean():+.4f} +/- {cd.std():.4f}")
        print(f"     top - ctrl           = {(ts - cd).mean():+.4f}   (>0 => the specific edge is load-bearing)")
    else:
        print(f"\n  (no contrastive controls at MIN_DIST={MIN_DIST}; necessity stands on the sever-TOP drop.)")
    if MC.SMOKE:
        assert torch.isfinite(td).all()
        print("\nSMOKE OK: sever-TOP gives a finite necessity drop; control reported separately.")


if __name__ == "__main__":
    run()

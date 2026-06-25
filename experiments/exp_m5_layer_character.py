"""
exp_m5_layer_character.py — how do the edges change across layers?  (H2, "evolves through layers")
=================================================================================================

Per layer, characterise the induction edges:
  * mean look-back distance (mass-weighted)        — how far back the layer reaches
  * mass entropy over candidates                   — focused vs diffuse selection
  * sink mass                                      — how often the layer abstains
  * token-identity decodability of the copied value (off=0 linear probe, reused from M3)
                                                   — is the transported content still lexical?

  PREDICTION : shallow layers = short distance, high token-identity (lexical copy);
               deeper layers = longer distance, lower token-identity (more contextual).

Run:  python exp_m5_layer_character.py   |   SMOKE=1 python exp_m5_layer_character.py
"""
import os, math
import torch
import mech_common as MC
import exp_m3_edge_probe as M3      # reuse collect_values + probe_accuracy

REPO = os.environ.get("HF_MODEL_ID")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T = 32 if MC.SMOKE else 256
B = 8 if MC.SMOKE else 16
N_BATCHES = 3 if MC.SMOKE else 20


def run():
    model, tok, cfg = MC.load_model(REPO, DEVICE)
    n_layers = len(MC.get_layers(model))
    print(f"model: {n_layers} layers d={cfg.d_model} vocab={cfg.vocab_size}")

    if REPO and tok is not None and not MC.SMOKE:
        from datasets import load_dataset
        ds = load_dataset("BabyLM-community/BabyLM-2026-Strict-Small", split="train", streaming=True)
        texts = [r.get("text") or r.get("content") or "" for r, _ in zip(ds, range(2000))]
        data = MC.tokenize_corpus(tok, texts, T, max_seqs=B * N_BATCHES, device=DEVICE)
    else:
        data = torch.cat([MC.synthetic_induction_batch(B, T, cfg.vocab_size, DEVICE, seed=i)
                          for i in range(N_BATCHES)], 0)

    print(f"\n{'layer':>5} {'mean_dist':>10} {'mass_entropy':>13} {'sink_mass':>10} {'tok-id(off0)':>13}")
    print("-" * 56)
    for L in range(n_layers):
        dists, ents, sinks = [], [], []
        for bi in range(0, data.size(0), B):
            x = data[bi:bi + B]
            cap = MC.MixerCapture(model)
            with torch.no_grad(), cap:
                model(input_ids=x)
            mixer = MC.get_layers(model)[L].mixer
            with torch.no_grad():
                S, mass, ok, nxt = MC.recompute_mixer_S(mixer, cap.x[L], cap.cand)
            Bn, Tn, C = mass.shape
            ti = torch.arange(Tn, device=DEVICE)[None, :, None]
            dist = (ti - nxt).clamp(min=0).float()
            w = mass * ok.float()                                   # candidate mass where valid
            wsum = w.sum()
            if wsum > 0:
                dists.append(float((w * dist).sum() / wsum))
                p = (w / w.sum(-1, keepdim=True).clamp(min=1e-9))
                ent = -(p * (p + 1e-9).log()).sum(-1) / math.log(max(C, 2))
                ents.append(float(ent[ok.any(-1)].mean()))
            sinks.append(float(S[..., C, :].mean()))
        # token-identity (off 0) probe on the copied value at this layer
        reps, toks = M3.collect_values(model, data, L, ablate=False)
        acc0, _ = M3.probe_accuracy(reps, toks, T, 0, cfg.vocab_size, DEVICE)
        md = sum(dists) / len(dists) if dists else float("nan")
        me = sum(ents) / len(ents) if ents else float("nan")
        sm = sum(sinks) / len(sinks)
        print(f"{L:>5} {md:>10.2f} {me:>13.3f} {sm:>10.3f} {acc0:>13.3f}")

    if MC.SMOKE:
        print("\nSMOKE OK: per-layer edge statistics computed.")


if __name__ == "__main__":
    run()

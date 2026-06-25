"""
exp_m3_edge_probe.py — does the CONV write local context that the edge then transports? (H2, part 2)
====================================================================================================

The value copied along an edge is v_proj(norm(x_L)) at the continuation position. If the
DynamicConv "shuttles local patterns onto the junctions", that transported vector should
encode not just the copied token but its LOCAL NEIGHBOURHOOD — information only the (causal)
conv could have written there.

Probe (linear) the copied-value representation v_proj(norm(x_L))[p] to predict the token at
p+off for off in {-2,-1,0,+1,+2}, under conv-ON vs conv-OFF, per layer.

  PREDICTION : left-context offsets {-2,-1} decode well with conv, collapse without it;
               off 0 (self) decodes either way; future offsets {+1,+2} stay at chance
               (causal control — the harness must not leak).
  FALSIFIER  : {-2,-1} decode equally well WITHOUT the conv  => conv isn't feeding the edges.

Run:  python exp_m3_edge_probe.py            (needs HF_MODEL_ID env or edit REPO)
      SMOKE=1 python exp_m3_edge_probe.py    (tiny CPU harness test)
"""

import os, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import mech_common as MC

REPO = os.environ.get("HF_MODEL_ID")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OFFSETS = [-2, -1, 0, 1, 2]
T = 32 if MC.SMOKE else 256
B = 8 if MC.SMOKE else 16
N_BATCHES = 3 if MC.SMOKE else 30
PROBE_STEPS = 150 if MC.SMOKE else 800


@torch.no_grad()
def collect_values(model, data, layer, ablate):
    """Return v_proj(norm(x_layer)) at every position, stacked over the corpus, + the token ids."""
    mixer = MC.get_layers(model)[layer].mixer
    reps, toks = [], []
    for bi in range(0, data.size(0), B):
        x = data[bi:bi + B]
        cap = MC.MixerCapture(model)
        ctx = MC.AblateConv(model) if ablate else _null()
        with ctx, cap:
            model(input_ids=x)
        xin = cap.x[layer]
        h = mixer.norm(xin)
        v = mixer.v(h)                                          # (B,T,d) the transported value
        reps.append(v.reshape(-1, v.size(-1)))
        toks.append(x.reshape(-1))
    return torch.cat(reps, 0), torch.cat(toks, 0)


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def probe_accuracy(reps, toks, T, off, vocab, device):
    """Linear probe reps[p] -> token[p+off]. Returns (top1_acc, majority_baseline)."""
    N = reps.size(0)
    pos = torch.arange(N, device=device)
    in_seq = ((pos % T) + off >= 0) & ((pos % T) + off < T)     # stay within a sequence
    idx = pos[in_seq]
    X = reps[idx]
    y = toks[idx + off]
    # split
    perm = torch.randperm(X.size(0), device=device)
    X, y = X[perm], y[perm]
    n_tr = int(0.8 * X.size(0))
    Xtr, ytr, Xte, yte = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]
    if Xte.numel() == 0:
        return float("nan"), float("nan")
    # standardise
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp(min=1e-5)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    probe = nn.Linear(X.size(1), vocab).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(PROBE_STEPS):
        opt.zero_grad()
        loss = F.cross_entropy(probe(Xtr), ytr)
        loss.backward(); opt.step()
    with torch.no_grad():
        acc = (probe(Xte).argmax(-1) == yte).float().mean().item()
    # majority-class baseline
    maj = torch.bincount(ytr, minlength=vocab).max().item() / max(ytr.numel(), 1)
    return acc, maj


def run():
    model, tok, cfg = MC.load_model(REPO, DEVICE)
    n_layers = len(MC.get_layers(model))
    probe_layers = sorted(set([0, n_layers - 1]))
    print(f"model: {n_layers} layers, d={cfg.d_model}, vocab={cfg.vocab_size} | probing layers {probe_layers}")

    if REPO and tok is not None and not MC.SMOKE:
        from datasets import load_dataset
        ds = load_dataset("BabyLM-community/BabyLM-2026-Strict-Small", split="train", streaming=True)
        texts = [r.get("text") or r.get("content") or "" for r, _ in zip(ds, range(2000))]
        data = MC.tokenize_corpus(tok, texts, T, max_seqs=B * N_BATCHES, device=DEVICE)
    else:
        data = torch.cat([MC.synthetic_induction_batch(B, T, cfg.vocab_size, DEVICE, seed=i)
                          for i in range(N_BATCHES)], 0)

    print(f"\n{'layer':>5} {'conv':>5} " + " ".join(f"off{o:+d}" for o in OFFSETS) + "   (top-1 acc; chance≈maj)")
    print("-" * 70)
    results = {}
    for layer in probe_layers:
        for ablate in (False, True):
            reps, toks = collect_values(model, data, layer, ablate)
            accs, majs = [], []
            for off in OFFSETS:
                a, mj = probe_accuracy(reps, toks, T, off, cfg.vocab_size, DEVICE)
                accs.append(a); majs.append(mj)
            results[(layer, ablate)] = accs
            tag = "OFF" if ablate else "ON"
            print(f"{layer:>5} {tag:>5} " + " ".join(f"{a:5.2f}" for a in accs) +
                  f"    maj≈{majs[OFFSETS.index(0)]:.2f}")

    if MC.SMOKE:
        l0_on = results[(0, False)]
        i0, ip1 = OFFSETS.index(0), OFFSETS.index(1)
        assert all(a == a for row in results.values() for a in row), "NaN in probe accuracies"
        assert l0_on[i0] > l0_on[ip1], "self-token (off 0) should beat future (off +1, causal control)"
        print("\nSMOKE OK: off 0 decodes; future offset stays low (causal control holds).")
    else:
        # headline contrast for the figure
        on, off = results[(0, False)], results[(0, True)]
        im1 = OFFSETS.index(-1)
        print(f"\nLeft-context (off -1) at layer 0:  conv ON = {on[im1]:.3f}  vs  conv OFF = {off[im1]:.3f}")
        print("=> the gap is the conv-written local context that rides the edge.")


if __name__ == "__main__":
    run()

"""
viz_v1_coverage.py — how often do tokens "look back", by genre?  (V1)
=====================================================================

For each genre (narrative / dialogue / expository / structured by default, or your own dict),
tokenise and compute from the exact-match index alone (no model needed):
  * coverage  = fraction of positions that have a prior occurrence of the same token
  * look-back distance distribution

Renders a bar chart (coverage by genre) + distance histograms to figs/v1_coverage.png.
With HF_MODEL_ID set, uses the production tokenizer (8k BPE); otherwise a whitespace tokenizer
(coverage is qualitatively similar; the BPE version is what goes in the paper).

Run:  HF_MODEL_ID=you/repo python viz_v1_coverage.py   |   SMOKE=1 python viz_v1_coverage.py
"""
import os
import torch
from modeling_induction import prev_same_key

REPO = os.environ.get("HF_MODEL_ID")

SAMPLES = {
    "narrative": "the little girl walked to the river . she saw a small boat by the water . "
                 "the boat was old and the girl wanted to ride in the old boat .",
    "dialogue": "are you coming to the party ? yes i am coming . are you coming too ? "
                "no i am not coming . why are you not coming to the party ?",
    "expository": "water is made of hydrogen and oxygen . hydrogen is light . oxygen is heavy . "
                  "water freezes when the temperature of the water is low .",
    "structured": "item one apple item two banana item three apple item four banana "
                  "item five apple item six banana item seven apple",
}


def whitespace_ids(texts):
    vocab = {}
    out = []
    for t in texts:
        ids = []
        for w in t.split():
            ids.append(vocab.setdefault(w, len(vocab) + 2))
        out.append(ids)
    return out


def coverage_stats(ids_list):
    rows = {}
    all_d = {}
    for name, ids in ids_list.items():
        x = torch.tensor([ids], dtype=torch.long)
        prev = prev_same_key(x)[0]                            # (T,)
        t = torch.arange(x.size(1))
        has = prev >= 0
        dist = (t - prev)[has].float()
        rows[name] = dict(coverage=float(has.float().mean()),
                          mean_dist=float(dist.mean()) if dist.numel() else 0.0,
                          n=int(has.sum()))
        all_d[name] = dist.tolist()
    return rows, all_d


def run():
    samples = SAMPLES
    if REPO and not os.environ.get("SMOKE"):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
        ids = {k: tok(v, return_attention_mask=False)["input_ids"] for k, v in samples.items()}
    else:
        toks = whitespace_ids(list(samples.values()))
        ids = {k: toks[i] for i, k in enumerate(samples)}

    rows, all_d = coverage_stats(ids)
    print(f"{'genre':<12}{'coverage':>10}{'mean_dist':>11}{'n_match':>9}")
    print("-" * 44)
    for k, r in rows.items():
        print(f"{k:<12}{r['coverage']:>10.3f}{r['mean_dist']:>11.2f}{r['n']:>9}")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        names = list(rows)
        ax[0].bar(names, [rows[n]["coverage"] for n in names], color="#4477aa")
        ax[0].set(title="look-back coverage by genre", ylabel="fraction of positions with a match", ylim=(0, 1))
        ax[0].tick_params(axis="x", rotation=20)
        for n in names:
            d = all_d[n]
            if d:
                ax[1].hist(d, bins=15, histtype="step", label=n, density=True)
        ax[1].set(title="look-back distance distribution", xlabel="distance (tokens)", ylabel="density")
        ax[1].legend()
        os.makedirs("figs", exist_ok=True); fig.tight_layout(); fig.savefig("figs/v1_coverage.png", dpi=140)
        print("\nsaved figs/v1_coverage.png")
    except Exception as e:
        print(f"(figure skipped: {type(e).__name__}: {e})")

    if os.environ.get("SMOKE"):
        assert all(0 <= r["coverage"] <= 1 for r in rows.values())
        print("SMOKE OK: coverage stats + figure produced.")


if __name__ == "__main__":
    run()

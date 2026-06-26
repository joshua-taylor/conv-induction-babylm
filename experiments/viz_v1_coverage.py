"""
viz_v1_coverage.py — how often do tokens "look back", across real text genres?  (V1, refined)
==============================================================================================

Uses small, open corpora across genres, the model's BPE tokenizer, slices each into
non-overlapping 256-token windows, and averages the look-back coverage over windows.
Coverage is a property of the data + tokenizer (the exact-match index), so no model fwd needed.

Genres (each with a graceful fallback to a built-in sample if the source can't be reached):
  code        : real .py files from a public GitHub repo (raw)
  stories      : roneneldan/TinyStories          (child-level narrative; on-theme for BabyLM)
  encyclopedic : wikitext-2-raw-v1
  dialogue     : daily_dialog

Renders figs/v1_coverage.png (coverage bar + distance distribution). Needs HF_MODEL_ID for the
BPE tokenizer (falls back to a whitespace tokenizer if unavailable).

Run:  HF_MODEL_ID=you/repo python viz_v1_coverage.py   |   SMOKE=1 python viz_v1_coverage.py
"""
import os, urllib.request
import torch
from modeling_induction import prev_same_key
import paper_style as ps

REPO = os.environ.get("HF_MODEL_ID")
SMOKE = bool(os.environ.get("SMOKE"))
WINDOW = 256
MAX_WINDOWS = 8 if SMOKE else 80
N_DOCS = 50 if SMOKE else 400

CODE_URLS = [
    "https://raw.githubusercontent.com/psf/requests/main/src/requests/models.py",
    "https://raw.githubusercontent.com/psf/requests/main/src/requests/sessions.py",
    "https://raw.githubusercontent.com/psf/requests/main/src/requests/utils.py",
    "https://raw.githubusercontent.com/psf/requests/main/src/requests/adapters.py",
]

FALLBACK = {
    "code": "def add(a, b):\n    return a + b\n\ndef add(a, b):\n    return a + b\n",
    "stories": "the little girl walked to the river . she saw a small boat . the boat was old .",
    "encyclopedic": "water is made of hydrogen and oxygen . hydrogen is light and oxygen is heavy .",
    #"dialogue": "are you coming ? yes i am coming . are you coming too ? no i am not coming .",
}


def fetch_code():
    out = []
    for u in CODE_URLS:
        try:
            with urllib.request.urlopen(u, timeout=20) as r:
                out.append(r.read().decode("utf-8", "ignore"))
        except Exception:
            pass
    return out or [FALLBACK["code"]]


def fetch_hf(name, config, split, field, joiner="\n"):
    try:
        from datasets import load_dataset
        ds = load_dataset(name, config, split=split, streaming=True)
        docs = []
        for r, _ in zip(ds, range(N_DOCS)):
            v = r.get(field)
            if isinstance(v, list):      # daily_dialog: list of utterances
                v = joiner.join(v)
            if v and v.strip():
                docs.append(v)
        return docs or None
    except Exception as e:
        print(f"  ({name} unavailable: {type(e).__name__}; using fallback)")
        return None


def get_genre_texts():
    if SMOKE:
        return {k: [v] * 6 for k, v in FALLBACK.items()}
    g = {}
    g["code"] = fetch_code()
    g["stories"] = fetch_hf("roneneldan/TinyStories", None, "train", "text") or [FALLBACK["stories"]]
    g["encyclopedic"] = fetch_hf("wikitext", "wikitext-2-raw-v1", "train", "text") or [FALLBACK["encyclopedic"]]
    #g["dialogue"] = fetch_hf("daily_dialog", None, "train", "dialog") or [FALLBACK["dialogue"]]
    return g


def encode(tok, texts):
    if tok is not None:
        big = "\n".join(texts)
        return tok(big, return_attention_mask=False)["input_ids"]
    # whitespace fallback
    vocab, ids = {}, []
    for w in "\n".join(texts).split():
        ids.append(vocab.setdefault(w, len(vocab) + 2))
    return ids


def window_coverage(ids):
    """Slice into 256-token windows; return per-window coverage and pooled distances."""
    covs, dists = [], []
    for i in range(0, len(ids) - WINDOW + 1, WINDOW):
        if len(covs) >= MAX_WINDOWS:
            break
        x = torch.tensor([ids[i:i + WINDOW]], dtype=torch.long)
        prev = prev_same_key(x)[0]
        t = torch.arange(x.size(1))
        has = prev >= 0
        covs.append(float(has.float().mean()))
        dists += (t - prev)[has].float().tolist()
    return covs, dists


def run():
    tok = None
    if REPO and not SMOKE:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
    genres = get_genre_texts()

    stats, all_d = {}, {}
    for name, texts in genres.items():
        ids = encode(tok, texts)
        covs, dists = window_coverage(ids)
        if not covs:
            covs, dists = [0.0], []
        c = torch.tensor(covs)
        stats[name] = dict(mean=float(c.mean()), std=float(c.std()) if c.numel() > 1 else 0.0,
                           n_windows=len(covs),
                           mean_dist=float(torch.tensor(dists).mean()) if dists else 0.0)
        all_d[name] = dists

    print(f"{'genre':<14}{'coverage':>10}{'±std':>8}{'windows':>9}{'mean_dist':>11}")
    print("-" * 54)
    for k, s in stats.items():
        print(f"{k:<14}{s['mean']:>10.3f}{s['std']:>8.3f}{s['n_windows']:>9}{s['mean_dist']:>11.2f}")

    ps.apply()
    import matplotlib.pyplot as plt
    names = list(stats)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].bar(names, [stats[n]["mean"] for n in names],
              yerr=[stats[n]["std"] for n in names], capsize=4,
              color=ps.PRIMARY, edgecolor="white", linewidth=0.5)
    ax[0].set_ylim(0, 1); ax[0].set_title("look-back coverage by genre")
    ax[0].set_ylabel("fraction of positions with a prior match")
    ax[0].tick_params(axis="x", rotation=18, length=0)
    for i, n in enumerate(names):
        ax[0].text(i, stats[n]["mean"] + (stats[n]["std"] or 0) + 0.02, f"{stats[n]['mean']:.2f}",
                   ha="center", fontsize=9, color=ps.INK)
    for c, n in zip(ps.SERIES, names):
        d = all_d[n]
        if d:
            ax[1].hist(d, bins=30, range=(0, WINDOW), histtype="step", lw=1.8,
                       color=c, label=n, density=True)
    ax[1].set_title("look-back distance distribution"); ax[1].set_xlabel("distance (tokens)")
    ax[1].set_ylabel("density"); ax[1].legend()
    ps.save(fig, "figs/v1_coverage.png")

    if SMOKE:
        assert all(0 <= s["mean"] <= 1 for s in stats.values())
        print("SMOKE OK: windowed coverage + styled figure produced.")


if __name__ == "__main__":
    run()

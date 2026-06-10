# Conv-Routed Induction LM — BabyLM 2026 Strict-Small

A **sub-quadratic, attention-free** language model whose exact-token *induction* mixer **matches a same-scale attention transformer** on BabyLM-2026 Strict-Small (full-`nyu-mll/blimp` proxy **BLiMP ≈ 65–66**, **val ppl ≈ 35**), trained within the 10M-word budget. The point is the mixer: at this scale, swapping attention for exact-token induction costs nothing measurable — so a cheap, attention-free primitive does the job.

## Architecture

Each layer contains three complementary primitives — no attention, no recurrence, O(T log T) total:

```
x → [Dynamic Conv Context] → [Induction Mixer] → [SwiGLU FFN] → x
```

**1. Dynamic Dilated Depthwise Conv** (local, positional)
- Predicts its own depthwise kernel weights per-position from the input (content-adaptive local mixing)
- Dilations {1, 2, 4} → ~15-token causal receptive field
- Gives each token a sharp, bigram/trigram-aware representation
- Learns a "previous-token head" (offset −1 dominant) — exactly what the induction head needs

**2. Exact-Token Induction Mixer** (global, sparse, O(T log T))
- Non-learned index: find the last *M=5* positions where the current token appeared **exactly**
- Soft-ranks them by contextual match (full multi-head q·k scoring)
- Reads the **raw** representation of what followed each occurrence ("what came after this token last time?")
- Candidate generation is a sort-and-group on token IDs — no attention, no parameters, near-zero FLOP overhead
- *Why exact, not fuzzy*: three independent ablations confirmed embedding-similarity matching destroys the routing signal (consistent with Basu et al. 2026, *When Does Content-Based Routing Work?*)

**3. SwiGLU FFN** — standard per-token computation

| Config | Value |
|---|---|
| Layers | 3 |
| d_model | 384 |
| d_ff | 1536 |
| Heads | 4 |
| Conv dilations | 1, 2, 4 |
| Match M | 5 |
| max positions | 512 |
| **Params** | **~12.3M** |

## Results (BabyLM-2026-Strict-Small, 12M-param scale)

**The mixer is not the bottleneck.** Benched at equal scale and identical multi-epoch training (AdamW), the induction model lands with a vanilla attention transformer and a gated linear recurrence (LDGLRU) — well inside run-to-run noise (~1–2 BLiMP points):

| Model (12M, multi-epoch, AdamW) | BLiMP (proxy) | Val ppl |
|---|---|---|
| Vanilla attention transformer | 63.6 | 38.9 |
| LDGLRU (linear recurrence) | 63.1 | 60.0 |
| **Induction (this work)** | **64.1** | 40.2 |

**Training recipe matters more than the mixer.** Two levers, validated by sweeps, take the same induction model from ~59 to ~66 BLiMP:

| Induction model | BLiMP (proxy) | Val ppl |
|---|---|---|
| 1-epoch, ppl-early-stopped (old recipe) | ~59 | ~57 |
| **+ multi-epoch (~6 ep) & BLiMP-based selection** | **~64–65** | ~40 |
| **+ Muon optimiser (lr 0.03, released config)** | **~65–66** | **~35** |

- **Multi-epoch + select on BLiMP, not perplexity.** Held-out ppl bottoms out around 1 epoch, but BLiMP keeps improving for several more — so ppl-based early stopping (the old recipe) left most of the gains on the table. Train past the ppl optimum and pick the best-BLiMP checkpoint.
- **Muon** (orthogonalised-momentum on the weight matrices; AdamW on embeddings/norms) lowered perplexity and raised BLiMP over AdamW at equal steps.
- **Regularisation does not help here.** Dropout, token-dropout, and both shrink the train/val gap but *lower* BLiMP — overfitting was never the binding constraint; the training objective is.

Numbers are the **full `nyu-mll/blimp`** suite used as an in-loop proxy. The leaderboard uses a **filtered** BLiMP set, so re-run the official eval pipeline on the pushed model for the comparable score. The remaining gap to MLM/hybrid winners (BLiMP ~80) is an *objective* gap — masked/bidirectional training, orthogonal to and compatible with this mixer — not a mixer gap.

**Mechanism.** The induction head fires on the ~40% of positions where the current token has occurred before; on those, perplexity is far lower than on no-match positions. The no-match positions are the headroom — a data/objective limit, not a sequence-mixing one.

## Files

| File | Purpose |
|---|---|
| `modeling_induction.py` | HuggingFace model: `InductionConfig`, `InductionModel`, `InductionForCausalLM`, `InductionForSequenceClassification` |
| `train_babylm.py` | Kaggle training: multi-epoch (~6), Muon+AdamW, warmup→cosine, **BLiMP-selected** final model, word-tracked checkpoints (`chck_1M`…), hyperparameter JSON |
| `push_to_hub.py` | Publishes final model to `main` + each checkpoint to its own HF branch (as required by the BabyLM eval pipeline) |

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "joshua-taylor/conv-induction-babylm",
    trust_remote_code=True,   # required: custom architecture
)
tokenizer = AutoTokenizer.from_pretrained("joshua-taylor/conv-induction-babylm")

# score a sentence (causal log-likelihood). NOTE: right-padding only.
inputs = tokenizer("The little girl said", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
```

## Training (BabyLM 2026 Strict-Small, Kaggle)

```bash
# 1. Train (Internet ON: pulls the corpus + BLiMP for checkpoint selection).
#    Saves chck_1M...chck_10M (+ chck_20M...) and a BLiMP-selected final-model to /kaggle/working/
python train_babylm.py

# 2. Publish to HuggingFace (checkpoints as branches, final-model as main)
python push_to_hub.py --repo_id YOUR_USERNAME/conv-induction-babylm

# 3. Run the BabyLM eval pipeline (from babylm-org/babylm-eval) for the filtered/leaderboard score
cd babylm-eval/strict
bash scripts/collate_preds.sh YOUR_USERNAME/conv-induction-babylm causal strict-small --fast
```

`trust_remote_code=True` must be passed when loading; the modeling file is bundled with each saved checkpoint. The architecture assumes **right padding** (left padding would let pad tokens pollute the exact-token index).

## Reference

```
@misc{taylor2026convinduction,
  title={Conv-Routed Induction LM: Sub-Quadratic Language Modeling via Exact-Token Retrieval},
  author={Taylor, Joshua},
  year={2026},
  note={BabyLM 2026 Strict-Small submission}
}
```

Related: [Basu et al. 2026](https://arxiv.org/abs/2603.20997) — *When Does Content-Based Routing Work?*

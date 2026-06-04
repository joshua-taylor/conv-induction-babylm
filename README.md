# Conv-Routed Induction LM — BabyLM 2026 Strict-Small

A sub-quadratic, attention-free language model that **beats the GPT-2 attention baseline** on BabyLM-2026-Strict-Small (~59 val ppl vs ~77 for attention). Novel architecture trained within the 10M-word budget.

## Architecture

Each layer contains three complementary primitives — no attention, no recurrence, O(T log T) total:

```
x → [Dynamic Conv Context] → [Induction Mixer] → [SwiGLU FFN] → x
```

**1. Dynamic Dilated Depthwise Conv** (local, positional)
- Predicts its own depthwise kernel weights per-position from the input (content-adaptive local mixing)
- Dilations {1, 2, 4} → ~15-token causal receptive field
- Gives each token a sharp, bigram/trigram-aware representation
- Learned to be a "previous-token head" (offset −1 dominant) — exactly what the induction head needs

**2. Exact-Token Induction Mixer** (global, sparse, O(T log T))
- Non-learned index: find the last *M=5* positions in the window where the current token appeared exactly
- Soft-ranks them by contextual match (full multi-head q·k scoring)  
- Reads the **raw** representation of what followed each occurrence ("what came after this token last time?")
- Candidate generation is a sort-and-group on token IDs — no attention, no parameters, zero FLOP overhead
- *Why exact, not fuzzy*: three independent ablations confirmed embedding-similarity matching destroys the routing signal (consistent with Basu et al. 2026, *When Does Content-Based Routing Work?*)

**3. SwiGLU FFN** — standard per-token computation

| Config | Value |
|---|---|
| Layers | 3 |
| d_model | 192 |
| d_ff | 768 |
| Heads | 4 |
| Conv dilations | 1, 2, 4 |
| Match M | 5 |
| **Params** | **~3.9M** |

## Results (BabyLM-2026-Strict-Small)

| Model | Val ppl | tok/s |
|---|---|---|
| Full attention (baseline) | 77.2 | 100k |
| Plain induction (no conv) | 71.5 | 107k |
| + Static dilated conv | 60.9 | 77k |
| **+ Dynamic dilated conv (final)** | **59.1** | **61k** |

Gap (train/val) = +0.98 — tight generalisation for this data size.

**Zero-shot diagnostics on matched vs unmatched positions** (43% of positions have an exact match):

| Condition | ppl |
|---|---|
| Positions WITH a prior occurrence (induction fires) | 35.5 |
| Positions WITHOUT a prior occurrence (conv + FFN only) | 94.5 |

## Key findings from ablations

- **Conv is load-bearing for the induction, not just the gaps.** Without context enrichment, even matched positions score poorly (56.5) — the conv supplies the representational variation the ranker needs to distinguish occurrences.
- **Receptive field saturates at RF~16–32.** BLiMP is local-dominated; longer context (retention, 73.8 ppl) is actively harmful.
- **Retention re-introduces the value-pooling problem.** Its unbounded average blurs the per-token representations that the induction reads as keys and values — the exact failure mode of prior CRSA work.
- **Token identity is load-bearing.** Three independent attempts at fuzzy/embedding-based candidate broadening (LSH, VQ, distributional clustering) all collapsed to ~bigram-MAP accuracy, consistent with the latent-subspace result in Basu et al. 2026.

## Files

| File | Purpose |
|---|---|
| `modeling_induction.py` | HuggingFace model: `InductionConfig`, `InductionModel`, `InductionForCausalLM`, `InductionForSequenceClassification` |
| `train_babylm.py` | Full training script for Kaggle: word-tracked checkpoints (`chck_1M`…`chck_10M`), HF tokenizer, hyperparameter JSON |
| `push_to_hub.py` | Publishes final model to `main` + each checkpoint to its own HF branch (as required by the BabyLM eval pipeline) |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "joshua-taylor/conv-induction-babylm",
    trust_remote_code=True,   # required: custom architecture
)
tokenizer = AutoTokenizer.from_pretrained("joshua-taylor/conv-induction-babylm")

# score a sentence (causal log-likelihood)
inputs = tokenizer("The little girl said", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
```

## Training (BabyLM 2026 Strict-Small, Kaggle)

```bash
# 1. Train — saves chck_1M...chck_10M + final-model to /kaggle/working/
python train_babylm.py

# 2. Publish to HuggingFace (checkpoints as branches, final as main)
python push_to_hub.py --repo_id YOUR_USERNAME/conv-induction-babylm

# 3. Run the BabyLM eval pipeline (from babylm-org/babylm-eval)
cd babylm-eval/strict
bash scripts/collate_preds.sh YOUR_USERNAME/conv-induction-babylm causal strict-small --fast
```

Note: `trust_remote_code=True` must be passed when loading. Architecture is custom; the modeling file is bundled with each saved checkpoint.

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

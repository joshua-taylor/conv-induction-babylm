---
license: mit
language:
- en
library_name: transformers
tags:
- babylm
- babylm-2026
- strict-small
- attention-free
- sub-quadratic
- language-model
datasets:
- BabyLM-community/BabyLM-2026-Strict-Small
pipeline_tag: text-generation
---

# Conv-Routed Induction LM

A small, **attention-free, sub-quadratic** language model built for the
**BabyLM 2026 Strict-Small** track (a ~10M-word training budget). It is designed to test a
specific hypothesis: that a transformer's self-attention can be replaced by a *division of
labour* between two cheaper, complementary primitives — one for local word order, one for
exact long-range recall — and still match a same-scale attention baseline on grammar
(BLiMP) and perplexity.

> ⚠️ This card describes the **architecture** (which is stable). Exact hyperparameters,
> sizes, and headline metrics are still being iterated and live in the repo's
> `hyperparameters.json` / training logs for each revision rather than here.

## Architecture

Each layer is three residual sub-blocks; none is redundant:

1. **Dynamic Conv** — *local, positional.* A gated depthwise dilated convolution whose
   kernel weights are **predicted per position from the token itself** (content-adaptive
   local mixing, ~15-token reach). This is the "what just came before me" channel.
2. **Induction Mixer** — *global, content-based, exact.* For each token it finds the last
   *M* occurrences of the **exact same token** earlier in the sequence (a non-learned
   `O(T log T)` index — sort/scatter, no attention matrix), softly ranks those occurrences
   by how well their surrounding context matches the present with a small multi-head score,
   and **copies the raw representation of whatever token followed each one**. In short:
   *"what came after this token last time?"* A learnable sink lets it abstain. Exactness and
   token identity are load-bearing — fuzzy/hashed matching destroys the effect.
3. **SwiGLU FFN** — *per-token computation.*

The design thesis: **conv handles local order, induction handles long-range exact recall,
the FFN computes** — splitting the work that dense attention does into two parts with
sharper inductive biases and no quadratic cost.

### Why it is sub-quadratic

There is no `T × T` attention anywhere. The induction index is built with a sort and a
scatter (`O(T log T)`), and each token reads only a fixed number (*M*) of prior continuations.
Memory and compute scale near-linearly in sequence length.

## Intended use & scope

Research artifact for data-efficient language modelling and architecture studies. It is a
small model trained on a developmentally-motivated English corpus; it is **not** intended
for production use, factual question answering, or deployment. Generations are short-range
and reflect the small training budget.

## How to load

The architecture is custom, so `trust_remote_code=True` is required (the `modeling_induction.py`
file ships with every revision):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "<your-username>/conv-induction-babylm-strict-small"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)
```

The model is causal (next-token; the index only ever references earlier positions) and is
**padding-side agnostic** — positions are derived from the attention mask and pad positions
are zeroed, so both left- and right-padded batches give identical results for the real
tokens. Learning-curve checkpoints are published on branches named `chck_1M`, `chck_2M`, …

## Training data

[BabyLM 2026 Strict-Small](https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict-Small)
(~10M words of developmentally-plausible English), tokenised with a byte-level BPE vocabulary
trained on the same corpus.

## Limitations

- Small capacity and budget: limited world knowledge and short effective context.
- English, child-directed / developmental register; not representative of general web text.
- A research architecture under active iteration — treat any single revision's numbers as
  provisional.

## License

MIT. Code: https://github.com/joshua-taylor/conv-induction-babylm

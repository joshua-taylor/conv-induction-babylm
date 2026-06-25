"""
exp_b5_copy_baseline.py — is it just a cache?  (B5, the deflationary null)
==========================================================================

A non-parametric baseline that uses the SAME exact-match index as the model but with no
learned ranker / conv / FFN: at each position, look back to prior occurrences of the current
token and predict the continuation. Two variants:
  * recency  : copy the continuation of the most recent prior occurrence
  * majority : majority vote over the continuations of the last M occurrences

We report next-token top-1 accuracy on positions that HAVE a match (coverage), and overall.
If HF_MODEL_ID is set, we also report the trained model's accuracy on the same positions.

  PREDICTION : the learned model >> copy baselines (especially overall and on structure),
               i.e. the learned ranker/conv/FFN add real value beyond retrieval-and-copy.
  FALSIFIER  : the copy baseline matches the model -> it really is "just a cache".

Run:  HF_MODEL_ID=you/repo python exp_b5_copy_baseline.py   |   SMOKE=1 python exp_b5_copy_baseline.py
"""
import os, math
import torch
import torch.nn.functional as F
from modeling_induction import prev_same_key, build_chain

SMOKE = bool(os.environ.get("SMOKE"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO = os.environ.get("HF_MODEL_ID")
T = 64 if SMOKE else 256
B = 16 if SMOKE else 32
N_SEQ = 64 if SMOKE else 1024
M = 5


def copy_predictions(x, M):
    """Return per-position predictions for recency & majority, plus a has-match mask.
    Prediction at t = continuation token of a prior occurrence of x[t]."""
    B, T = x.shape
    dev = x.device
    prev = prev_same_key(x)                                   # (B,T) most recent prior occ
    cand = build_chain(prev, M)                               # (B,T,M)
    ti = torch.arange(T, device=dev)[None, :, None]
    nxt = cand + 1
    ok = (cand >= 0) & (nxt < ti)                             # valid continuation strictly before t
    has = ok[..., 0]                                          # recency match exists (1st chain link)
    nxt_c = nxt.clamp(0, T - 1)
    cont_tok = torch.gather(x.unsqueeze(1).expand(B, T, M), 1,
                            torch.zeros_like(nxt_c)) if False else None
    # gather continuation tokens at nxt for each candidate
    cont_tok = torch.gather(x.unsqueeze(2).expand(B, T, M), 1, nxt_c)   # (B,T,M) tokens at nxt
    cont_tok = cont_tok.masked_fill(~ok, -1)
    recency = cont_tok[..., 0]                                # most recent occurrence's continuation
    # majority vote over valid continuations
    maj = torch.full((B, T), -1, dtype=torch.long, device=dev)
    for b in range(B):
        for t in range(T):
            vals = cont_tok[b, t][ok[b, t]]
            if vals.numel():
                u, c = vals.unique(return_counts=True)
                maj[b, t] = u[c.argmax()]
    return recency, maj, has


def make_data(tok):
    if SMOKE or tok is None:
        g = torch.Generator().manual_seed(7)
        # planted repeats so matches exist
        x = torch.randint(2, 64, (N_SEQ, T), generator=g)
        for b in range(N_SEQ):
            for t in range(4, T):
                if torch.rand(1, generator=g).item() < 0.4:
                    s = torch.randint(1, t, (1,), generator=g).item(); x[b, t] = x[b, s]
        return x.to(DEVICE), 64
    from datasets import load_dataset
    ds = load_dataset("BabyLM-community/BabyLM-2026-Strict-Small", split="train", streaming=True)
    ids, buf = [], []
    for r, _ in zip(ds, range(4000)):
        buf += tok(r.get("text") or r.get("content") or "", return_attention_mask=False)["input_ids"]
        while len(buf) >= T:
            ids.append(buf[:T]); buf = buf[T:]
        if len(ids) >= N_SEQ:
            break
    return torch.tensor(ids[:N_SEQ], dtype=torch.long, device=DEVICE), tok.vocab_size


def run():
    tok = None
    model = None
    if REPO and not SMOKE:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(REPO, trust_remote_code=True).to(DEVICE).eval()
    x, vocab = make_data(tok)

    tgt = x[:, 1:]                                            # realised next token (positions 0..T-2 predict)
    rec_all, maj_all, has_all = [], [], []
    model_correct = []
    for bi in range(0, x.size(0), B):
        xb = x[bi:bi + B]
        rec, maj, has = copy_predictions(xb, M)
        rec_all.append(rec[:, :-1]); maj_all.append(maj[:, :-1]); has_all.append(has[:, :-1])
        if model is not None:
            with torch.no_grad():
                pred = model(input_ids=xb).logits[:, :-1].argmax(-1)
            model_correct.append((pred == xb[:, 1:]))
    rec = torch.cat([r for r in rec_all], 0); maj = torch.cat(maj_all, 0); has = torch.cat(has_all, 0)
    y = tgt

    def acc(pred, mask):
        m = mask & (pred >= 0)
        return float(((pred == y) & m).sum()) / float(m.sum().clamp(min=1))

    cov = float(has.sum()) / has.numel()
    print(f"data: {x.size(0)} seqs x {T} | vocab {vocab} | match coverage = {cov:.3f}")
    print(f"\n{'baseline':<22}{'acc | match':>14}{'acc | all':>12}")
    print("-" * 50)
    allmask = torch.ones_like(has)
    print(f"{'copy: recency':<22}{acc(rec, has):>14.3f}{acc(rec, allmask):>12.3f}")
    print(f"{'copy: majority':<22}{acc(maj, has):>14.3f}{acc(maj, allmask):>12.3f}")
    if model is not None:
        mc = torch.cat(model_correct, 0)
        am = float(mc.sum()) / mc.numel()
        mm = float((mc & has).sum()) / float(has.sum().clamp(min=1))
        print(f"{'LEARNED model':<22}{mm:>14.3f}{am:>12.3f}")
        print("\n=> gap (model - best copy) is the value of the learned ranker/conv/FFN.")
    else:
        print("\n(set HF_MODEL_ID to add the learned model's row for the head-to-head.)")

    if SMOKE:
        assert 0 <= acc(rec, has) <= 1 and 0 <= cov <= 1
        print("\nSMOKE OK: copy baselines computed over the exact-match index.")


if __name__ == "__main__":
    run()

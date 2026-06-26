"""
mech_common.py — shared machinery for the mechanistic "why" experiments
=======================================================================

The induction index is *explicit*: InductionModel computes `cand` (B,T,M), the last M
prior positions sharing the current token, and the mixer copies the RAW continuation at
`cand+1`. That makes causal interventions easy — we know every edge a priori, with no head
search. This module provides:

  * load_model            — the trained HF model (trust_remote_code) or a tiny SMOKE model
  * MixerCapture          — capture each layer's residual-stream input to the mixer + `cand`
  * recompute_mixer_S     — re-derive per-candidate attention mass from a captured input
                            (lets us rank edges by how much the mixer actually used them)
  * select_top_edges      — per (b,t): the highest-mass *long-range* edge (t -> source)
  * PatchMixerInput       — overwrite the mixer's input residual at chosen positions
                            (used to corrupt a copied source and watch the query change)
  * AblateConv            — zero the DynamicConv contribution (test what the conv feeds in)
  * synthetic_induction_batch / tokenize_corpus — data

Run modes: set SMOKE=1 for a tiny random CPU model + synthetic data (harness test).
"""

import os, math
import torch
import torch.nn as nn
import torch.nn.functional as F

SMOKE = bool(os.environ.get("SMOKE"))
NEG = float("-inf")


# ----------------------------------------------------------------------
# model / structure access
# ----------------------------------------------------------------------
def load_model(repo=None, device="cpu", dtype=torch.float32):
    """Tiny random model under SMOKE; otherwise the trained HF model via trust_remote_code."""
    if SMOKE or repo is None:
        import modeling_induction as M
        cfg = M.InductionConfig(vocab_size=64, d_model=64, d_ff=128, n_layers=3, n_heads=4,
                                max_position_embeddings=128, dyn_groups=4, match_m=5,
                                conv_dilations=(1, 2, 4), conv_kernel=3)
        model = M.InductionForCausalLM(cfg).to(device).eval()
        return model, None, cfg
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True,
                                                 torch_dtype=dtype).to(device).eval()
    return model, tok, model.config


def get_base(model):
    return model.model if hasattr(model, "model") else model


def get_layers(model):
    return get_base(model).layers


# ----------------------------------------------------------------------
# capture the residual stream entering each mixer (+ the shared `cand`)
# ----------------------------------------------------------------------
class MixerCapture:
    def __init__(self, model):
        self.layers = get_layers(model)
        self.x = [None] * len(self.layers)
        self.cand = None
        self._h = []

    def __enter__(self):
        for i, layer in enumerate(self.layers):
            def pre(mod, args, idx=i):
                self.x[idx] = args[0].detach()
                if self.cand is None:
                    self.cand = args[1].detach()
                return None
            self._h.append(layer.mixer.register_forward_pre_hook(pre))
        return self

    def __exit__(self, *a):
        for h in self._h:
            h.remove()
        self._h = []


# ----------------------------------------------------------------------
# re-derive per-candidate attention mass from a captured mixer input
# (mirrors InductionMixer.forward exactly; verified in the smoke test)
# ----------------------------------------------------------------------
def recompute_mixer_S(mixer, x, cand):
    from modeling_induction import gather_cand
    B, T, d = x.shape
    H, dh = mixer.H, mixer.dh
    C = cand.size(-1)
    dev = x.device
    h = mixer.norm(x)
    ti = torch.arange(T, device=dev)[None, :, None]
    q = mixer.q(h).view(B, T, H, dh)
    valid = cand >= 0
    nxt = cand + 1
    ok = valid & (nxt < ti)
    kk = mixer.k(gather_cand(h, cand)).view(B, T, C, H, dh)
    sc = torch.einsum("bthd,btchd->btch", q, kk) / math.sqrt(dh)
    sc = sc.masked_fill(~ok.unsqueeze(-1), NEG)
    ssink = torch.einsum("bthd,hd->bth", q, mixer.sink_k) / math.sqrt(dh)
    S = torch.cat([sc, ssink.unsqueeze(2)], 2).softmax(2)        # (B,T,C+1,H)
    mass = S[..., :C, :].mean(-1)                                # (B,T,C) mean over heads
    return S, mass, ok, nxt


def mixer_output(mixer, x, cand):
    """Full mixer forward, recomputed (for the smoke equivalence check)."""
    from modeling_induction import gather_cand
    B, T, d = x.shape
    H, dh = mixer.H, mixer.dh
    C = cand.size(-1)
    S, mass, ok, nxt = recompute_mixer_S(mixer, x, cand)
    h = mixer.norm(x)
    vv = mixer.v(gather_cand(h, nxt.clamp(0, T - 1))).view(B, T, C, H, dh)
    Vv = torch.cat([vv, mixer.sink_v[None, None, None].expand(B, T, 1, H, dh)], 2)
    out = torch.einsum("btch,btchd->bthd", S, Vv).reshape(B, T, d)
    return mixer.o(out)


# ----------------------------------------------------------------------
# pick, per (b,t), the highest-mass edge whose source is "long-range"
# (so the only path from source to query is the induction edge, not the conv)
# ----------------------------------------------------------------------
def select_top_edges(mass, cand, ok, min_dist, t_min=8):
    """Returns per-batch selection: (b_idx, t_sel, src_pos, cont_pos, mass_sel).
    One edge per sequence (the strongest qualifying one)."""
    B, T, C = mass.shape
    dev = mass.device
    nxt = cand + 1                                               # continuation position
    ti = torch.arange(T, device=dev)[None, :, None]
    dist = ti - nxt                                              # query - continuation
    good = ok & (dist >= min_dist) & (ti >= t_min)
    m = mass.masked_fill(~good, -1.0)                            # (B,T,C)
    flat = m.view(B, -1)
    best = flat.argmax(-1)                                       # (B,)
    best_val = flat.gather(1, best[:, None]).squeeze(1)
    t_sel = best // C
    c_sel = best % C
    src = cand[torch.arange(B, device=dev), t_sel, c_sel]
    cont = src + 1
    keep = best_val > 0                                          # had >=1 qualifying edge
    return dict(b=torch.arange(B, device=dev)[keep], t=t_sel[keep],
                src=src[keep], cont=cont[keep], mass=best_val[keep])


def distance_matched_control(input_ids, b, t, cont, cand, tol=4):
    """For each selected edge pick a control continuation position at ~same distance from t
    that is NOT one of t's candidates and carries a different token than the copied one."""
    B, T = input_ids.shape
    dev = input_ids.device
    ctrl = torch.full_like(cont, -1)
    cand_set = cand[b, t]                                        # (n, C) source positions
    for i in range(b.numel()):
        bi, ti, ci = int(b[i]), int(t[i]), int(cont[i])
        target_d = ti - ci
        y_star = int(input_ids[bi, ci])
        forbidden = set((cand_set[i] + 1).tolist()) | {ci}
        cands = []
        for dd in range(0, tol + 1):
            for s in (ti - target_d + dd, ti - target_d - dd):
                if 1 <= s < ti and s not in forbidden and int(input_ids[bi, s]) != y_star:
                    cands.append(s)
            if cands:
                break
        ctrl[i] = cands[0] if cands else max(1, ti - target_d)
    return ctrl


# ----------------------------------------------------------------------
# patch the mixer input at layer L, positions (b,pos) -> replacement vector
# ----------------------------------------------------------------------
class PatchMixerInput:
    def __init__(self, model, layer_idx, b_idx, positions, replacement):
        self.mixer = get_layers(model)[layer_idx].mixer
        self.b = b_idx
        self.pos = positions
        self.repl = replacement                                 # (d,) or (n,d)
        self._h = None

    def __enter__(self):
        def pre(mod, args):
            x, cand = args[0], args[1]
            x = x.clone()
            x[self.b, self.pos] = self.repl.to(x.dtype)
            return (x, cand)
        self._h = self.mixer.register_forward_pre_hook(pre)
        return self

    def __exit__(self, *a):
        self._h.remove()


class AblateConv:
    """Zero the DynamicConv contribution in every block (x = x + conv(x) -> x = x + 0)."""
    def __init__(self, model):
        self.layers = get_layers(model)
        self._h = []

    def __enter__(self):
        for layer in self.layers:
            def hook(mod, inp, out):
                return torch.zeros_like(out)
            self._h.append(layer.conv.register_forward_hook(hook))
        return self

    def __exit__(self, *a):
        for h in self._h:
            h.remove()
        self._h = []


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
def synthetic_induction_batch(B, T, vocab, device, seed=0, repeat_p=0.4):
    """Random tokens with planted repeats so the exact-match index has plenty of edges."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randint(2, vocab, (B, T), generator=g)
    # plant: with prob repeat_p, copy a token (and its successor) from earlier -> induction
    for b in range(B):
        for t in range(4, T - 1):
            if torch.rand(1, generator=g).item() < repeat_p:
                s = torch.randint(1, t - 1, (1,), generator=g).item()
                x[b, t] = x[b, s]                                # same token => edge s->t
    return x.to(device)


def tokenize_corpus(tok, texts, T, max_seqs=512, device="cpu"):
    """Pack the whole corpus into one token stream, then slice into T-length windows.
    (Previously short documents were dropped, which starved the long-range analyses.)"""
    stream = []
    for txt in texts:
        if not txt or not txt.strip():
            continue
        stream.extend(tok(txt, return_attention_mask=False)["input_ids"])
        if len(stream) >= max_seqs * T:
            break
    n = min(len(stream) // T, max_seqs)
    if n == 0:
        return torch.zeros(0, T, dtype=torch.long, device=device)
    ids = [stream[i * T:(i + 1) * T] for i in range(n)]
    return torch.tensor(ids, dtype=torch.long, device=device)

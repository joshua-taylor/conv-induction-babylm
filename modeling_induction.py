"""
modeling_induction.py — Conv-Routed Induction LM (HuggingFace-format, BabyLM 2026)
==================================================================================

A sub-quadratic, attention-free language model. Each layer is three complementary
primitives:

  1. DYNAMIC CONV  (positional, local)  — a gated depthwise dilated conv whose kernel
     weights are predicted per-position from the token (content-adaptive local mixing).
     Gives every token a sharp, context-aware representation (~15-token reach).
  2. INDUCTION MIXER (content, global, EXACT) — finds the last M occurrences of the
     *exact current token* (a non-learned O(T log T) index), softly ranks them by how
     well their context matches the present, and copies the RAW representation of what
     followed each. "What came after this token last time?"
  3. SwiGLU FFN (per-token compute).

The split is the whole idea: conv = local word-order, induction = long-range exact
recall, FFN = computation. None is redundant. The released configuration is ~12.2M params
(d_model=384, d_ff=1536, 3 layers); trained multi-epoch with the Muon optimiser it reaches
~35 val ppl and BLiMP ~66 (full nyu-mll/blimp proxy) on BabyLM-2026 Strict-Small, matching a
same-scale attention baseline while remaining attention-free (see train_babylm.py / README).

Exposes three HF classes so the BabyLM eval pipeline can load it:
  InductionModel                    -> hidden states
  InductionForCausalLM              -> zero-shot (BLiMP/COMPS/entity-tracking, log-likelihoods)
  InductionForSequenceClassification-> fine-tuning (GLUE)

Causality is exact (next-token; the index only references earlier same-token positions).
NOTE: assumes RIGHT padding (left padding would let pad tokens pollute the index).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import (
    BaseModelOutput, CausalLMOutput, SequenceClassifierOutput,
)


# ======================================================================
# Config
# ======================================================================
class InductionConfig(PretrainedConfig):
    model_type = "induction_lm"

    def __init__(
        self,
        vocab_size: int = 8000,
        d_model: int = 384,
        d_ff: int = 1536,
        n_layers: int = 3,
        n_heads: int = 4,
        max_position_embeddings: int = 512,
        conv_dilations=(1, 2, 4),
        conv_kernel: int = 3,
        dyn_groups: int = 16,
        match_m: int = 5,
        num_labels: int = 2,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_position_embeddings = max_position_embeddings
        self.conv_dilations = list(conv_dilations)
        self.conv_kernel = conv_kernel
        self.dyn_groups = dyn_groups
        self.match_m = match_m
        super().__init__(
            num_labels=num_labels,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


# ======================================================================
# Primitive helpers
# ======================================================================
class RMSNorm(nn.Module):
    def __init__(self, c, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c)); self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def prev_same_key(k):
    """k:(B,T) token ids -> prev[t] = most recent s<t with k[s]==k[t], else -1. Causal."""
    B, T = k.shape
    order = k.argsort(dim=1, stable=True)
    ks = torch.gather(k, 1, order)
    prev_pos = F.pad(order, (1, -1), value=-1)
    prev_key = F.pad(ks, (1, -1), value=-1)
    prev_in_sorted = torch.where(ks == prev_key, prev_pos, torch.full_like(prev_pos, -1))
    prev = torch.empty_like(k)
    prev.scatter_(1, order, prev_in_sorted)
    return prev


def build_chain(prev1, M):
    """Last M occurrence positions sharing the token, by chaining prev. (B,T,M)."""
    cands = [prev1]; cur = prev1
    for _ in range(M - 1):
        older = torch.gather(prev1, 1, cur.clamp(min=0))
        cur = torch.where(cur >= 0, older, torch.full_like(cur, -1))
        cands.append(cur)
    return torch.stack(cands, -1)


def gather_cand(h, cand):
    """h:(B,T,d), cand:(B,T,C) -> (B,T,C,d) reps at candidate positions."""
    B, T, d = h.shape; C = cand.size(-1)
    return torch.gather(h.unsqueeze(2).expand(B, T, C, d), 1,
                        cand.clamp(min=0).unsqueeze(-1).expand(B, T, C, d))


# ======================================================================
# Layer sublayers
# ======================================================================
class DynamicConv(nn.Module):
    """Gated dilated depthwise conv with per-position, content-predicted kernels."""
    def __init__(self, d, dils, k, groups):
        super().__init__()
        self.norm = RMSNorm(d); self.k = k; self.dils = dils
        self.G = groups; self.cpg = d // groups
        self.kproj = nn.ModuleList([nn.Linear(d, groups * k) for _ in dils])
        self.gates = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in dils])
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x):
        h = self.norm(x); B, T, d = h.shape
        cur = h
        for proj, gate, di in zip(self.kproj, self.gates, self.dils):
            kw = proj(cur).view(B, T, self.G, self.k).softmax(-1)
            kw = kw.repeat_interleave(self.cpg, dim=2)                  # (B,T,d,k)
            taps = []
            for j in range(self.k):
                shift = di * (self.k - 1 - j)
                taps.append(F.pad(cur, (0, 0, shift, 0))[:, :T] if shift else cur)
            out = (torch.stack(taps, -1) * kw).sum(-1)                  # (B,T,d)
            cur = cur + torch.sigmoid(gate(out)) * out
        return self.o(cur)


class InductionMixer(nn.Module):
    """Exact-token retrieval with full multi-head soft ranking over M occurrences.
       value = RAW representation of the continuation (cand+1)."""
    def __init__(self, d, H, M):
        super().__init__()
        self.H = H; self.dh = d // H; self.M = M
        self.norm = RMSNorm(d)
        self.q = nn.Linear(d, d, bias=False); self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False); self.o = nn.Linear(d, d, bias=False)
        self.sink_k = nn.Parameter(torch.randn(H, self.dh) * 0.02)
        self.sink_v = nn.Parameter(torch.randn(H, self.dh) * 0.02)

    def forward(self, x, cand):
        B, T, d = x.shape; H, dh = self.H, self.dh; dev = x.device; C = cand.size(-1)
        h = self.norm(x); ti = torch.arange(T, device=dev)[None, :, None]
        q = self.q(h).view(B, T, H, dh)
        valid = cand >= 0; nxt = cand + 1; ok = valid & (nxt < ti)
        kk = self.k(gather_cand(h, cand)).view(B, T, C, H, dh)
        vv = self.v(gather_cand(h, nxt.clamp(0, T - 1))).view(B, T, C, H, dh)
        sc = torch.einsum('bthd,btchd->btch', q, kk) / math.sqrt(dh)
        sc = sc.masked_fill(~ok.unsqueeze(-1), float('-inf'))
        ssink = torch.einsum('bthd,hd->bth', q, self.sink_k) / math.sqrt(dh)
        S = torch.cat([sc, ssink.unsqueeze(2)], 2).softmax(2)
        Vv = torch.cat([vv, self.sink_v[None, None, None].expand(B, T, 1, H, dh)], 2)
        out = torch.einsum('btch,btchd->bthd', S, Vv).reshape(B, T, d)
        return self.o(out)


class SwiGLU(nn.Module):
    def __init__(self, d, dff):
        super().__init__()
        self.norm = RMSNorm(d)
        self.g = nn.Linear(d, dff, bias=False); self.u = nn.Linear(d, dff, bias=False)
        self.dn = nn.Linear(dff, d, bias=False)

    def forward(self, x):
        h = self.norm(x)
        return self.dn(F.silu(self.g(h)) * self.u(h))


class InductionBlock(nn.Module):
    def __init__(self, cfg: InductionConfig):
        super().__init__()
        self.conv = DynamicConv(cfg.d_model, cfg.conv_dilations, cfg.conv_kernel, cfg.dyn_groups)
        self.mixer = InductionMixer(cfg.d_model, cfg.n_heads, cfg.match_m)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x, cand):
        x = x + self.conv(x)
        x = x + self.mixer(x, cand)
        x = x + self.ffn(x)
        return x


# ======================================================================
# Base model
# ======================================================================
class InductionPreTrainedModel(PreTrainedModel):
    config_class = InductionConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["InductionBlock"]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)


class InductionModel(InductionPreTrainedModel):
    def __init__(self, config: InductionConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.pos = nn.Parameter(torch.zeros(1, config.max_position_embeddings, config.d_model))
        self.layers = nn.ModuleList([InductionBlock(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, input_ids, attention_mask=None, **kwargs):
        B, T = input_ids.shape
        Tp = min(T, self.config.max_position_embeddings)
        cand = build_chain(prev_same_key(input_ids), self.config.match_m)
        x = self.embed_tokens(input_ids)
        x = x + self.pos[:, :Tp] if T <= Tp else x + F.pad(self.pos, (0, 0, 0, T - Tp))[:, :T]
        for layer in self.layers:
            x = layer(x, cand)
        return BaseModelOutput(last_hidden_state=self.norm(x))


# ======================================================================
# Causal LM head (zero-shot: BLiMP, COMPS, entity-tracking)
# ======================================================================
class InductionForCausalLM(InductionPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: InductionConfig):
        super().__init__(config)
        self.model = InductionModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new):
        self.lm_head = new

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        hidden = self.model(input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1), ignore_index=-100,
            )
        return CausalLMOutput(loss=loss, logits=logits)


# ======================================================================
# Sequence-classification head (fine-tuning: GLUE)
# ======================================================================
class InductionForSequenceClassification(InductionPreTrainedModel):
    def __init__(self, config: InductionConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = InductionModel(config)
        self.score = nn.Linear(config.d_model, config.num_labels, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        hidden = self.model(input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.score(hidden)                                    # (B,T,num_labels)
        # pool the last non-pad token (GPT-2-style), right padding assumed
        if attention_mask is not None:
            lengths = attention_mask.long().sum(-1) - 1
        elif self.config.pad_token_id is not None:
            lengths = (input_ids != self.config.pad_token_id).long().sum(-1) - 1
        else:
            lengths = torch.full((input_ids.size(0),), input_ids.size(1) - 1, device=input_ids.device)
        lengths = lengths.clamp(min=0)
        pooled = logits[torch.arange(input_ids.size(0), device=input_ids.device), lengths]
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = F.mse_loss(pooled.squeeze(-1), labels.float())
            else:
                loss = F.cross_entropy(pooled, labels)
        return SequenceClassifierOutput(loss=loss, logits=pooled)


# Register for AutoConfig / AutoModel* so the eval pipeline can load with trust_remote_code
InductionConfig.register_for_auto_class()
InductionModel.register_for_auto_class("AutoModel")
InductionForCausalLM.register_for_auto_class("AutoModelForCausalLM")
InductionForSequenceClassification.register_for_auto_class("AutoModelForSequenceClassification")

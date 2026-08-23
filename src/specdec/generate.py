"""Incremental, KV-cached inference and speculative decoding.

Training (``model.py``) builds a ``Tensor`` autodiff graph over a full
sequence because it needs gradients. Generation needs neither gradients nor
a full re-run over the whole sequence every step, so this module reimplements
the same block math directly on plain NumPy arrays, with a key/value cache
so each new token costs one position's worth of work instead of the whole
prefix. It shares ``functional.py``'s ``softmax``/``layer_norm``/``gelu``
(those work on plain arrays too, see that module's docstring) rather than
duplicating the elementwise math; only the shape/cache bookkeeping differs
from ``model.Block``.

``forward_step`` is the one function both plain autoregressive decoding and
speculative decoding are built from: given ``K`` new token ids and the
cache of everything before them, it returns logits for all ``K`` new
positions from a *single* batched forward pass, extending the cache by
``K`` in one shot. ``K = 1`` is ordinary one-token-at-a-time decoding.
``K > 1`` is exactly what makes speculative decoding fast: verifying a
whole block of draft tokens costs about the same as generating one token,
because it's one matmul-shaped forward pass over ``K`` positions, not
``K`` sequential ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .functional import gelu, layer_norm, masked_fill, softmax
from .model import GPTConfig


@dataclass
class KVCache:
    n_layers: int
    k: list = field(default_factory=list)
    v: list = field(default_factory=list)
    length: int = 0

    def __post_init__(self):
        if not self.k:
            self.k = [None] * self.n_layers
            self.v = [None] * self.n_layers

    def clone(self) -> KVCache:
        return KVCache(
            self.n_layers,
            k=[None if x is None else x.copy() for x in self.k],
            v=[None if x is None else x.copy() for x in self.v],
            length=self.length,
        )


def _step_mask(cache_len: int, k: int) -> np.ndarray:
    """``(k, cache_len + k)`` bool array: row i may attend to cached
    positions and to new positions ``0..i``, nothing further ahead."""
    total = cache_len + k
    i = np.arange(k)[:, None]
    j = np.arange(total)[None, :]
    return j > (cache_len + i)


def forward_step(weights: dict, cfg: GPTConfig, new_ids: np.ndarray, cache: KVCache) -> np.ndarray:
    """Run ``new_ids`` (shape ``(K,)``) through the model, extending ``cache``
    in place. Returns logits of shape ``(K, vocab_size)``."""
    k_new = len(new_ids)
    cache_len = cache.length
    positions = np.arange(cache_len, cache_len + k_new)
    x = weights["token_emb"][new_ids] + weights["pos_emb"][positions]

    mask = _step_mask(cache_len, k_new)
    h, dh = cfg.n_heads, cfg.d_head

    for li, blk in enumerate(weights["blocks"]):
        normed = layer_norm(x, blk["ln1_g"], blk["ln1_b"])
        q = normed @ blk["wq"] + blk["bq"]
        new_k = normed @ blk["wk"] + blk["bk"]
        new_v = normed @ blk["wv"] + blk["bv"]

        full_k = new_k if cache.k[li] is None else np.concatenate([cache.k[li], new_k], axis=0)
        full_v = new_v if cache.v[li] is None else np.concatenate([cache.v[li], new_v], axis=0)
        cache.k[li], cache.v[li] = full_k, full_v

        total = cache_len + k_new
        qh = q.reshape(k_new, h, dh).transpose(1, 0, 2)
        kh = full_k.reshape(total, h, dh).transpose(1, 0, 2)
        vh = full_v.reshape(total, h, dh).transpose(1, 0, 2)

        scores = (qh @ kh.transpose(0, 2, 1)) / np.sqrt(dh)
        scores = masked_fill(scores, mask[None, :, :], -1e9)
        attn = softmax(scores, axis=-1) @ vh
        attn = attn.transpose(1, 0, 2).reshape(k_new, h * dh)
        x = x + (attn @ blk["wo"] + blk["bo"])

        normed2 = layer_norm(x, blk["ln2_g"], blk["ln2_b"])
        ffn = gelu(normed2 @ blk["w1"] + blk["b1"]) @ blk["w2"] + blk["b2"]
        x = x + ffn

    cache.length = cache_len + k_new
    x = layer_norm(x, weights["ln_f_g"], weights["ln_f_b"])
    return x @ weights["token_emb"].T


# -- sampling -----------------------------------------------------------------


def probs_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return softmax(logits / max(temperature, 1e-6), axis=-1)


def sample(probs: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(len(probs), p=probs))


# -- plain autoregressive decoding ---------------------------------------------


def generate_naive(weights, cfg, prompt_ids, n_tokens, rng, temperature=1.0):
    """One token at a time, no speculation. Returns the full id sequence."""
    cache = KVCache(cfg.n_layers)
    ids = list(prompt_ids)
    logits = forward_step(weights, cfg, np.array(prompt_ids), cache)
    next_probs = probs_from_logits(logits[-1], temperature)
    for _ in range(n_tokens):
        next_id = sample(next_probs, rng)
        ids.append(next_id)
        logits = forward_step(weights, cfg, np.array([next_id]), cache)
        next_probs = probs_from_logits(logits[-1], temperature)
    return ids


# -- speculative decoding -------------------------------------------------------


@dataclass
class SpeculativeStats:
    proposed: int = 0
    accepted: int = 0
    main_forward_calls: int = 0
    draft_forward_calls: int = 0


def speculative_accept_or_resample(
    token: int, p_draft: np.ndarray, p_main: np.ndarray, rng: np.random.Generator
) -> tuple[int, bool]:
    """The core accept/reject/resample rule from speculative decoding.

    ``token`` was sampled from ``p_draft``; accept it with probability
    ``min(1, p_main[token]/p_draft[token])``. On rejection, replace it with a
    sample from the residual distribution ``normalize(max(0, p_main - p_draft))``.
    Returns ``(output_token, was_accepted)``. This is deliberately the only
    place that rule is implemented: ``generate_speculative`` calls it once
    per draft token, and ``tests/test_speculative.py`` calls it directly (with
    hand-picked distributions, no model involved) to check that the token it
    returns is distributed exactly as ``p_main``, regardless of ``p_draft``.
    """
    accept_prob = min(1.0, p_main[token] / max(p_draft[token], 1e-12))
    if rng.random() < accept_prob:
        return token, True
    residual = np.clip(p_main - p_draft, 0.0, None)
    total = residual.sum()
    residual = residual / total if total > 0 else p_main
    return sample(residual, rng), False


def generate_speculative(
    draft_weights,
    draft_cfg,
    main_weights,
    main_cfg,
    prompt_ids,
    n_tokens: int,
    rng: np.random.Generator,
    lookahead: int = 4,
    temperature: float = 1.0,
    on_token=None,
) -> tuple[list[int], SpeculativeStats]:
    """Speculative decoding (Leviathan et al. 2023 / Chen et al. 2023):
    the draft model proposes ``lookahead`` tokens autoregressively, the main
    model verifies all of them in one batched forward pass, and each draft
    token is accepted with probability ``min(1, p_main/p_draft)``; on the
    first rejection the rest of the block is replaced by one sample from the
    *residual* distribution ``normalize(max(0, p_main - p_draft))``. This
    keeps the output distribution identical to sampling from the main model
    alone (``tests/test_speculative.py`` checks that exactly), while costing
    one main-model forward pass per accepted block instead of one per token.

    ``on_token(token_id, kind)`` fires once per output token, with ``kind``
    one of ``"accepted"`` (fast-forwarded from the draft), ``"corrected"``
    (the draft's proposal was rejected and replaced), or ``"bonus"`` (every
    draft token in the block was accepted, so one extra token came free from
    the main model). ``live_generate.py`` uses this to color-code the output.
    """
    stats = SpeculativeStats()
    ids = list(prompt_ids)
    draft_cache = KVCache(draft_cfg.n_layers)
    main_cache = KVCache(main_cfg.n_layers)

    # Prime both caches on the prompt.
    draft_logits = forward_step(draft_weights, draft_cfg, np.array(prompt_ids), draft_cache)
    main_logits = forward_step(main_weights, main_cfg, np.array(prompt_ids), main_cache)
    stats.draft_forward_calls += 1
    stats.main_forward_calls += 1
    next_draft_probs = probs_from_logits(draft_logits[-1], temperature)
    next_main_probs = probs_from_logits(main_logits[-1], temperature)

    while len(ids) - len(prompt_ids) < n_tokens:
        k = min(lookahead, n_tokens - (len(ids) - len(prompt_ids)))

        # Snapshot both caches at the start of the block: whatever happens
        # during proposal/verification below gets discarded and rebuilt from
        # here once we know how many draft tokens were actually accepted.
        draft_checkpoint = draft_cache.clone()
        main_checkpoint = main_cache.clone()

        draft_tokens = []
        draft_probs_seq = [next_draft_probs]
        for _ in range(k):
            tok = sample(draft_probs_seq[-1], rng)
            draft_tokens.append(tok)
            logits = forward_step(draft_weights, draft_cfg, np.array([tok]), draft_cache)
            stats.draft_forward_calls += 1
            draft_probs_seq.append(probs_from_logits(logits[-1], temperature))

        # Verify all k draft tokens in one batched forward pass (plus one
        # extra distribution "for free" if every token is accepted).
        verify_logits = forward_step(main_weights, main_cfg, np.array(draft_tokens), main_cache)
        stats.main_forward_calls += 1
        main_probs_seq = [next_main_probs] + [probs_from_logits(verify_logits[i], temperature) for i in range(k)]

        stats.proposed += k
        n_accepted = 0
        rejected = False
        for i in range(k):
            output_token, accepted = speculative_accept_or_resample(
                draft_tokens[i], draft_probs_seq[i], main_probs_seq[i], rng
            )
            ids.append(output_token)
            if on_token:
                on_token(output_token, kind="accepted" if accepted else "corrected")
            if accepted:
                n_accepted += 1
            else:
                rejected = True
                break
        stats.accepted += n_accepted

        if rejected:
            # draft_cache/main_cache went on to propose/verify tokens beyond
            # the rejection point that never made it into `ids`. Roll back
            # to the checkpoint and redo through exactly what was accepted
            # plus the correction, so both caches match `ids` going forward.
            accepted_and_final = ids[draft_checkpoint.length :]
            draft_cache = draft_checkpoint
            main_cache = main_checkpoint
            d_logits = forward_step(draft_weights, draft_cfg, np.array(accepted_and_final), draft_cache)
            m_logits = forward_step(main_weights, main_cfg, np.array(accepted_and_final), main_cache)
            stats.draft_forward_calls += 1
            stats.main_forward_calls += 1
        else:
            # Every draft token was accepted, so draft_cache (from the
            # proposal loop) and main_cache (from the batched verify call)
            # already correctly reflect all k of them. Recomputing from the
            # checkpoint here would just redo work already done; only the
            # bonus token, sampled straight from the main model, is new.
            bonus = sample(main_probs_seq[k], rng)
            ids.append(bonus)
            if on_token:
                on_token(bonus, kind="bonus")
            d_logits = forward_step(draft_weights, draft_cfg, np.array([bonus]), draft_cache)
            m_logits = forward_step(main_weights, main_cfg, np.array([bonus]), main_cache)
            stats.draft_forward_calls += 1
            stats.main_forward_calls += 1

        next_draft_probs = probs_from_logits(d_logits[-1], temperature)
        next_main_probs = probs_from_logits(m_logits[-1], temperature)

    # A fully-accepted block also emits a bonus token "for free" (see above),
    # which can overshoot n_tokens by exactly one on the final block; trim
    # back to precisely what was asked for.
    return ids[: len(prompt_ids) + n_tokens], stats

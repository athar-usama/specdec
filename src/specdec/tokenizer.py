"""A byte-level BPE tokenizer, trained from scratch (no tiktoken, no
HuggingFace tokenizers). Same algorithm family as GPT-2's tokenizer:
start from the 256 raw byte values so any UTF-8 text encodes without an
unknown-token fallback, then greedily merge the most frequent adjacent
pair into a new token, repeated until the target vocabulary size is hit.
"""

from __future__ import annotations

import pickle
from pathlib import Path


class BPETokenizer:
    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # -- training -----------------------------------------------------------
    @staticmethod
    def _pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
        counts: dict[tuple[int, int], int] = {}
        for a, b in zip(ids, ids[1:], strict=False):
            counts[(a, b)] = counts.get((a, b), 0) + 1
        return counts

    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        out = []
        i, n = 0, len(ids)
        while i < n:
            if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (the raw byte alphabet)")
        ids = list(text.encode("utf-8"))
        vocab = {i: bytes([i]) for i in range(256)}
        merges: dict[tuple[int, int], int] = {}
        next_id = 256
        for step in range(vocab_size - 256):
            counts = self._pair_counts(ids)
            if not counts:
                break
            best = max(counts, key=counts.get)
            if counts[best] < 2:
                break  # no pair repeats; further merges wouldn't compress anything
            ids = self._merge(ids, best, next_id)
            merges[best] = next_id
            vocab[next_id] = vocab[best[0]] + vocab[best[1]]
            if verbose and step % 50 == 0:
                print(f"merge {step:4d}: {vocab[best[0]]!r} + {vocab[best[1]]!r} -> id {next_id}")
            next_id += 1
        self.merges = merges
        self.vocab = vocab

    # -- encode / decode ------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            counts = self._pair_counts(ids)
            candidate, candidate_rank = None, None
            for pair in counts:
                rank = self.merges.get(pair)
                if rank is not None and (candidate_rank is None or rank < candidate_rank):
                    candidate, candidate_rank = pair, rank
            if candidate is None:
                break
            ids = self._merge(ids, candidate, self.merges[candidate])
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    # -- persistence ----------------------------------------------------------
    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"merges": self.merges, "vocab": self.vocab}, f)

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        with open(path, "rb") as f:
            state = pickle.load(f)
        tok = cls()
        tok.merges = state["merges"]
        tok.vocab = state["vocab"]
        return tok

"""Trains the tokenizer, the bigger "main" model, and the small "draft"
model, all from scratch on a small bundled text corpus (``data/corpus.txt``,
an original short story written for this project. See the README for why
it isn't a downloaded/public-domain text.

The draft model is trained by **distillation** from the already-trained
main model (matching its output distribution directly, the standard way a
real speculative-decoding draft model is built) rather than independently
on the raw corpus labels. That choice isn't cosmetic: two small language
models trained independently on a few thousand tokens overfit to different
specific continuations and barely agree once generation drifts from the
literal training text, which tanks the draft-acceptance rate speculative
decoding depends on (measured at 3-7% before this change; see git history).
Distillation directly optimizes for the thing that actually matters here:
agreement with the main model. It's also a legitimate, common technique on
its own, not a trick to inflate the benchmark.

This is a from-scratch *language model* training script, not the point of
the package (that's the inference engine in ``generate.py``). It exists so
``benchmark.py`` and ``live_generate.py`` have two real, differently-sized
trained models to run speculative decoding between.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..checkpoint import save_weights
from ..functional import soft_cross_entropy, softmax
from ..model import GPT, GPTConfig
from ..optim import Adam
from ..tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent.parent.parent.parent
CORPUS_PATH = ROOT / "data" / "corpus.txt"
CHECKPOINT_DIR = ROOT / "checkpoints"

VOCAB_SIZE = 512
BLOCK_SIZE = 160


def make_batches(ids: np.ndarray, block_size: int, batch_size: int, rng: np.random.Generator):
    n = len(ids) - block_size - 1
    starts = rng.integers(0, n, size=batch_size)
    x = np.stack([ids[s : s + block_size] for s in starts])
    y = np.stack([ids[s + 1 : s + block_size + 1] for s in starts])
    return x, y


def train_one_model(cfg: GPTConfig, ids: np.ndarray, *, steps: int, lr: float, batch_size: int, seed: int, label: str):
    rng = np.random.default_rng(seed)
    model = GPT(cfg, seed=seed)
    opt = Adam(model.parameters(), lr=lr)

    t0 = time.time()
    for step in range(steps):
        x, y = make_batches(ids, BLOCK_SIZE, batch_size, rng)
        opt.zero_grad()
        loss = model.loss(x, y)
        loss.backward(np.array(1.0))
        opt.step()
        if step % max(steps // 10, 1) == 0 or step == steps - 1:
            print(f"[{label}] step {step:4d}/{steps}  loss={float(loss.data):.4f}")
    print(f"[{label}] trained in {time.time() - t0:.1f}s")
    return model


def train_draft_via_distillation(
    draft_cfg: GPTConfig, teacher: GPT, ids: np.ndarray, *, steps: int, lr: float, batch_size: int, seed: int
) -> GPT:
    rng = np.random.default_rng(seed)
    student = GPT(draft_cfg, seed=seed)
    opt = Adam(student.parameters(), lr=lr)

    t0 = time.time()
    for step in range(steps):
        x, _ = make_batches(ids, BLOCK_SIZE, batch_size, rng)
        teacher_probs = softmax(teacher.forward(x).data)  # frozen teacher, no backward needed
        opt.zero_grad()
        student_logits = student.forward(x)
        b, t, v = student_logits.shape
        loss = soft_cross_entropy(student_logits.reshape(b * t, v), teacher_probs.reshape(b * t, v))
        loss.backward(np.array(1.0))
        opt.step()
        if step % max(steps // 10, 1) == 0 or step == steps - 1:
            print(f"[draft distill] step {step:4d}/{steps}  loss={float(loss.data):.4f}")
    print(f"[draft distill] trained in {time.time() - t0:.1f}s")
    return student


def main() -> None:
    text = CORPUS_PATH.read_text(encoding="utf-8")

    print(f"training BPE tokenizer on {len(text)} characters, target vocab {VOCAB_SIZE}...")
    tok = BPETokenizer()
    tok.train(text, vocab_size=VOCAB_SIZE)
    ids = np.array(tok.encode(text))
    print(f"corpus encodes to {len(ids)} tokens (vs {len(text.encode('utf-8'))} raw bytes)")

    # Every training batch uses positions 0..BLOCK_SIZE-1 (see model.forward),
    # so those are the only position embeddings that ever get trained --
    # max_seq_len has to equal BLOCK_SIZE, not exceed it, or generation runs
    # into position embeddings that were never updated from their random init.
    draft_cfg = GPTConfig(vocab_size=tok.vocab_size, d_model=32, n_heads=2, n_layers=1, max_seq_len=BLOCK_SIZE)
    main_cfg = GPTConfig(vocab_size=tok.vocab_size, d_model=64, n_heads=2, n_layers=2, max_seq_len=BLOCK_SIZE)

    # Deliberately stopped well short of the tiny corpus being memorized
    # (loss plateaus around 1.0-1.5, not near 0): an overfit teacher's
    # predictions are so sharply pinned to specific memorized windows that
    # a much smaller student can't track them once generation drifts even
    # slightly from the literal training text, which collapses the draft
    # acceptance rate speculative decoding depends on.
    main_model = train_one_model(main_cfg, ids, steps=250, lr=2e-3, batch_size=16, seed=1, label="main ")
    draft_model = train_draft_via_distillation(
        draft_cfg, main_model, ids, steps=600, lr=3e-3, batch_size=16, seed=0
    )

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    tok.save(CHECKPOINT_DIR / "tokenizer.pkl")
    save_weights(draft_model.export_weights(), CHECKPOINT_DIR / "draft.pkl")
    save_weights(main_model.export_weights(), CHECKPOINT_DIR / "main.pkl")
    print(f"saved checkpoints to {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()

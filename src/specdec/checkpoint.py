"""Save/load a model's exported weight dict (see ``GPT.export_weights``)."""

from __future__ import annotations

import pickle
from pathlib import Path


def save_weights(weights: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(weights, f)


def load_weights(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)

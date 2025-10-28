"""TensorFlow Conformer fine-tuning stub.

This module documents a potential workflow for adapting a Conformer-based ASR
model with TensorFlowASR while also exposing a lightweight inference function
that mirrors the Whisper wrapper. It intentionally keeps runtime lightweight so
CI can execute without GPU access.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import importlib
import numpy as np

tf_spec = importlib.util.find_spec("tensorflow")
tf = importlib.import_module("tensorflow") if tf_spec is not None else None


@dataclass
class StubConfig:
    learning_rate: float = 1e-4
    batch_size: int = 4
    num_epochs: int = 20
    model_name: str = "conformer_small"
    dataset_path: str = "data/processed/manifest.jsonl"


def load_config() -> StubConfig:
    return StubConfig()


def fine_tune(config: Optional[StubConfig] = None) -> None:
    config = config or load_config()
    print("[tf_conformer_stub] This is a placeholder for actual fine-tuning.")
    print(f"Would train {config.model_name} on {config.dataset_path} "
          f"for {config.num_epochs} epochs with lr={config.learning_rate}.")
    print("Integrate TensorFlowASR scripts here when real data and GPU are available.")


def infer(
    wav_path: str | Path,
    language_hint: Optional[str] = None,
    hotwords: Optional[Iterable[str]] = None,
) -> str:
    """Simple deterministic stub returning pseudo text."""

    wav_path = Path(wav_path)
    seed = abs(hash(str(wav_path))) % (2**32)
    rng = np.random.default_rng(seed)
    tokens = ["la", "na", "ga", "ma", "ha"]
    length = 6 if not language_hint else max(4, len(language_hint))
    output = " ".join(rng.choice(tokens, size=length))
    if hotwords:
        output += " " + " ".join(hotwords)
    return output


__all__ = ["StubConfig", "load_config", "fine_tune", "infer"]

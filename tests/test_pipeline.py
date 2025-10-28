from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import soundfile as sf

from scripts import prepare_data


def test_pipeline_stub(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_TEST_MODE", "1")

    processed_dir = Path("data/processed")
    raw_dir = Path("data/raw")
    for directory in (processed_dir, raw_dir):
        if directory.exists():
            for file in directory.glob("*.wav"):
                file.unlink()

    # Prepare synthetic dataset
    prepare_data.generate_dataset(prepare_data.DIALECT_SENTENCES[:1])
    input_dir = processed_dir
    assert any(input_dir.glob("*.wav"))

    output_dir = tmp_path / "outputs"
    cmd = [
        sys.executable,
        "pipeline_infer.py",
        "--input_dir",
        str(input_dir),
        "--output_dir",
        str(output_dir),
        "--language",
        "zh",
    ]
    subprocess.run(cmd, check=True)

    transcripts = output_dir / "transcripts.csv"
    assert transcripts.exists()
    with transcripts.open("r", encoding="utf-8") as fp:
        lines = fp.readlines()
    assert len(lines) >= 2

    tts_outputs = list(output_dir.glob("*_tts.wav"))
    assert tts_outputs, "TTS output not generated"
    wav, sr = sf.read(tts_outputs[0])
    assert sr == 16_000
    assert wav.shape[0] > 0

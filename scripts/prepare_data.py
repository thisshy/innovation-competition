#!/usr/bin/env python
"""Prepare demo dialect dataset.

This script synthesizes toy waveforms that mimic dialect utterances for
end-to-end testing. Real deployments should replace this with actual
recordings and ensure corresponding licensing is in place.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Iterable, List

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
METADATA_PATH = PROCESSED_DIR / "metadata.csv"
MANIFEST_PATH = PROCESSED_DIR / "manifest.jsonl"
SAMPLE_RATE = 16_000
MAX_DURATION = 15.0

DIALECT_SENTENCES: List[str] = [
    "侬好呀，今朝吃过早饭伐？",
    "這擺哩天色好清爽，出去耍一哈。",
    "倷伲老屋里有好多故事等侬来听。",
    "恁要去圩上买菜，记得带伢儿哈。",
]


def create_sine_speech(text: str, sr: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    duration = min(3.0 + rng.random() * 4.0, MAX_DURATION)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    base_freq = 180 + rng.normal(0, 20)
    formants = [base_freq * (i + 1) for i in range(1, 4)]
    waveform = np.zeros_like(t)
    for i, freq in enumerate(formants):
        waveform += (1.0 / (i + 1)) * np.sin(2 * np.pi * freq * t)
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 2 * t + rng.random()))
    waveform *= envelope
    noise = 0.02 * rng.standard_normal(size=t.shape)
    waveform += noise
    waveform /= np.max(np.abs(waveform) + 1e-6)
    return waveform.astype(np.float32)


def trim_silence(waveform: np.ndarray, threshold: float = 0.02) -> np.ndarray:
    energy = np.abs(waveform)
    mask = energy > threshold
    if not mask.any():
        return waveform
    start = np.argmax(mask)
    end = len(mask) - np.argmax(mask[::-1])
    return waveform[start:end]


def rms_normalize(waveform: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    rms = np.sqrt(np.mean(np.square(waveform))) + 1e-6
    rms_db = 20 * np.log10(rms)
    gain_db = target_db - rms_db
    gain = np.power(10.0, gain_db / 20)
    return waveform * gain


def lowpass_filter(waveform: np.ndarray, sr: int, cutoff: float = 7_000.0) -> np.ndarray:
    nyq = 0.5 * sr
    b, a = butter(N=4, Wn=cutoff / nyq, btype="low")
    return filtfilt(b, a, waveform)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_audio(path: Path, waveform: np.ndarray, sr: int) -> None:
    sf.write(path, waveform, sr)


def generate_dataset(texts: Iterable[str]) -> None:
    ensure_dir(RAW_DIR)
    ensure_dir(PROCESSED_DIR)

    manifest_rows = []
    metadata_rows = [["utt_id", "path", "text"]]

    for idx, sentence in enumerate(tqdm(list(texts), desc="Synthesizing demo data")):
        utt_id = f"demo_{idx:03d}"
        raw_path = RAW_DIR / f"{utt_id}.wav"
        processed_path = PROCESSED_DIR / f"{utt_id}.wav"

        waveform = create_sine_speech(sentence, SAMPLE_RATE, seed=idx)
        waveform = lowpass_filter(waveform, SAMPLE_RATE)
        waveform = trim_silence(waveform)
        waveform = rms_normalize(waveform)
        save_audio(raw_path, waveform, SAMPLE_RATE)

        # Re-load for uniform resampling and ensure <= 15 s
        waveform_proc, _ = librosa.load(raw_path, sr=SAMPLE_RATE)
        if waveform_proc.shape[0] / SAMPLE_RATE > MAX_DURATION:
            waveform_proc = waveform_proc[: int(MAX_DURATION * SAMPLE_RATE)]
        waveform_proc = trim_silence(waveform_proc)
        waveform_proc = rms_normalize(waveform_proc)
        save_audio(processed_path, waveform_proc, SAMPLE_RATE)

        manifest_rows.append({
            "utt_id": utt_id,
            "path": str(processed_path.relative_to(ROOT)),
            "text": sentence,
        })
        metadata_rows.append([utt_id, str(processed_path.relative_to(ROOT)), sentence])

    with MANIFEST_PATH.open("w", encoding="utf-8") as fp:
        for row in manifest_rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    with METADATA_PATH.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(metadata_rows)

    print(f"Prepared {len(manifest_rows)} utterances at {PROCESSED_DIR}.")
    print(f"Metadata saved to {METADATA_PATH} and {MANIFEST_PATH}.")


def main() -> None:
    random.shuffle(DIALECT_SENTENCES)
    generate_dataset(DIALECT_SENTENCES)


if __name__ == "__main__":
    main()

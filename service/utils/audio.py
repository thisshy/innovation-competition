"""Audio utility helpers used across the service layer."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf

from fastapi import HTTPException

DEFAULT_SR = 16_000


def load_audio(path: str | Path, sr: int = DEFAULT_SR) -> Tuple[np.ndarray, int]:
    waveform, sample_rate = librosa.load(path, sr=sr, mono=True)
    return waveform.astype(np.float32), sample_rate


def save_audio(path: str | Path, waveform: np.ndarray, sr: int = DEFAULT_SR) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform, sr)
    return path


def wav_bytes_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def wav_file_to_base64(path: str | Path) -> str:
    with open(path, "rb") as fp:
        return wav_bytes_to_base64(fp.read())


def audio_to_bytes(waveform: np.ndarray, sr: int = DEFAULT_SR) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sr, format="WAV")
    return buffer.getvalue()


def decode_base64_audio(audio_b64: str, sr: int = DEFAULT_SR) -> Tuple[np.ndarray, int]:
    """Decode base64 audio payload into a mono waveform at the target sample rate."""

    try:
        raw = base64.b64decode(audio_b64)
    except (base64.binascii.Error, ValueError) as exc:  # type: ignore[attr-defined]
        raise HTTPException(status_code=400, detail="不支持二进制文件，请上传有效音频。") from exc

    with io.BytesIO(raw) as buffer:
        try:
            waveform, sample_rate = sf.read(buffer, dtype="float32")
        except RuntimeError as exc:  # pragma: no cover - depends on external libs
            raise HTTPException(status_code=400, detail="音频解析失败，请使用 WAV/MP3/M4A 格式。") from exc

    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    if sample_rate != sr:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=sr)
        sample_rate = sr
    return waveform.astype(np.float32), sample_rate


__all__ = [
    "DEFAULT_SR",
    "load_audio",
    "save_audio",
    "wav_bytes_to_base64",
    "wav_file_to_base64",
    "audio_to_bytes",
    "decode_base64_audio",
]

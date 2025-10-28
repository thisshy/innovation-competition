"""TensorFlowTTS inference helper with CPU/GPU fallback and stub."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

tf_spec = importlib.util.find_spec("tensorflow")
tf = importlib.import_module("tensorflow") if tf_spec is not None else None

tftts_spec = importlib.util.find_spec("TensorFlowTTS.inference")
if tftts_spec is not None:  # pragma: no branch
    tftts_infer = importlib.import_module("TensorFlowTTS.inference")
    TFAutoModel = getattr(tftts_infer, "TFAutoModel", None)
else:  # pragma: no cover
    TFAutoModel = None  # type: ignore

DEFAULT_TTS_MODEL = os.getenv("TTS_MODEL", "tensorspeech/tts-tacotron2-ljspeech-en")
DEFAULT_VOCODER = os.getenv("TTS_VOCODER", "tensorspeech/mb-melgan-ljspeech-en")


class TensorFlowTTSInfer:
    def __init__(self, model_name: str = DEFAULT_TTS_MODEL, vocoder_name: str = DEFAULT_VOCODER):
        self.model_name = model_name
        self.vocoder_name = vocoder_name
        self._is_stub = bool(os.getenv("PIPELINE_TEST_MODE"))
        self._tts = None
        self._vocoder = None
        if not self._is_stub and TFAutoModel is not None:
            try:
                self._tts = TFAutoModel.from_pretrained(model_name)
                self._vocoder = TFAutoModel.from_pretrained(vocoder_name)
            except Exception as exc:  # pragma: no cover
                print(f"[TensorFlowTTSInfer] Unable to load pretrained models: {exc}. Using stub.")
                self._is_stub = True
        else:
            if TFAutoModel is None:
                print("[TensorFlowTTSInfer] TensorFlowTTS not available, using stub mode.")
            self._is_stub = True

    def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: float = 1.0,
        out_path: str | Path = "assets/synth.wav",
    ) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if self._is_stub:
            waveform = self._stub_synthesize(text, speed=speed)
            sf.write(out_path, waveform, 16_000)
            return out_path

        # pragma: no cover - heavy branch
        if tf is None:
            raise RuntimeError("TensorFlow is not available for real inference")
        mel_outputs, mel_lengths, alignments = self._tts.inference(
            tf.constant([text]),
            speaker_ids=tf.constant([speaker_id or 0]),
            speed_ratios=tf.constant([speed]),
        )
        audios = self._vocoder.inference(mel_outputs)
        audio = audios[0].numpy()
        sf.write(out_path, audio, 22_050)
        return out_path

    # ------------------------------------------------------------------
    def _stub_synthesize(self, text: str, speed: float) -> np.ndarray:
        duration = max(1.5, min(len(text) / 8.0, 6.0))
        sr = 16_000
        t = np.linspace(0, duration / max(speed, 0.1), int(sr * duration / max(speed, 0.1)), endpoint=False)
        freqs = [220, 330, 440]
        waveform = sum(np.sin(2 * np.pi * f * t) for f in freqs)
        waveform /= len(freqs)
        envelope = np.linspace(0.1, 1.0, waveform.size)
        waveform *= envelope
        waveform += 0.01 * np.random.standard_normal(size=waveform.shape)
        waveform /= np.max(np.abs(waveform) + 1e-6)
        return waveform.astype(np.float32)


def synthesize(text: str, speaker_id: Optional[int] = None, speed: float = 1.0, out_path: str | Path = "assets/synth.wav") -> Path:
    return TensorFlowTTSInfer().synthesize(text=text, speaker_id=speaker_id, speed=speed, out_path=out_path)


__all__ = ["TensorFlowTTSInfer", "synthesize"]

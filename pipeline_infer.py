#!/usr/bin/env python
"""Command line pipeline for ASR -> normalize -> TTS."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

from tqdm import tqdm

from asr.whisper_infer import WhisperASR
from nlp.normalize import normalize
from tts.tensorflowtts_infer import TensorFlowTTSInfer

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    raw = CONFIG_PATH.read_text(encoding="utf-8")
    if not raw.strip():
        return {}
    if yaml is not None:
        try:
            parsed = yaml.safe_load(raw)
            if isinstance(parsed, dict):
                return parsed
        except yaml.YAMLError:
            pass
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid configuration at {CONFIG_PATH}: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end pipeline on a directory of wav files")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing wav files")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to store outputs")
    parser.add_argument("--language", type=str, default=None, help="Language hint for ASR")
    parser.add_argument("--speed", type=float, default=1.0, help="TTS speaking rate")
    args = parser.parse_args()

    cfg = load_config()
    pipeline_cfg = cfg.get("pipeline", {})
    hotwords = pipeline_cfg.get("hotwords", [])
    service_cfg = cfg.get("service", {})
    language = args.language or service_cfg.get("default_language", "zh")

    asr = WhisperASR(model_name=pipeline_cfg.get("asr_model", "openai/whisper-small"))
    tts = TensorFlowTTSInfer(
        model_name=pipeline_cfg.get("tts_model", os.getenv("TTS_MODEL", "tensorspeech/tts-tacotron2-ljspeech-en")),
        vocoder_name=pipeline_cfg.get("tts_vocoder", os.getenv("TTS_VOCODER", "tensorspeech/mb-melgan-ljspeech-en")),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    transcripts_path = args.output_dir / "transcripts.csv"
    with transcripts_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["utt_id", "text", "normalized", "tts_path"])
        for wav_path in tqdm(sorted(args.input_dir.glob("*.wav")), desc="Pipeline inference"):
            utt_id = wav_path.stem
            asr_result = asr.transcribe(wav_path, language_hint=language, hotwords=hotwords)
            normalized = normalize(asr_result.text, hotwords=hotwords)
            synth_path = args.output_dir / f"{utt_id}_tts.wav"
            tts.synthesize(normalized, speaker_id=None, speed=args.speed, out_path=synth_path)
            try:
                synth_display = str(synth_path.relative_to(ROOT))
            except ValueError:
                synth_display = str(synth_path)
            writer.writerow([utt_id, asr_result.text, normalized, synth_display])
            print(f"Processed {utt_id}: {asr_result.text}")

    print(f"Transcripts saved to {transcripts_path}")


if __name__ == "__main__":
    main()

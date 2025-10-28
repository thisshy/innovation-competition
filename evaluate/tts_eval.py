#!/usr/bin/env python
"""TTS evaluation utilities combining objective and subjective templates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
from tqdm import tqdm

from service.utils.audio import load_audio


@dataclass
class TTSMetric:
    utt_id: str
    f0_l1: float
    energy_l1: float


@dataclass
class EvaluationReport:
    metrics: List[TTSMetric]
    overall_f0: float
    overall_energy: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "overall_f0_l1": self.overall_f0,
            "overall_energy_l1": self.overall_energy,
        }


SUBJECTIVE_TEMPLATE = """
主观评测建议：
1. 采用MOS五点评分（1-差，5-优），邀请至少3名母语者聆听合成音频。
2. 评分项目包含：音质自然度、方言准确性、情感表达、一致性。
3. 记录听感反馈与常见错误，补充至 `reports/tts_subjective.md`。
""".strip()


def compute_f0_energy(path: Path) -> Dict[str, np.ndarray]:
    y, sr = load_audio(path)
    f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
    f0 = np.nan_to_num(f0)
    energy = librosa.feature.rms(y=y)[0]
    return {"f0": f0, "energy": energy}


def align_and_compare(ref_feat: Dict[str, np.ndarray], syn_feat: Dict[str, np.ndarray]) -> Dict[str, float]:
    min_len = min(len(ref_feat["f0"]), len(syn_feat["f0"]))
    ref_f0 = ref_feat["f0"][:min_len]
    syn_f0 = syn_feat["f0"][:min_len]
    ref_energy = ref_feat["energy"][:min_len]
    syn_energy = syn_feat["energy"][:min_len]
    return {
        "f0_l1": float(np.mean(np.abs(ref_f0 - syn_f0))),
        "energy_l1": float(np.mean(np.abs(ref_energy - syn_energy))),
    }


def evaluate_tts(ref_dir: Path, syn_dir: Path) -> EvaluationReport:
    metrics: List[TTSMetric] = []
    for ref_path in tqdm(sorted(ref_dir.glob("*.wav")), desc="Evaluating TTS"):
        utt_id = ref_path.stem
        syn_path = syn_dir / f"{utt_id}.wav"
        if not syn_path.exists():
            continue
        ref_feat = compute_f0_energy(ref_path)
        syn_feat = compute_f0_energy(syn_path)
        diff = align_and_compare(ref_feat, syn_feat)
        metrics.append(TTSMetric(utt_id=utt_id, **diff))
    overall_f0 = float(np.mean([m.f0_l1 for m in metrics]) if metrics else 0.0)
    overall_energy = float(np.mean([m.energy_l1 for m in metrics]) if metrics else 0.0)
    return EvaluationReport(metrics=metrics, overall_f0=overall_f0, overall_energy=overall_energy)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TTS outputs against references")
    parser.add_argument("--ref_dir", type=Path, required=True, help="Directory with reference WAV files")
    parser.add_argument("--syn_dir", type=Path, required=True, help="Directory with synthesized WAV files")
    args = parser.parse_args()

    report = evaluate_tts(args.ref_dir, args.syn_dir)
    print("Objective metrics:")
    for key, value in report.as_dict().items():
        print(f"  {key}: {value:.4f}")
    print("\n" + SUBJECTIVE_TEMPLATE)


if __name__ == "__main__":
    main()

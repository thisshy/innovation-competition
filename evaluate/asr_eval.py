#!/usr/bin/env python
"""ASR evaluation script computing CER/WER using jiwer."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

from jiwer import cer, wer


def load_metadata(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            data[row["utt_id"]] = row["text"]
    return data


def load_transcripts(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            data[row["utt_id"]] = row["text"]
    return data


def evaluate(reference: Dict[str, str], hypothesis: Dict[str, str]) -> Dict[str, float]:
    refs = []
    hyps = []
    for utt_id, ref_text in reference.items():
        hyp_text = hypothesis.get(utt_id, "")
        refs.append(ref_text)
        hyps.append(hyp_text)
    return {
        "WER": wer(refs, hyps),
        "CER": cer(refs, hyps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CER/WER for ASR outputs")
    parser.add_argument("--ref", type=Path, required=True, help="Reference metadata CSV")
    parser.add_argument("--hyp", type=Path, required=True, help="Hypothesis transcript CSV")
    args = parser.parse_args()

    ref = load_metadata(args.ref)
    hyp = load_transcripts(args.hyp)
    metrics = evaluate(ref, hyp)
    print("ASR Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()

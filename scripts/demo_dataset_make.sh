#!/usr/bin/env bash
set -euo pipefail

python scripts/prepare_data.py

python - <<'PY'
from pathlib import Path
from tqdm import tqdm

processed = Path('data/processed')
files = sorted(processed.glob('*.wav'))
for wav in tqdm(files, desc='Listing processed clips'):
    print(wav)
PY

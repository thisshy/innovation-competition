# Assets Directory

Binary demo artifacts are generated on demand to keep pull requests text-only friendly.

## Generate Sample Audio

Run the data preparation helper to synthesize toy dialect utterances at 16 kHz:

```bash
python scripts/prepare_data.py
```

The processed waveforms will be written to `data/processed/`. You can copy any of
those files into this folder when demonstrating the Streamlit front end.

## Cache and Outputs

The application writes temporary cache files and synthesized speech outputs here
at runtime. These are ignored by git via `.gitignore` patterns so they never
block code reviews.

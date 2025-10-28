.PHONY: setup run_api run_web demo eval lint test docker-build docker-up

setup:
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

run_api:
uvicorn service.main:app --host 0.0.0.0 --port 8000

run_web:
streamlit run web/app.py

demo:
python scripts/prepare_data.py && python pipeline_infer.py --input_dir data/processed --output_dir assets/demo_outputs

eval:
python evaluate/asr_eval.py --hyp assets/demo_outputs/transcripts.csv --ref data/processed/metadata.csv

lint:
python -m compileall .

test:
pytest -q

docker-build:
docker build -t dialect-platform -f docker/Dockerfile .

docker-up:
docker-compose up --build

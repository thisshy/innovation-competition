FROM python:3.10-slim AS base

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8000
EXPOSE 8501

CMD ["bash", "-c", "uvicorn service.main:app --host 0.0.0.0 --port 8000 & streamlit run web/app.py --server.port 8501 --server.address 0.0.0.0"]

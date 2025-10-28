"""FastAPI service exposing ASR, TTS and pipeline routes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    import yaml
except ImportError:  # pragma: no cover - should be available via requirements
    yaml = None  # type: ignore

from asr.whisper_infer import WhisperASR
from nlp.normalize import normalize
from service.schemas import (
    ASRRequest,
    ASRResponse,
    PipelineRequest,
    PipelineResponse,
    TTSRequest,
    TTSResponse,
)
from service.utils.audio import decode_base64_audio, save_audio
from tts.tensorflowtts_infer import TensorFlowTTSInfer

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
CACHE_DIR = ROOT / "assets" / "cache"
UPLOAD_DIR = CACHE_DIR / "uploads"
SYNTH_DIR = CACHE_DIR / "tts"


def load_config(path: Path = CONFIG_PATH) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        raw = fp.read()
    if not raw.strip():
        return {}
    if yaml is not None:
        try:
            data = yaml.safe_load(raw)
            if isinstance(data, dict):
                return data
        except yaml.YAMLError:
            pass
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid configuration at {path}: {exc}")


CONFIG = load_config()
PIPELINE_CFG = CONFIG.get("pipeline", {})
SERVICE_CFG = CONFIG.get("service", {})
DEVICE = os.getenv("DEVICE", PIPELINE_CFG.get("device", "cpu"))
ASR_MODEL = PIPELINE_CFG.get("asr_model", "openai/whisper-small")
TTS_MODEL = PIPELINE_CFG.get("tts_model")
TTS_VOCODER = PIPELINE_CFG.get("tts_vocoder")
DEFAULT_LANGUAGE = SERVICE_CFG.get("default_language", "zh")
DEFAULT_HOTWORDS = PIPELINE_CFG.get("hotwords", [])

CACHE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SYNTH_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AI 方言保护与传播平台 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ASR_ENGINE = WhisperASR(model_name=ASR_MODEL, device=None if DEVICE == "cpu" else DEVICE)
TTS_ENGINE = TensorFlowTTSInfer(
    model_name=TTS_MODEL or os.getenv("TTS_MODEL", "tensorspeech/tts-tacotron2-ljspeech-en"),
    vocoder_name=TTS_VOCODER or os.getenv("TTS_VOCODER", "tensorspeech/mb-melgan-ljspeech-en"),
)


def _persist_base64_audio(audio_b64: str, prefix: str = "upload") -> Path:
    waveform, _ = decode_base64_audio(audio_b64)
    unique = uuid.uuid4().hex[:10]
    path = UPLOAD_DIR / f"{prefix}_{unique}.wav"
    save_audio(path, waveform)
    return path


def _resolve_audio_path(wav_path: Optional[str], audio_base64: Optional[str]) -> Path:
    if audio_base64:
        return _persist_base64_audio(audio_base64)
    if wav_path:
        resolved = Path(wav_path)
        if not resolved.is_absolute():
            resolved = (ROOT / wav_path).resolve()
        if not resolved.exists():
            raise HTTPException(status_code=404, detail=f"Audio path not found: {wav_path}")
        return resolved
    raise HTTPException(status_code=400, detail="Either wav_path or audio_base64 must be provided")


@app.post("/asr", response_model=ASRResponse)
async def asr_endpoint(request: ASRRequest) -> ASRResponse:
    audio_path = _resolve_audio_path(request.wav_path, request.audio_base64)
    hotwords = list(request.hotwords or []) + list(DEFAULT_HOTWORDS)
    result = ASR_ENGINE.transcribe(
        wav_path=audio_path,
        language_hint=request.language_hint or DEFAULT_LANGUAGE,
        hotwords=hotwords or None,
    )
    return ASRResponse(text=result.text, segments=list(result.segments or []))


@app.post("/tts", response_model=TTSResponse)
async def tts_endpoint(request: TTSRequest) -> TTSResponse:
    output_path = request.output_path or SYNTH_DIR / "tts_output.wav"
    audio_path = TTS_ENGINE.synthesize(
        text=request.text,
        speaker_id=request.speaker_id,
        speed=request.speed,
        out_path=output_path,
    )
    return TTSResponse(audio_path=str(audio_path.relative_to(ROOT)))


@app.post("/pipeline", response_model=PipelineResponse)
async def pipeline_endpoint(request: PipelineRequest) -> PipelineResponse:
    audio_path = _resolve_audio_path(request.wav_path, request.audio_base64)
    hotwords = list(request.hotwords or []) + list(DEFAULT_HOTWORDS)

    asr_result = ASR_ENGINE.transcribe(
        wav_path=audio_path,
        language_hint=request.language_hint or DEFAULT_LANGUAGE,
        hotwords=hotwords or None,
    )
    normalized = normalize(asr_result.text, hotwords=hotwords)
    translation = normalized if request.enable_translation else None

    synth_path = SYNTH_DIR / f"tts_{audio_path.stem}.wav"
    synth_path = TTS_ENGINE.synthesize(
        normalized,
        speaker_id=None,
        speed=request.tts_speed,
        out_path=synth_path,
    )

    intermediate = {
        "upload": str(audio_path.relative_to(ROOT)) if audio_path.is_relative_to(ROOT) else str(audio_path),
        "tts": str(synth_path.relative_to(ROOT)),
    }

    return PipelineResponse(
        transcript=asr_result.text,
        normalized=normalized,
        translation=translation,
        tts_audio_path=str(synth_path.relative_to(ROOT)),
        intermediate_paths=intermediate,
    )


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "device": DEVICE}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

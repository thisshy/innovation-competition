"""Pydantic schemas for FastAPI service."""

from __future__ import annotations

from typing import Iterable, Optional

from pydantic import BaseModel, Field


class ASRRequest(BaseModel):
    wav_path: Optional[str] = Field(None, description="Path to wav file on server or uploaded temp file")
    audio_base64: Optional[str] = Field(None, description="Base64 encoded audio content")
    language_hint: Optional[str] = Field(None, description="Language code for Whisper")
    hotwords: Optional[Iterable[str]] = Field(None, description="Hotwords to bias decoding")


class ASRResponse(BaseModel):
    text: str
    segments: Optional[list] = None


class TTSRequest(BaseModel):
    text: str
    speaker_id: Optional[int] = Field(None, description="Speaker identifier for multi-speaker models")
    speed: float = Field(1.0, description="Speaking rate multiplier")
    output_path: Optional[str] = Field(None, description="Optional path to save the waveform")


class TTSResponse(BaseModel):
    audio_path: str


class PipelineRequest(BaseModel):
    wav_path: Optional[str] = None
    audio_base64: Optional[str] = None
    enable_translation: bool = Field(False, description="Whether to produce Mandarin translation")
    hotwords: Optional[Iterable[str]] = None
    tts_speed: float = Field(1.0, description="TTS speaking rate")
    language_hint: Optional[str] = Field(None, description="Language hint for ASR")


class PipelineResponse(BaseModel):
    transcript: str
    normalized: str
    translation: Optional[str]
    tts_audio_path: Optional[str]
    intermediate_paths: dict


__all__ = [
    "ASRRequest",
    "ASRResponse",
    "TTSRequest",
    "TTSResponse",
    "PipelineRequest",
    "PipelineResponse",
]

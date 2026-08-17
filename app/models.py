from enum import Enum
from typing import Literal, Optional, List
from pydantic import BaseModel, Field


class Tone(str, Enum):
    NEUTRAL = "neutral"
    SAD = "sad"
    HAPPY = "happy"
    ANGRY = "angry"
    EXCITED = "excited"
    CALM = "calm"
    NERVOUS = "nervous"
    SARCASTIC = "sarcastic"


class Segment(BaseModel):
    text: str
    tone: Tone
    intensity: int = Field(default=2, ge=1, le=3)


class TTSChunkInfo(BaseModel):
    chunk_index: int
    raw_text: str
    clean_text: str  # The exact clean Thai text passed to TTS
    tone: str
    prosody_rate: str
    prosody_pitch: str
    voice: str
    char_length: Optional[int] = None


class AnnotateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    guidance: Optional[str] = Field(default=None, description="Optional custom emotion/tone guidance")


class AnnotateResponse(BaseModel):
    original: str
    segments: list[Segment]
    clean_tts_text: Optional[str] = None
    tts_chunks: Optional[List[TTSChunkInfo]] = None
    model_used: str
    fallback: bool  # True = fallback occurred
    fallback_reason: Optional[str] = None
    latency_ms: Optional[float] = None
    clauses_count: Optional[int] = None
    timestamp: Optional[str] = None


class RenderRequest(BaseModel):
    segments: list[Segment]
    engine: Literal["rvc", "cosyvoice", "fishspeech", "gemini", "elevenlabs", "voxcpm", "siangtts"]


class RenderResponse(BaseModel):
    text: str  # text ready for TTS / instruction prompt
    prompt: Optional[str] = None  # for engines using separate field (Gemini/RVC instruction summary)


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    guidance: Optional[str] = Field(default=None, description="Optional custom emotion/tone guidance")
    engine: Literal["rvc", "cosyvoice", "fishspeech", "gemini", "elevenlabs", "voxcpm", "siangtts"] = "rvc"


class SpeakResponse(BaseModel):
    engine: Literal["rvc", "cosyvoice", "fishspeech", "gemini", "elevenlabs", "voxcpm", "siangtts"]
    text: str
    clean_tts_text: Optional[str] = None
    tts_chunks: Optional[List[TTSChunkInfo]] = None
    prompt: Optional[str] = None
    segments: list[Segment]
    model_used: str
    fallback: bool
    fallback_reason: Optional[str] = None
    latency_ms: Optional[float] = None
    clauses_count: Optional[int] = None
    timestamp: Optional[str] = None


class SpeakerInfo(BaseModel):
    id: str
    name: str
    filename: str
    cached: bool
    model_type: str = "rvc"  # "rvc", "rvc_model", or "audio_ref"
    index_file: Optional[str] = None
    default_pitch: int = 0


class SpeakerListResponse(BaseModel):
    speakers: List[SpeakerInfo]


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    speaker_id: Optional[str] = None
    guidance: Optional[str] = None
    engine: Literal["rvc", "cosyvoice", "fishspeech", "gemini", "voxcpm", "siangtts", "elevenlabs"] = "rvc"
    pitch_shift: int = Field(default=0, ge=-24, le=24, description="RVC Pitch shift in semitones (-12 to +12)")
    index_rate: float = Field(default=0.75, ge=0.0, le=1.0, description="RVC feature retrieval strength")
    f0_method: Literal["rmvpe", "harvest", "pm", "crepe"] = "rmvpe"
    cfg_value: float = Field(default=2.5, ge=1.0, le=10.0)
    inference_timesteps: int = Field(default=10, ge=4, le=50)
    auto_annotate: bool = True


class LLMClauseItem(BaseModel):
    i: int
    text: str


class LLMClauseLabel(BaseModel):
    i: int
    tone: Tone
    intensity: int = Field(default=2, ge=1, le=3)


class LLMAnnotationResult(BaseModel):
    labels: list[LLMClauseLabel]

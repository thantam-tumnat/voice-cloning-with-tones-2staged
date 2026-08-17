from app.renderers.base import BaseRenderer
from app.renderers.gemini import GeminiRenderer
from app.renderers.elevenlabs import ElevenLabsRenderer
from app.renderers.voxcpm import VoxCPMRenderer
from app.renderers.rvc import RVCRenderer
from app.renderers.cosyvoice import CosyVoiceRenderer
from app.renderers.fishspeech import FishSpeechRenderer


def get_renderer(engine: str) -> BaseRenderer:
    eng = engine.lower().strip()
    if eng in ("rvc",):
        return RVCRenderer()
    elif eng in ("cosyvoice", "cosyvoice2"):
        return CosyVoiceRenderer()
    elif eng in ("fishspeech", "fish_speech"):
        return FishSpeechRenderer()
    elif eng == "gemini":
        return GeminiRenderer()
    elif eng == "elevenlabs":
        return ElevenLabsRenderer()
    elif eng in ("voxcpm", "siangtts"):
        return VoxCPMRenderer()
    else:
        raise ValueError(f"Unknown engine: {engine}. Choose from 'rvc', 'cosyvoice', 'fishspeech', 'gemini', 'elevenlabs', 'voxcpm', 'siangtts'")


__all__ = [
    "BaseRenderer",
    "GeminiRenderer",
    "ElevenLabsRenderer",
    "VoxCPMRenderer",
    "RVCRenderer",
    "CosyVoiceRenderer",
    "FishSpeechRenderer",
    "get_renderer",
]

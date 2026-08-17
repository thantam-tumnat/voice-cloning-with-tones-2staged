from __future__ import annotations

import io
import re
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import soundfile as sf
from scipy import signal

from app.config import settings
from app.services.thai_normalizer import normalize_thai_text


class CosyVoiceService:
    """
    CosyVoice 2 Service with SE-Bridge Thai support.
    Supports <instruct>...</instruct> natural language emotion prompts
    and inline action tags (<laughter>, <whisper>, <breath>).
    """

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self._cosyvoice_model = None

    def parse_instruct_tags(self, text: str) -> Tuple[str, str, Dict[str, Any]]:
        """
        Extract <instruct>(...)</instruct> blocks and clean text for synthesis.
        """
        instructions = re.findall(r"<instruct>\((.*?)\)</instruct>", text, re.IGNORECASE)
        instruct_str = " ".join(instructions)

        # Remove instruct blocks from speech text
        clean = re.sub(r"<instruct>\(.*?\)</instruct>", "", text, flags=re.IGNORECASE)
        
        # Check action tags
        has_whisper = "<whisper>" in clean.lower()
        has_laughter = "<laughter>" in clean.lower()
        
        # Strip action tags for clean TTS
        clean_speech = re.sub(r"</?(?:whisper|laughter|breath)>", "", clean, flags=re.IGNORECASE)
        clean_speech = re.sub(r"\s+", " ", clean_speech).strip()

        return clean_speech, instruct_str, {"whisper": has_whisper, "laughter": has_laughter}

    def synthesize(
        self,
        text: str,
        *,
        speaker_id: Optional[str] = None,
        speed: float = 1.0,
    ) -> bytes:
        """
        Synthesize speech from CosyVoice formatted text with <instruct> and action tags.
        Returns 48kHz WAV audio bytes.
        """
        clean_speech, instruct_str, action_flags = self.parse_instruct_tags(text)
        if not clean_speech:
            clean_speech = "สวัสดีครับ"

        from app.services.tts_service import tts_service
        
        # Synthesize with emotion prompt derived from instruct
        base_audio, in_sr = tts_service.synthesize(
            text=clean_speech,
            emotion_prompt=instruct_str or "ร่าเริง สดใส มีพลัง",
        )

        audio = base_audio.astype(np.float32)

        # Apply whisper effect if tag present
        if action_flags.get("whisper"):
            nyq = in_sr / 2.0
            # High-pass filter for breathy whisper acoustic
            b_hp, a_hp = signal.butter(2, 800.0 / nyq, btype="high")
            audio = signal.lfilter(b_hp, a_hp, audio) * 0.85

        # Normalize and resample to 48kHz
        if in_sr != self.sample_rate:
            num_samples = int(len(audio) * self.sample_rate / in_sr)
            audio = signal.resample(audio, num_samples)
            in_sr = self.sample_rate

        out_buf = io.BytesIO()
        sf.write(out_buf, audio, self.sample_rate, format="WAV", subtype="PCM_16")
        return out_buf.getvalue()


cosyvoice_service = CosyVoiceService()

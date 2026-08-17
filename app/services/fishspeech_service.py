from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import soundfile as sf
from scipy import signal

from app.config import settings
from app.services.thai_normalizer import normalize_thai_text


class FishSpeechService:
    """
    Fish Speech 1.5 Service.
    Supports [Tone: ...] prompt-conditioned emotional speech generation
    and native multilingual Thai synthesis.
    """

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self._fish_model = None

    def parse_tone_tags(self, text: str) -> Tuple[str, str]:
        """
        Extract [Tone: ...] description blocks and clean text for synthesis.
        """
        tones = re.findall(r"\[Tone:\s*(.*?)\]", text, re.IGNORECASE)
        tone_str = " ".join(tones)

        # Remove tone blocks from speech text
        clean = re.sub(r"\[Tone:\s*.*?\]", "", text, flags=re.IGNORECASE)
        clean = re.sub(r"\s+", " ", clean).strip()

        return clean, tone_str

    def synthesize(
        self,
        text: str,
        *,
        reference_audio: Optional[bytes] = None,
        speaker_id: Optional[str] = None,
    ) -> bytes:
        """
        Synthesize speech from Fish Speech prompt-conditioned format.
        Returns 48kHz WAV audio bytes.
        """
        clean_speech, tone_desc = self.parse_tone_tags(text)
        if not clean_speech:
            clean_speech = "สวัสดีครับ"

        from app.services.tts_service import tts_service

        base_audio, in_sr = tts_service.synthesize(
            text=clean_speech,
            emotion_prompt=tone_desc or "Natural tone",
        )

        audio = base_audio.astype(np.float32)

        # Resample to 48kHz if needed
        if in_sr != self.sample_rate:
            num_samples = int(len(audio) * self.sample_rate / in_sr)
            audio = signal.resample(audio, num_samples)
            in_sr = self.sample_rate

        out_buf = io.BytesIO()
        sf.write(out_buf, audio, self.sample_rate, format="WAV", subtype="PCM_16")
        return out_buf.getvalue()


fishspeech_service = FishSpeechService()

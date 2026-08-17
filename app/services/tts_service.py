from __future__ import annotations

import io
import os
import re
import asyncio
import tempfile
import numpy as np
import soundfile as sf
from scipy import signal
from typing import Optional, Dict, Any, Tuple
from app.config import settings
from app.services.thai_normalizer import normalize_thai_text


class EmotionTTSService:
    """
    Emotion & Style Instruction TTS Service.
    Synthesizes expressive Thai speech based on text with emotional prompts and style cues.
    """

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.default_voice = "th-TH-PremwadeeNeural"

    def clean_text_for_speech(self, text: str) -> str:
        """Strip inline tag brackets like [calm], (Sad...), [happily] before speech generation."""
        t = re.sub(r"\[[a-zA-Z\s]+\]", "", text)
        t = re.sub(r"\([a-zA-Z\s,.-ก-๙]+\)", "", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def get_prosody_params(self, emotion_prompt: Optional[str] = None, text: str = "") -> Tuple[str, str, str]:
        """
        Map emotional instruction or detected tone to Edge-TTS prosody parameters.
        Returns (rate, pitch, voice_name).
        """
        combined = f"{emotion_prompt or ''} {text}".lower()
        
        voice = self.default_voice
        rate = "+0%"
        pitch = "+0Hz"

        if any(w in combined for w in ["calm", "สงบ", "นุ่มนวล", "ผ่อนคลาย"]):
            rate = "-12%"
            pitch = "-4Hz"
        elif any(w in combined for w in ["sad", "เศร้า", "เสียใจ", "ตัดพ้อ"]):
            rate = "-15%"
            pitch = "-8Hz"
        elif any(w in combined for w in ["happy", "ร่าเริง", "ดีใจ", "ยิ้ม", "happily"]):
            rate = "+10%"
            pitch = "+8Hz"
        elif any(w in combined for w in ["angry", "โกรธ", "ดุดัน", "เสียงแข็ง"]):
            rate = "+15%"
            pitch = "+10Hz"
        elif any(w in combined for w in ["excited", "ตื่นเต้น", "กระตือรือร้น"]):
            rate = "+20%"
            pitch = "+12Hz"
        elif any(w in combined for w in ["sarcastic", "ประชด", "แดกดัน"]):
            rate = "-5%"
            pitch = "+6Hz"
        elif any(w in combined for w in ["nervous", "ประหม่า", "ลังเล"]):
            rate = "+8%"
            pitch = "+5Hz"

        return rate, pitch, voice

    def build_tts_chunks(self, segments: list) -> Tuple[list, str]:
        """
        Build detailed TTS chunk debug info for each segment.
        Returns (chunks_list, full_clean_text).
        """
        chunks = []
        clean_parts = []
        for idx, seg in enumerate(segments):
            raw = getattr(seg, "text", str(seg))
            tone_val = getattr(seg, "tone", "neutral")
            if hasattr(tone_val, "value"):
                tone_val = tone_val.value
            clean = self.clean_text_for_speech(raw)
            clean = normalize_thai_text(clean)
            rate, pitch, voice = self.get_prosody_params(str(tone_val), clean)
            
            chunk_info = {
                "chunk_index": idx + 1,
                "raw_text": raw,
                "clean_text": clean,
                "tone": str(tone_val),
                "prosody_rate": rate,
                "prosody_pitch": pitch,
                "voice": voice,
                "char_length": len(clean),
            }
            chunks.append(chunk_info)
            if clean:
                clean_parts.append(clean)
        
        full_clean = " ".join(clean_parts)
        return chunks, full_clean

    def synthesize(
        self,
        text: str,
        *,
        emotion_prompt: Optional[str] = None,
        voice_style: Optional[str] = None,
        cfg_value: float = 2.5,
        inference_timesteps: int = 10,
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize real natural Thai speech with emotional prosody.
        Returns (audio_samples: np.ndarray, sample_rate: int).
        """
        clean_text = self.clean_text_for_speech(text)
        clean_text = normalize_thai_text(clean_text)
        if not clean_text:
            clean_text = "สวัสดีครับ"

        # 1. Try Edge-TTS (Neural High-Quality Thai Speech)
        try:
            rate, pitch, voice = self.get_prosody_params(emotion_prompt, text)
            import edge_tts

            async def _run_edge():
                communicate = edge_tts.Communicate(clean_text, voice, rate=rate, pitch=pitch)
                audio_bytes = bytearray()
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_bytes.extend(chunk["data"])
                return bytes(audio_bytes)

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        audio_data = pool.submit(asyncio.run, _run_edge()).result()
                else:
                    audio_data = loop.run_until_complete(_run_edge())
            except Exception:
                audio_data = asyncio.run(_run_edge())

            if audio_data and len(audio_data) > 200:
                with io.BytesIO(audio_data) as buf:
                    data, sr = sf.read(buf)
                    if data.ndim > 1:
                        data = np.mean(data, axis=1)
                    if sr != self.sample_rate:
                        num_samples = int(len(data) * self.sample_rate / sr)
                        data = signal.resample(data, num_samples)
                        sr = self.sample_rate
                    return data.astype(np.float32), sr
        except Exception as e:
            print(f"[TTS] Edge-TTS error: {e}. Trying fallback TTS...")

        # 2. Try gTTS (Google Translate TTS) Fallback
        try:
            from gtts import gTTS
            tts = gTTS(text=clean_text, lang="th", slow=False)
            with io.BytesIO() as buf:
                tts.write_to_fp(buf)
                buf.seek(0)
                data, sr = sf.read(buf)
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                if sr != self.sample_rate:
                    num_samples = int(len(data) * self.sample_rate / sr)
                    data = signal.resample(data, num_samples)
                    sr = self.sample_rate
                return data.astype(np.float32), sr
        except Exception as e:
            print(f"[TTS] gTTS error: {e}. Trying local fallback...")

        # 3. Fallback waveform synthesis if completely offline
        return self._generate_fallback_audio(clean_text, emotion_prompt)

    def _generate_fallback_audio(self, text: str, emotion_prompt: Optional[str] = None) -> Tuple[np.ndarray, int]:
        sr = self.sample_rate
        duration_sec = max(1.2, min(10.0, len(text) * 0.1))
        t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
        audio = (0.3 * np.sin(2 * np.pi * 220 * t) * np.exp(-t / 2.5)).astype(np.float32)
        return audio, sr


tts_service = EmotionTTSService()

import io
import soundfile as sf
import numpy as np
import pytest

from app.services.tts_service import tts_service
from app.services.rvc_service import rvc_service
from app.services.speaker_manager import speaker_manager


def test_tts_service_emotional_synthesis():
    audio, sr = tts_service.synthesize("สวัสดีครับ ยินดีต้อนรับสู่สตูดิโอ", emotion_prompt="ร่าเริง สดใส")
    assert isinstance(audio, np.ndarray)
    assert sr == 48000
    assert len(audio) > 0


def test_rvc_voice_conversion():
    # Generate source audio
    src_audio, sr = tts_service.synthesize("ข้อความทดสอบ RVC", emotion_prompt="สงบ นุ่มนวล")
    
    # Convert with pitch shift + index
    converted, out_sr = rvc_service.convert_voice(
        audio_data=src_audio,
        sr=sr,
        speaker_id="anime_girl",
        pitch_shift=12,
        index_rate=0.8,
        f0_method="rmvpe",
    )
    assert isinstance(converted, np.ndarray)
    assert len(converted) > 0
    assert out_sr == 48000


def test_speaker_manager_lifecycle():
    speakers = speaker_manager.list_speakers()
    assert len(speakers) >= 1
    
    # Register temporary profile
    dummy_wav = io.BytesIO()
    sf.write(dummy_wav, np.zeros(16000, dtype=np.float32), 16000, format="WAV")
    
    registered = speaker_manager.register_speaker(
        speaker_id="test_speaker_tmp",
        file_bytes=dummy_wav.getvalue(),
        filename="test_speaker.wav"
    )
    assert registered["id"] == "test_speaker_tmp"
    
    # Delete
    deleted = speaker_manager.delete_speaker("test_speaker_tmp")
    assert deleted is True

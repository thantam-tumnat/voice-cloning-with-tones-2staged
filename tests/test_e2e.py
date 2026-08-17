import io
import soundfile as sf
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "thai-tts-tone-rvc"
    assert "speakers_count" in data


def test_speakers_endpoint():
    response = client.get("/speakers")
    assert response.status_code == 200
    data = response.json()
    assert "speakers" in data
    assert isinstance(data["speakers"], list)


def test_render_endpoint():
    payload = {
        "segments": [
            {"text": "สวัสดีครับ", "tone": "happy", "intensity": 2}
        ],
        "engine": "rvc"
    }
    response = client.post("/render", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "prompt" in data


def test_synthesize_endpoint():
    payload = {
        "text": "[calm] หายใจเข้าลึกๆ ผ่อนคลาย แล้วค่อยๆ ปล่อยวางทุกอย่างลงนะ",
        "speaker_id": "anime_girl",
        "engine": "rvc",
        "pitch_shift": 12,
        "index_rate": 0.75,
        "f0_method": "rmvpe",
        "auto_annotate": True
    }
    response = client.post("/synthesize", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert len(response.content) > 100


def test_synthesize_with_upload_endpoint():
    dummy_wav = io.BytesIO()
    sf.write(dummy_wav, np.random.randn(8000).astype(np.float32) * 0.1, 16000, format="WAV")
    dummy_wav.seek(0)

    response = client.post(
        "/synthesize/upload",
        data={
            "text": "ทดสอบการสังเคราะห์และแปลงเสียงด้วยไฟล์อัพโหลด",
            "pitch_shift": "0",
            "index_rate": "0.75",
            "f0_method": "rmvpe",
            "auto_annotate": "true",
        },
        files={"file": ("sample.wav", dummy_wav, "audio/wav")}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert len(response.content) > 100


def test_cosyvoice_synthesize():
    payload = {
        "text": "<instruct>(พูดด้วยน้ำเสียงตื่นเต้น ดีใจ)</instruct> ยินดีด้วยนะ! ในที่สุดก็ทำสำเร็จแล้ว",
        "engine": "cosyvoice",
        "auto_annotate": False
    }
    response = client.post("/synthesize", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert len(response.content) > 100


def test_fishspeech_synthesize():
    payload = {
        "text": "[Tone: Joyful, cheerful] ยินดีด้วยนะ! ในที่สุดก็ทำสำเร็จแล้ว",
        "engine": "fishspeech",
        "auto_annotate": False
    }
    response = client.post("/synthesize", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert len(response.content) > 100


import pytest
from app.models import Segment, Tone
from app.renderers import get_renderer


def test_gemini_renderer():
    renderer = get_renderer("gemini")
    segments = [
        Segment(text="สวัสดีครับ ", tone=Tone.HAPPY, intensity=2),
        Segment(text="ยินดีต้อนรับ", tone=Tone.HAPPY, intensity=2),
    ]
    res = renderer.render(segments)
    assert res.text == "สวัสดีครับ ยินดีต้อนรับ"
    assert "ร่าเริง" in res.prompt


def test_elevenlabs_renderer():
    renderer = get_renderer("elevenlabs")
    segments = [
        Segment(text="ขอโทษนะ", tone=Tone.SAD, intensity=2),
        Segment(text="ฉันเสียใจ", tone=Tone.SAD, intensity=2),
    ]
    res = renderer.render(segments)
    assert "[sad]" in res.text


def test_voxcpm_renderer():
    renderer = get_renderer("voxcpm")
    segments = [
        Segment(text="สบายใจจัง", tone=Tone.CALM, intensity=2),
    ]
    res = renderer.render(segments)
    assert "Calm" in res.text


def test_rvc_renderer():
    renderer = get_renderer("rvc")
    segments = [
        Segment(text="ขอโทษนะ ", tone=Tone.SAD, intensity=2),
        Segment(text="ทำไมไม่ฟังเลย", tone=Tone.ANGRY, intensity=2),
    ]
    res = renderer.render(segments)
    assert "[sad]" in res.text
    assert "[angry]" in res.text
    assert "เศร้า" in res.prompt or "โกรธ" in res.prompt

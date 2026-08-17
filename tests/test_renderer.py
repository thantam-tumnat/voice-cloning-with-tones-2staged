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


def test_cosyvoice_renderer():
    renderer = get_renderer("cosyvoice")
    segments = [
        Segment(text="ยินดีด้วยนะ!", tone=Tone.EXCITED, intensity=3),
        Segment(text="แต่เหนื่อยมากเลย", tone=Tone.SAD, intensity=2),
    ]
    res = renderer.render(segments)
    assert "<instruct>" in res.text
    assert "ยินดีด้วยนะ!" in res.text
    assert "เหนื่อยมากเลย" in res.text


def test_fishspeech_renderer():
    renderer = get_renderer("fishspeech")
    segments = [
        Segment(text="ยินดีด้วยนะ!", tone=Tone.EXCITED, intensity=3),
        Segment(text="แต่เหนื่อยมากเลย", tone=Tone.SAD, intensity=2),
    ]
    res = renderer.render(segments)
    assert "[Tone:" in res.text
    assert "ยินดีด้วยนะ!" in res.text


import pytest
from app.segmenter import segment_text


def test_segment_empty_string():
    assert segment_text("") == []


def test_segment_single_clause():
    text = "สวัสดีครับ"
    result = segment_text(text)
    assert len(result) >= 1
    assert "".join(result) == text


def test_segment_multi_clauses():
    text = "ขอโทษนะ ฉันไม่ได้ตั้งใจ แต่เธอก็ไม่ฟังฉันเลย"
    result = segment_text(text)
    assert len(result) > 1
    assert "".join(result) == text


def test_segment_preserves_whitespaces_and_punctuation():
    text = "สวัสดี 123! [calm] ข้อความทดสอบ ... พิเศษ ?"
    result = segment_text(text)
    assert "".join(result) == text

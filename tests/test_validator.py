import pytest
from app.models import Tone, Segment
from app.validator import validate_and_build_segments, ValidationError


def test_validator_valid_input():
    original = "สวัสดีครับ วันนี้อากาศดีนะ"
    clauses = ["สวัสดีครับ ", "วันนี้อากาศดีนะ"]
    raw_labels = [
        {"i": 0, "tone": "happy", "intensity": 2},
        {"i": 1, "tone": "calm", "intensity": 1},
    ]
    segments = validate_and_build_segments(original, clauses, raw_labels)
    assert len(segments) == 2
    assert segments[0].tone == Tone.HAPPY
    assert segments[1].tone == Tone.CALM
    assert "".join(s.text for s in segments) == original


def test_validator_missing_index_raises():
    original = "สวัสดีครับ วันนี้อากาศดีนะ"
    clauses = ["สวัสดีครับ ", "วันนี้อากาศดีนะ"]
    raw_labels = [{"i": 0, "tone": "happy", "intensity": 2}]
    with pytest.raises(ValidationError):
        validate_and_build_segments(original, clauses, raw_labels)


def test_validator_invalid_tone_falls_back_to_neutral():
    original = "สวัสดีครับ"
    clauses = ["สวัสดีครับ"]
    raw_labels = [{"i": 0, "tone": "unknown_emotion", "intensity": 2}]
    segments = validate_and_build_segments(original, clauses, raw_labels)
    assert len(segments) == 1
    assert segments[0].tone == Tone.NEUTRAL


def test_validator_merges_same_consecutive_tones():
    original = "สวัสดีครับ วันนี้ยินดีด้วยนะ"
    clauses = ["สวัสดีครับ ", "วันนี้ยินดีด้วยนะ"]
    raw_labels = [
        {"i": 0, "tone": "happy", "intensity": 2},
        {"i": 1, "tone": "happy", "intensity": 3},
    ]
    segments = validate_and_build_segments(original, clauses, raw_labels)
    assert len(segments) == 1
    assert segments[0].tone == Tone.HAPPY
    assert segments[0].intensity == 3
    assert segments[0].text == original

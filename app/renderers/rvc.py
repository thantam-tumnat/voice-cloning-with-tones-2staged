from typing import List
from app.models import Segment, Tone, RenderResponse
from app.renderers.base import BaseRenderer

RVC_TAG_MAPPING = {
    Tone.NEUTRAL: "",
    Tone.SAD: "[sad]",
    Tone.HAPPY: "[happily]",
    Tone.ANGRY: "[angry]",
    Tone.EXCITED: "[excited]",
    Tone.CALM: "[calm]",
    Tone.NERVOUS: "[nervous]",
    Tone.SARCASTIC: "[sarcastic]",
}

RVC_EMOTION_PROMPT_MAP = {
    Tone.NEUTRAL: "น้ำเสียงปกติ เป็นธรรมชาติ",
    Tone.SAD: "น้ำเสียงเศร้า สั่นเครือ แฝงความเสียใจ",
    Tone.HAPPY: "น้ำเสียงสดใส ร่าเริง ยิ้มแย้มขณะพูด",
    Tone.ANGRY: "น้ำเสียงโกรธ ดุดัน กระแทกเสียง",
    Tone.EXCITED: "น้ำเสียงตื่นเต้น กระตือรือร้น มีพลัง",
    Tone.CALM: "น้ำเสียงสงบ นุ่มนวล ช้าๆ ผ่อนคลาย",
    Tone.NERVOUS: "น้ำเสียงประหม่า ลังเล หวาดหวั่น",
    Tone.SARCASTIC: "น้ำเสียงประชดประชัน แดกดัน กวนๆ",
}


class RVCRenderer(BaseRenderer):
    """
    Renders emotional segments into inline audio tags [tag] and instruction prompts
    for the Emotion TTS -> RVC Voice conversion pipeline.
    """
    def render(self, segments: List[Segment]) -> RenderResponse:
        if not segments:
            return RenderResponse(text="", prompt=None)

        # 1. Build tagged text with [tag] prefixes
        result_parts = []
        for seg in segments:
            tag = RVC_TAG_MAPPING.get(seg.tone, "")
            if tag:
                result_parts.append(f"{tag} {seg.text.strip()} ")
            else:
                result_parts.append(seg.text)

        tagged_text = "".join(result_parts).strip()

        # 2. Build emotion instruction prompt
        non_neutral = [s for s in segments if s.tone != Tone.NEUTRAL]

        if not non_neutral:
            prompt = "อ่านด้วยน้ำเสียงปกติ เป็นธรรมชาติ ชัดถ้อยชัดคำ"
            return RenderResponse(text=tagged_text, prompt=prompt)

        # Single tone across entire text
        tones = {s.tone for s in segments}
        if len(tones) == 1:
            tone = list(tones)[0]
            desc = RVC_EMOTION_PROMPT_MAP.get(tone, "น้ำเสียงปกติ")
            prompt = f"อ่านด้วย{desc}"
            return RenderResponse(text=tagged_text, prompt=prompt)

        # Multi-segment prompt breakdown
        parts = []
        for idx, seg in enumerate(segments, 1):
            desc = RVC_EMOTION_PROMPT_MAP.get(seg.tone, "น้ำเสียงปกติ")
            parts.append(f"ท่อนที่ {idx} \"{seg.text.strip()}\" ({desc})")

        prompt = "อ่านออกเสียงโดยปรับอารมณ์ตามแต่ละท่อน:\n" + "\n".join(parts)
        return RenderResponse(text=tagged_text, prompt=prompt)

from typing import List
from app.models import Segment, Tone, RenderResponse
from app.renderers.base import BaseRenderer

COSYVOICE_INSTRUCT_MAP = {
    Tone.NEUTRAL: "พูดด้วยน้ำเสียงปกติ เป็นธรรมชาติ ชัดถ้อยชัดคำ",
    Tone.SAD: "พูดด้วยน้ำเสียงเศร้า เสียใจ สะอื้นเล็กน้อย",
    Tone.HAPPY: "พูดด้วยน้ำเสียงร่าเริง สดใส มีความสุข",
    Tone.ANGRY: "พูดด้วยน้ำเสียงโกรธ ดุดัน ตะคอก",
    Tone.EXCITED: "พูดด้วยน้ำเสียงตื่นเต้น ดีใจสุดขีด มีพลัง",
    Tone.CALM: "พูดด้วยน้ำเสียงสงบ นุ่มนวล ผ่อนคลาย ช้าๆ",
    Tone.NERVOUS: "พูดด้วยน้ำเสียงประหม่า ลังเล หวาดหวั่น",
    Tone.SARCASTIC: "พูดด้วยน้ำเสียงประชดประชัน แดกดัน ยิ้มมุมปาก",
}

COSYVOICE_ACTION_TAG_MAP = {
    Tone.HAPPY: "<laughter>",
    Tone.EXCITED: "<laughter>",
    Tone.CALM: "<whisper>",
    Tone.NERVOUS: "<breath>",
}


class CosyVoiceRenderer(BaseRenderer):
    """
    Renders emotional segments into CosyVoice 2 syntax with <instruct>...</instruct>
    blocks and inline action tags like <laughter>, <whisper>, <breath>.
    """
    def render(self, segments: List[Segment]) -> RenderResponse:
        if not segments:
            return RenderResponse(text="", prompt=None)

        # 1. Build tagged text with <instruct> and action tags
        result_parts = []
        instructions = []

        for idx, seg in enumerate(segments):
            clean_t = seg.text.strip()
            if not clean_t:
                continue

            instruct_desc = COSYVOICE_INSTRUCT_MAP.get(seg.tone, "พูดด้วยน้ำเสียงปกติ")
            instructions.append(f"ท่อนที่ {idx + 1}: {instruct_desc}")

            if seg.tone != Tone.NEUTRAL:
                # Add instruct block
                instruct_block = f"<instruct>({instruct_desc})</instruct>"
                
                # Optional action tag for expressive tones
                action_tag = COSYVOICE_ACTION_TAG_MAP.get(seg.tone, "")
                if action_tag == "<whisper>":
                    result_parts.append(f"{instruct_block} <whisper> {clean_t} </whisper>")
                elif action_tag == "<laughter>" and seg.tone == Tone.EXCITED:
                    result_parts.append(f"{instruct_block} {clean_t} <laughter> ฮ่าๆ </laughter>")
                else:
                    result_parts.append(f"{instruct_block} {clean_t}")
            else:
                result_parts.append(clean_t)

        formatted_text = " ".join(result_parts).strip()
        summary_prompt = " | ".join(instructions)

        return RenderResponse(text=formatted_text, prompt=summary_prompt)

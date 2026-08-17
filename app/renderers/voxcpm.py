from typing import List
from app.models import Segment, Tone, RenderResponse
from app.renderers.base import BaseRenderer

VOXCPM_INSTRUCTIONS = {
    Tone.NEUTRAL: "",
    Tone.SAD: "(Sad and melancholic voice, slight sighs)",
    Tone.HAPPY: "(Happy and cheerful voice, smiling while speaking)",
    Tone.ANGRY: "(Angry, firm and aggressive tone)",
    Tone.EXCITED: "(Excited and energetic tone)",
    Tone.CALM: "(Calm and soothing voice, speaking softly)",
    Tone.NERVOUS: "(Nervous and trembling voice, hesitant)",
    Tone.SARCASTIC: "(Sarcastic and mocking tone)",
}


class VoxCPMRenderer(BaseRenderer):
    def render(self, segments: List[Segment]) -> RenderResponse:
        if not segments:
            return RenderResponse(text="", prompt=None)

        result_parts = []
        for seg in segments:
            inst = VOXCPM_INSTRUCTIONS.get(seg.tone, "")
            if inst:
                result_parts.append(f"{inst} {seg.text.strip()} ")
            else:
                result_parts.append(seg.text)

        rendered_text = "".join(result_parts).strip()
        return RenderResponse(text=rendered_text, prompt=None)

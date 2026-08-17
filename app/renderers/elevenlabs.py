from typing import List
from app.models import Segment, Tone, RenderResponse
from app.renderers.base import BaseRenderer

ELEVENLABS_TAG_MAPPING = {
    Tone.NEUTRAL: "",
    Tone.SAD: "[sad]",
    Tone.HAPPY: "[happily]",
    Tone.ANGRY: "[angry]",
    Tone.EXCITED: "[excited]",
    Tone.CALM: "[calm]",
    Tone.NERVOUS: "[nervous]",
    Tone.SARCASTIC: "[sarcastic]",
}


class ElevenLabsRenderer(BaseRenderer):
    def render(self, segments: List[Segment]) -> RenderResponse:
        if not segments:
            return RenderResponse(text="", prompt=None)

        result_parts = []
        for seg in segments:
            tag = ELEVENLABS_TAG_MAPPING.get(seg.tone, "")
            if tag:
                # Add tag before the segment text
                result_parts.append(f"{tag} {seg.text.strip()} ")
            else:
                result_parts.append(seg.text)

        rendered_text = "".join(result_parts).strip()
        return RenderResponse(text=rendered_text, prompt=None)

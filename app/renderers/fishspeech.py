from typing import List
from app.models import Segment, Tone, RenderResponse
from app.renderers.base import BaseRenderer

FISH_TONE_MAP = {
    Tone.NEUTRAL: "Neutral, calm and clear",
    Tone.SAD: "Melancholic, sad, tearful voice",
    Tone.HAPPY: "Joyful, cheerful, happy with a smile",
    Tone.ANGRY: "Furious, angry, shouting aggressively",
    Tone.EXCITED: "Extremely excited, energetic, enthusiastic",
    Tone.CALM: "Peaceful, soft, relaxing and warm",
    Tone.NERVOUS: "Nervous, hesitant, trembling slightly",
    Tone.SARCASTIC: "Sarcastic, cynical, mocking tone",
}


class FishSpeechRenderer(BaseRenderer):
    """
    Renders emotional segments into Fish Speech 1.5 prompt-conditioned format
    with [Tone: ...] tags and natural language acoustic prompts.
    """
    def render(self, segments: List[Segment]) -> RenderResponse:
        if not segments:
            return RenderResponse(text="", prompt=None)

        result_parts = []
        tone_descriptions = []

        for idx, seg in enumerate(segments):
            clean_t = seg.text.strip()
            if not clean_t:
                continue

            tone_desc = FISH_TONE_MAP.get(seg.tone, "Neutral, natural tone")
            tone_descriptions.append(f"Section {idx + 1}: {tone_desc}")

            if seg.tone != Tone.NEUTRAL:
                result_parts.append(f"[Tone: {tone_desc}] {clean_t}")
            else:
                result_parts.append(clean_t)

        formatted_text = " ".join(result_parts).strip()
        summary_prompt = " | ".join(tone_descriptions)

        return RenderResponse(text=formatted_text, prompt=summary_prompt)

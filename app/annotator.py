import json
import time
import logging
import re
from datetime import datetime
from typing import List, Tuple, Any, Optional
from app.config import settings
from app.models import Segment, Tone, AnnotateResponse, LLMAnnotationResult
from app.prompts import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, ANNOTATE_TOOL
from app.validator import validate_and_build_segments, ValidationError
from app.merger import merge_segments

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Anthropic Helpers
# ---------------------------------------------------------------------------

def build_anthropic_system_blocks() -> list:
    """Build Anthropic system message with prompt caching enabled."""
    return [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"}
        }
    ]


def build_anthropic_messages(clauses: List[str], guidance: Optional[str] = None) -> list:
    """Construct Anthropic conversation messages with few-shot examples."""
    messages = []
    for eg in FEW_SHOT_EXAMPLES:
        messages.append({
            "role": "user",
            "content": json.dumps(eg["input"], ensure_ascii=False)
        })
        messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": f"call_{eg['output']['labels'][0]['i']}",
                    "name": "annotate_clauses",
                    "input": eg["output"]
                }
            ]
        })

    user_payload: dict[str, Any] = {
        "clauses": [{"i": idx, "text": clause} for idx, clause in enumerate(clauses)]
    }
    if guidance and guidance.strip():
        user_payload["user_tone_guidance"] = guidance.strip()

    messages.append({
        "role": "user",
        "content": json.dumps(user_payload, ensure_ascii=False)
    })
    return messages


# ---------------------------------------------------------------------------
# Gemini (Google AI Studio) Helpers
# ---------------------------------------------------------------------------

def build_gemini_prompt(clauses: List[str], guidance: Optional[str] = None) -> str:
    """Build prompt for Gemini containing instructions, few-shot examples, and target clauses."""
    few_shot_text = ""
    for idx, eg in enumerate(FEW_SHOT_EXAMPLES, 1):
        few_shot_text += f"\n--- Example {idx} ---\nInput:\n{json.dumps(eg['input'], ensure_ascii=False)}\nOutput:\n{json.dumps(eg['output'], ensure_ascii=False)}\n"

    target_payload: dict[str, Any] = {
        "clauses": [{"i": idx, "text": clause} for idx, clause in enumerate(clauses)]
    }
    if guidance and guidance.strip():
        target_payload["user_tone_guidance"] = guidance.strip()

    return f"{few_shot_text}\n--- Target Task ---\nInput:\n{json.dumps(target_payload, ensure_ascii=False)}\nOutput:"


# ---------------------------------------------------------------------------
# Heuristic Fallback Helper
# ---------------------------------------------------------------------------

def detect_heuristic_tone(text: str, guidance: Optional[str] = None) -> Tone:
    """Intelligently detects emotion from bracket tags or Thai emotion keywords."""
    combined = f"{guidance or ''} {text}".lower()

    # 1. Bracket tag detection
    if "[calm]" in combined or "(calm" in combined:
        return Tone.CALM
    if "[sad]" in combined or "(sad" in combined:
        return Tone.SAD
    if "[angry]" in combined or "(angry" in combined:
        return Tone.ANGRY
    if "[happily]" in combined or "[happy]" in combined or "(happy" in combined:
        return Tone.HAPPY
    if "[excited]" in combined or "(excited" in combined:
        return Tone.EXCITED
    if "[nervous]" in combined or "(nervous" in combined:
        return Tone.NERVOUS
    if "[sarcastic]" in combined or "(sarcastic" in combined:
        return Tone.SARCASTIC

    # 2. Thai Keyword Heuristic
    if any(k in combined for k in ["excited", "ตื่นเต้น", "สุดยอด", "ดีใจสุดขีด", "เย้", "สำเร็จแล้ว"]):
        return Tone.EXCITED
    if any(k in combined for k in ["happy", "ดีใจ", "ร่าเริง", "ยินดี", "มีความสุข", "ยิ้ม"]):
        return Tone.HAPPY
    if any(k in combined for k in ["sad", "เศร้า", "เสียใจ", "ขอโทษ", "ตัดพ้อ", "ผิดหวัง"]):
        return Tone.SAD
    if any(k in combined for k in ["angry", "โกรธ", "ดุดัน", "โมโห", "เสียงแข็ง", "ไม่พอใจ", "พังหมด"]):
        return Tone.ANGRY
    if any(k in combined for k in ["calm", "สงบ", "นุ่มนวล", "ผ่อนคลาย", "ช้าๆ", "หายใจเข้า"]):
        return Tone.CALM
    if any(k in combined for k in ["sarcastic", "ประชด", "แดกดัน", "แหม", "เก่งจังเลย"]):
        return Tone.SARCASTIC
    if any(k in combined for k in ["nervous", "ประหม่า", "ลังเล", "กลัว", "กังวล"]):
        return Tone.NERVOUS

    return Tone.NEUTRAL


# ---------------------------------------------------------------------------
# Main Annotator Engine
# ---------------------------------------------------------------------------

class Annotator:
    def __init__(self, anthropic_client: Optional[Any] = None, gemini_client: Optional[Any] = None):
        self._anthropic_client = anthropic_client
        self._gemini_client = gemini_client

    def get_anthropic_client(self):
        if self._anthropic_client is not None:
            return self._anthropic_client
        import anthropic
        return anthropic.Anthropic(api_key=settings.anthropic_api_key or "dummy-key")

    def get_gemini_client(self):
        if self._gemini_client is not None:
            return self._gemini_client
        from google import genai
        return genai.Client(api_key=settings.effective_gemini_api_key or "dummy-key")

    def _call_anthropic(self, client: Any, model: str, clauses: List[str], guidance: Optional[str] = None) -> List[Any]:
        """Execute structured tool use call via Anthropic."""
        system_blocks = build_anthropic_system_blocks()
        messages = build_anthropic_messages(clauses, guidance=guidance)

        response = client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=0,
            system=system_blocks,
            messages=messages,
            tools=[ANNOTATE_TOOL],
            tool_choice={"type": "tool", "name": "annotate_clauses"}
        )

        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == "annotate_clauses":
                input_data = getattr(block, "input", {})
                if isinstance(input_data, dict) and "labels" in input_data:
                    return input_data["labels"]
                elif isinstance(input_data, list):
                    return input_data

        raise ValidationError("Anthropic LLM did not invoke annotate_clauses tool properly")

    def _call_gemini(self, client: Any, model: str, clauses: List[str], guidance: Optional[str] = None) -> List[Any]:
        """Execute Structured Output JSON call via Google AI Studio (Gemini)."""
        from google.genai import types

        prompt = build_gemini_prompt(clauses, guidance=guidance)
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=LLMAnnotationResult,
            temperature=0.0,
        )

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config
        )

        if not response.text:
            raise ValidationError("Gemini returned empty response")

        try:
            parsed = LLMAnnotationResult.model_validate_json(response.text)
            return [label.model_dump() for label in parsed.labels]
        except Exception as e:
            try:
                data = json.loads(response.text)
                if isinstance(data, dict) and "labels" in data:
                    return data["labels"]
                elif isinstance(data, list):
                    return data
            except Exception:
                pass
            raise ValidationError(f"Failed to parse Gemini output: {e}")

    def _run_provider(self, provider: str, model: str, clauses: List[str], guidance: Optional[str] = None) -> List[Any]:
        if provider == "gemini":
            client = self.get_gemini_client()
            return self._call_gemini(client, model, clauses, guidance=guidance)
        else:
            client = self.get_anthropic_client()
            return self._call_anthropic(client, model, clauses, guidance=guidance)

    def annotate(self, original_text: str, clauses: List[str], guidance: Optional[str] = None) -> AnnotateResponse:
        """
        Annotate clauses with emotional tones.
        Executes primary model -> fallback models -> intelligent heuristic tone extraction.
        """
        start_time = time.perf_counter()
        now_iso = datetime.now().isoformat()
        
        if not clauses:
            return AnnotateResponse(
                original=original_text,
                segments=[],
                model_used="none",
                fallback=False,
                latency_ms=0.0,
                clauses_count=0,
                timestamp=now_iso,
            )

        provider = settings.llm_provider.lower()
        if provider == "gemini":
            models_to_try = [
                settings.gemini_model,
                "gemini-3.5-flash",
                "gemini-3.7-flash",
                "gemini-3.1-flash-lite",
            ]
        else:
            models_to_try = [settings.llm_model, settings.llm_escalate_model]

        errors_encountered = []
        for model in models_to_try:
            if not model:
                continue
            try:
                raw_labels = self._run_provider(provider, model, clauses, guidance=guidance)
                segments = validate_and_build_segments(
                    original_text=original_text,
                    clauses=clauses,
                    raw_labels=raw_labels,
                    max_segments=settings.max_segments
                )
                from app.services.tts_service import tts_service
                tts_chunks, clean_tts_text = tts_service.build_tts_chunks(segments)
                latency = round((time.perf_counter() - start_time) * 1000, 2)
                return AnnotateResponse(
                    original=original_text,
                    segments=segments,
                    clean_tts_text=clean_tts_text,
                    tts_chunks=tts_chunks,
                    model_used=model,
                    fallback=False,
                    latency_ms=latency,
                    clauses_count=len(clauses),
                    timestamp=now_iso,
                )
            except Exception as err:
                err_msg = str(err)
                short_err = "429 RateLimit" if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg else ("503 Busy" if "503" in err_msg else "API Error")
                errors_encountered.append(f"{model}: {short_err}")
                logger.warning(f"Model {model} ({provider}) failed: {err}")

        # Intelligent Fallback: Extract tone from tags or keywords
        detected_tone = detect_heuristic_tone(original_text, guidance=guidance)
        clean_clauses = [re.sub(r"\[[a-zA-Z\s]+\]", "", c).strip() for c in clauses]
        clean_clauses = [c for c in clean_clauses if c] or [original_text]

        fallback_segments = []
        for c in clean_clauses:
            clause_tone = detect_heuristic_tone(c, guidance=guidance) or detected_tone
            fallback_segments.append(
                Segment(
                    text=c,
                    tone=clause_tone if clause_tone != Tone.NEUTRAL else detected_tone,
                    intensity=2
                )
            )

        merged_fallback = merge_segments(fallback_segments, max_segments=settings.max_segments)
        from app.services.tts_service import tts_service
        tts_chunks, clean_tts_text = tts_service.build_tts_chunks(merged_fallback)
        latency = round((time.perf_counter() - start_time) * 1000, 2)
        fallback_desc = f"API Unavailable ({', '.join(errors_encountered)}) -> Switched to Smart Rule-Based Emotion Engine" if errors_encountered else "Local Rule-Based Engine"

        return AnnotateResponse(
            original=original_text,
            segments=merged_fallback,
            clean_tts_text=clean_tts_text,
            tts_chunks=tts_chunks,
            model_used="rule-based-emotion-detector",
            fallback=True,
            fallback_reason=fallback_desc,
            latency_ms=latency,
            clauses_count=len(clauses),
            timestamp=now_iso,
        )


annotator = Annotator()

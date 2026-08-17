import os
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.models import (
    AnnotateRequest,
    AnnotateResponse,
    RenderRequest,
    RenderResponse,
    SpeakRequest,
    SpeakResponse,
    SpeakerListResponse,
    SynthesizeRequest,
)
from app.segmenter import segment_text
from app.annotator import annotator
from app.renderers import get_renderer
from app.services.tts_service import tts_service
from app.services.rvc_service import rvc_service
from app.services.speaker_manager import speaker_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize RVC models and voice directory on startup
    try:
        speakers = speaker_manager.list_speakers()
        print(f"[Startup] Thai TTS Tone & RVC Studio initialized with {len(speakers)} voice profiles.")
    except Exception as e:
        print(f"[Startup] Warning during startup initialization: {e}")
    yield


app = FastAPI(
    title="Thai TTS Tone Annotation & RVC Voice Studio",
    description="Expressive Thai Emotion-Instruction TTS with Retrieval-based Voice Conversion (RVC).",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
def root_ui():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Thai TTS Tone & RVC Voice API is running. Visit /docs for API documentation."}


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "thai-tts-tone-rvc",
        "version": "2.1.0",
        "speakers_count": len(speaker_manager.list_speakers()),
    }


# ---------------------------------------------------------------------------
# Tone & Emotion Annotation Endpoints
# ---------------------------------------------------------------------------

@app.post("/annotate", response_model=AnnotateResponse)
def annotate_endpoint(req: AnnotateRequest):
    """
    Segment input Thai text into clauses, query LLM for tone & intensity labels,
    validate, merge, and return annotated segments.
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    clauses = segment_text(text)
    response = annotator.annotate(original_text=text, clauses=clauses, guidance=req.guidance)
    return response


@app.post("/render", response_model=RenderResponse)
def render_endpoint(req: RenderRequest):
    """
    Render annotated segments for a specific engine (rvc, gemini, elevenlabs, voxcpm).
    """
    try:
        renderer = get_renderer(req.engine)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return renderer.render(req.segments)


@app.post("/speak", response_model=SpeakResponse)
def speak_endpoint(req: SpeakRequest):
    """
    End-to-end preparation pipeline:
    Receives raw text -> annotates tone -> renders engine payload.
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    clauses = segment_text(text)
    annotated = annotator.annotate(original_text=text, clauses=clauses, guidance=req.guidance)
    
    renderer = get_renderer(req.engine)
    rendered = renderer.render(annotated.segments)

    return SpeakResponse(
        engine=req.engine,
        text=rendered.text,
        prompt=rendered.prompt,
        segments=annotated.segments,
        model_used=annotated.model_used,
        fallback=annotated.fallback,
    )


# ---------------------------------------------------------------------------
# RVC Models & Voice Profiles Management Endpoints
# ---------------------------------------------------------------------------

@app.get("/speakers", response_model=SpeakerListResponse)
def list_speakers_endpoint():
    """List all available RVC models and voice profiles."""
    speakers = speaker_manager.list_speakers()
    return SpeakerListResponse(speakers=speakers)


@app.post("/speakers")
async def register_speaker_endpoint(
    file: UploadFile = File(...),
    speaker_id: Optional[str] = Form(None),
):
    """Upload an RVC model (.pth / .index) or audio reference clip."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    sid = speaker_id.strip() if speaker_id and speaker_id.strip() else os.path.splitext(file.filename)[0]
    result = speaker_manager.register_speaker(
        speaker_id=sid,
        file_bytes=content,
        filename=file.filename or "voice.wav",
    )
    return result


@app.delete("/speakers/{speaker_id}")
def delete_speaker_endpoint(speaker_id: str):
    """Remove an RVC model or voice profile."""
    deleted = speaker_manager.delete_speaker(speaker_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Speaker not found")
    return {"deleted": True, "speaker_id": speaker_id}


# ---------------------------------------------------------------------------
# Audio Synthesis & RVC Voice Conversion Endpoints
# ---------------------------------------------------------------------------

@app.post("/synthesize")
async def synthesize_endpoint(req: SynthesizeRequest):
    """
    Synthesizes expressive speech with emotional prompt instructions and converts voice with RVC.
    Returns 48kHz WAV audio stream.
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    emotion_prompt = req.guidance
    clean_text = text

    # If auto-annotate requested, extract emotions and build instruction prompt
    if req.auto_annotate and not text.startswith("("):
        clauses = segment_text(text)
        annotated = annotator.annotate(original_text=text, clauses=clauses, guidance=req.guidance)
        renderer = get_renderer("rvc")
        rendered = renderer.render(annotated.segments)
        emotion_prompt = rendered.prompt or req.guidance
        clean_text = rendered.text or text

    try:
        wav_bytes = rvc_service.synthesize_and_convert(
            text=clean_text,
            emotion_prompt=emotion_prompt,
            speaker_id=req.speaker_id,
            pitch_shift=req.pitch_shift,
            index_rate=req.index_rate,
            f0_method=req.f0_method,
            cfg_value=req.cfg_value,
            inference_timesteps=req.inference_timesteps,
        )
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": 'inline; filename="synthesized_rvc.wav"'},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis & Voice Conversion failed: {str(e)}")


@app.post("/synthesize/upload")
async def synthesize_with_upload_endpoint(
    text: str = Form(...),
    file: Optional[UploadFile] = File(None),
    guidance: Optional[str] = Form(None),
    pitch_shift: int = Form(0),
    index_rate: float = Form(0.75),
    f0_method: str = Form("rmvpe"),
    cfg_value: float = Form(2.5),
    inference_timesteps: int = Form(10),
    auto_annotate: bool = Form(True),
):
    """
    Synthesizes speech with a direct one-off uploaded reference audio or RVC model.
    """
    clean_text = text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    emotion_prompt = guidance
    if auto_annotate and not clean_text.startswith("("):
        clauses = segment_text(clean_text)
        annotated = annotator.annotate(original_text=clean_text, clauses=clauses, guidance=guidance)
        renderer = get_renderer("rvc")
        rendered = renderer.render(annotated.segments)
        emotion_prompt = rendered.prompt or guidance
        clean_text = rendered.text or clean_text

    audio_bytes = await file.read() if file else None
    speaker_id = None

    # Register temporary profile if file provided
    if audio_bytes and file:
        temp_speaker = speaker_manager.register_speaker(
            speaker_id=f"temp_{os.path.splitext(file.filename)[0]}",
            file_bytes=audio_bytes,
            filename=file.filename,
        )
        speaker_id = temp_speaker["id"]

    try:
        wav_bytes = rvc_service.synthesize_and_convert(
            text=clean_text,
            emotion_prompt=emotion_prompt,
            speaker_id=speaker_id,
            pitch_shift=pitch_shift,
            index_rate=index_rate,
            f0_method=f0_method,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
        )
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": 'inline; filename="synthesized_rvc_custom.wav"'},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


@app.post("/convert_voice")
async def convert_voice_endpoint(
    file: UploadFile = File(...),
    speaker_id: Optional[str] = Form(None),
    pitch_shift: int = Form(0),
    index_rate: float = Form(0.75),
    f0_method: str = Form("rmvpe"),
):
    """
    Direct voice conversion of any uploaded audio file via RVC.
    """
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Audio file is empty")

    try:
        converted, out_sr = rvc_service.convert_voice(
            audio_data=content,
            speaker_id=speaker_id,
            pitch_shift=pitch_shift,
            index_rate=index_rate,
            f0_method=f0_method,
        )
        import io
        import soundfile as sf
        out_buf = io.BytesIO()
        sf.write(out_buf, converted, out_sr, format="WAV", subtype="PCM_16")
        return Response(
            content=out_buf.getvalue(),
            media_type="audio/wav",
            headers={"Content-Disposition": 'inline; filename="rvc_converted.wav"'},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice conversion failed: {str(e)}")

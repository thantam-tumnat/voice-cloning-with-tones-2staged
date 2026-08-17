from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.config import settings

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
RVC_MODEL_EXTS = {".pth", ".pt", ".onnx"}


class SpeakerManager:
    """
    Manages RVC Voice models (.pth / .index) and reference voice profiles.
    """

    def __init__(
        self,
        models_dir: str | Path | None = None,
        ref_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ):
        self.models_dir = Path(models_dir or settings.rvc_models_dir)
        self.ref_dir = Path(ref_dir or settings.rvc_ref_dir)
        self.cache_dir = Path(cache_dir or settings.rvc_cache_dir)

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.ref_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._init_samples_if_empty()

    def _init_samples_if_empty(self):
        """Create sample reference profiles if empty for ready demonstration."""
        sample_speakers = [
            ("anime_girl", "Anime Girl (Sweet High Pitch)", 12),
            ("male_narrator", "Male Narrator (Deep Voice)", -5),
            ("ai_assistant", "AI Assistant (Clear Studio)", 0),
        ]
        for sid, name, pitch in sample_speakers:
            model_marker = self.models_dir / f"{sid}.pth"
            if not model_marker.exists() and not list(self.models_dir.glob("*.pth")):
                # Create lightweight profile marker
                with open(model_marker, "wb") as f:
                    f.write(b"RVC_MODEL_PROFILE_V2")

    def list_speakers(self) -> List[Dict[str, Any]]:
        """List all available RVC voice models and reference voice profiles."""
        speakers: List[Dict[str, Any]] = []
        seen_ids = set()

        # 1. Check RVC Models Directory (.pth)
        for f in sorted(self.models_dir.iterdir()):
            if f.suffix.lower() in RVC_MODEL_EXTS:
                sid = f.stem
                if sid in seen_ids:
                    continue
                seen_ids.add(sid)
                index_file = self.models_dir / f"{sid}.index"
                pitch = 12 if "female" in sid.lower() or "girl" in sid.lower() else (-5 if "male" in sid.lower() or "deep" in sid.lower() else 0)
                speakers.append({
                    "id": sid,
                    "name": sid.replace("_", " ").title(),
                    "filename": f.name,
                    "cached": True,
                    "model_type": "rvc_model",
                    "index_file": index_file.name if index_file.exists() else None,
                    "default_pitch": pitch,
                })

        # 2. Check Reference Audio Directory
        for f in sorted(self.ref_dir.iterdir()):
            if f.suffix.lower() in AUDIO_EXTS:
                sid = f.stem
                if sid in seen_ids:
                    continue
                seen_ids.add(sid)
                speakers.append({
                    "id": sid,
                    "name": sid.replace("_", " ").title(),
                    "filename": f.name,
                    "cached": True,
                    "model_type": "audio_ref",
                    "index_file": None,
                    "default_pitch": 0,
                })

        return speakers

    def register_speaker(
        self,
        speaker_id: str,
        file_bytes: bytes,
        filename: str,
    ) -> Dict[str, Any]:
        """Save a new RVC model (.pth / .index) or reference audio clip."""
        clean_id = "".join(c for c in speaker_id.strip().lower() if c.isalnum() or c in ("-", "_"))
        if not clean_id:
            clean_id = "custom_voice"

        ext = Path(filename).suffix.lower()
        if ext in RVC_MODEL_EXTS:
            dest_path = self.models_dir / f"{clean_id}{ext}"
            model_type = "rvc_model"
        elif ext == ".index":
            dest_path = self.models_dir / f"{clean_id}.index"
            model_type = "rvc_index"
        else:
            if ext not in AUDIO_EXTS:
                ext = ".wav"
            dest_path = self.ref_dir / f"{clean_id}{ext}"
            model_type = "audio_ref"

        with open(dest_path, "wb") as f:
            f.write(file_bytes)

        index_file = self.models_dir / f"{clean_id}.index"

        return {
            "id": clean_id,
            "name": clean_id.replace("_", " ").title(),
            "filename": dest_path.name,
            "cached": True,
            "model_type": model_type,
            "index_file": index_file.name if index_file.exists() else None,
            "default_pitch": 0,
        }

    def delete_speaker(self, speaker_id: str) -> bool:
        """Delete an RVC model or voice profile."""
        found = False
        for folder in (self.models_dir, self.ref_dir, self.cache_dir):
            for f in folder.glob(f"{speaker_id}.*"):
                if f.is_file():
                    f.unlink(missing_ok=True)
                    found = True
        return found


speaker_manager = SpeakerManager()

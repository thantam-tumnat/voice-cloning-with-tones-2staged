# Thai TTS Tone Annotation & RVC Voice Studio

FastAPI service and Interactive Web Studio for analyzing emotional tones and intensities in Thai text, synthesizing expressive speech via **Emotion-Instruction TTS**, and converting the speaker's vocal timbre into any target voice with **Retrieval-based Voice Conversion (RVC)**.

---

## Architecture Overview

```
Thai Raw Text / Script (e.g. '[calm] หายใจเข้า...') ───► 1. SEGMENT & ANNOTATE (PyThaiNLP + Gemini LLM)
                                                                 │
                                                                 ▼
Target RVC Model (.pth / .index) ──────────────► 2. EMOTION INSTRUCTION TTS
(or Uploaded Reference Audio)                       (Synthesizes audio with emotion prompt/tags)
                                                                 │
                                                                 ▼
                                                        3. RVC VOICE CONVERTER
                                                           (F0 Pitch Extraction [RMVPE/Harvest/PM]
                                                            + FAISS Index Feature Retrieval
                                                            + Timbre/Formant Transformation)
                                                                 │
                                                                 ▼
                                                        4. 48kHz WAV Audio Output
```

---

## Tone Enum & Emotion Mapping

| Tone | RVC / TTS Emotion Prompt (Thai) | ElevenLabs Tag | VoxCPM Instruction |
|---|---|---|---|
| `neutral` | น้ำเสียงปกติ เป็นธรรมชาติ ชัดถ้อยชัดคำ | *(no tag)* | *(no instruction)* |
| `sad` | น้ำเสียงเศร้า สั่นเครือ แฝงความเสียใจ | `[sad]` | `(Sad and melancholic voice, slight sighs)` |
| `happy` | น้ำเสียงสดใส ร่าเริง ยิ้มแย้มขณะพูด | `[happily]` | `(Happy and cheerful voice, smiling while speaking)` |
| `angry` | น้ำเสียงโกรธ ดุดัน กระแทกเสียง | `[angry]` | `(Angry, firm and aggressive tone)` |
| `excited` | น้ำเสียงตื่นเต้น กระตือรือร้น มีพลัง | `[excited]` | `(Excited and energetic tone)` |
| `calm` | น้ำเสียงสงบ นุ่มนวล ช้าๆ ผ่อนคลาย | `[calm]` | `(Calm and soothing voice, speaking softly)` |
| `nervous` | น้ำเสียงประหม่า ลังเล หวาดหวั่น | `[nervous]` | `(Nervous and trembling voice, hesitant)` |
| `sarcastic` | น้ำเสียงประชดประชัน แดกดัน กวนๆ | `[sarcastic]` | `(Sarcastic and mocking tone)` |

---

## Setup & Running

### 1. Install Dependencies
```bash
py -m pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env` and configure your API keys:
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
RVC_MODELS_DIR=models/rvc
```

### 3. Run FastAPI Server & Web Studio
```bash
py -m uvicorn app.main:app --reload --port 8000
```
Then open your browser and navigate to:
👉 **`http://localhost:8000/`** to access the interactive **Thai TTS & RVC Voice Studio**.

### 4. Run Test Suite
```bash
py -m pytest -v
```

---

## Web Studio Features

- **Interactive Script & Emotion Editor**: Real-time parsing of bracket tags like `[calm] ...` into audio instructions.
- **RVC Voice Changer & Model Selector**:
  - Select target RVC models from `models/rvc/` or registered reference voices.
  - Drag-and-drop / Upload any `.pth` model or reference audio clip (`.wav`, `.mp3`) for zero-shot voice conversion.
- **RVC Pitch & Feature Controls**:
  - **Pitch Shift**: Semitone adjustment (-12 to +12, for female / male pitch conversion).
  - **Index Rate**: Feature retrieval strength (0.0 - 1.0).
  - **F0 Pitch Algorithm**: Switch between `RMVPE`, `Harvest`, `PM`, and `CREPE`.
- **Live Visual Tag & Instruction Preview**: Dynamic highlighting of emotion tags and prompts.
- **Built-in Audio Player Studio**: 1-click **"🎙️ สร้างเสียงพูด (Synthesize & RVC)"** with waveform player and WAV download.

---

## API Endpoints

### `POST /synthesize`
Synthesizes speech with emotion instructions and converts voice with target RVC model:
```json
{
  "text": "[calm] หายใจเข้าลึกๆ ผ่อนคลาย แล้วค่อยๆ ปล่อยวางทุกอย่างลงนะ",
  "speaker_id": "anime_girl",
  "engine": "rvc",
  "pitch_shift": 12,
  "index_rate": 0.75,
  "f0_method": "rmvpe",
  "auto_annotate": true
}
```
*Returns: `audio/wav` binary stream (48kHz)*

### `POST /synthesize/upload`
Synthesizes speech with a direct one-off uploaded reference audio or RVC model file (Multipart Form).

### `POST /convert_voice`
Converts any uploaded audio file directly into the target voice using RVC.

### `GET /speakers`
Lists all available RVC models (`.pth`) and registered voice profiles.

### `POST /speakers`
Uploads a new RVC model (`.pth` / `.index`) or reference voice file.

### `DELETE /speakers/{speaker_id}`
Deletes an RVC model profile.

### `POST /annotate`
Analyzes raw Thai text into emotional clauses and intensities.

### `POST /render`
Renders annotated segments into engine-specific formats (`rvc`, `gemini`, `elevenlabs`, `voxcpm`).

### `GET /health`
Health check endpoint returning system status and voice count.

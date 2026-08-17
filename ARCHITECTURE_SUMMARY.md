# สรุปสถาปัตยกรรมและโมเดลทั้งหมด (System Architecture & Model Summary)
## Thai TTS Tone Annotation & RVC Voice Studio (2-Staged Voice Cloning)

---

## 📌 ภาพรวมระบบ (System Overview)
โปรเจกต์นี้เป็นระบบสังเคราะห์เสียงพูดภาษาไทยพร้อมอารมณ์และแปลงเสียงตัวละครแบบ 2 สเตจ (**2-Staged Voice Cloning Architecture**) ที่ผสานความฉลาดในการเข้าใจบริบทของ **LLM**, ความชัดเจนถูกต้องตามหลักภาษาไทยของ **Neural TTS**, และความสามารถในการเปลี่ยนเนื้อเสียงของ **RVC v2**

```
[ ข้อความภาษาไทย / แท็กอารมณ์ ]
              │
              ▼ (Stage 1: Voice Director)
     [ LLM Emotion Engine ]  ──► วิเคราะห์ Tone (8 อารมณ์) & Intensity
              │
              ▼ (Stage 2: Voice Generation)
    [ Neural Thai TTS Engine ] ──► สังเคราะห์เสียงพูดภาษาไทย (48kHz Waveform)
              │
              ▼ (Stage 3: Voice Conversion)
       [ RVC v2 Voice Engine ]  ──► สกัด F0 (RMVPE) + สวมทับ Timbre ตัวละคร
              │
              ▼
[ เสียงพากย์ภาษาไทยอารมณ์สมจริงในโทนเสียงของตัวละครเป้าหมาย (48kHz WAV) ]
```

---

## 🤖 1. รายชื่อโมเดลและเทคโนโลยีทั้งหมดที่ใช้

### 1.1 สมองสั่งการอารมณ์ (LLM Emotion Analysis)
* **โมเดลหลัก (Primary)**:
  * `gemini-2.5-flash` / `gemini-3.5-flash` *(Google DeepMind)*
* **โมเดลสำรอง (Escalation Fallbacks)**:
  * `gemini-3.7-flash` / `gemini-3.1-flash-lite` *(Google DeepMind)*
  * `claude-3-5-sonnet` / `claude-3-haiku` *(Anthropic)*
* **ระบบตรวจจับอารมณ์ออฟไลน์ (Heuristic Fallback Engine)**:
  * Rule-based Keyword & Tag Regex Analyzer ทำงานได้ 100% แม้ไม่มีอินเทอร์เน็ตหรือติด Quota Limit

### 1.2 การสังเคราะห์เสียงพูดภาษาไทย (Thai Neural TTS)
* **โมเดลหลัก (Primary Neural TTS)**:
  * `th-TH-PremwadeeNeural` *(Microsoft Edge Neural TTS - เสียงผู้หญิง)*
  * `th-TH-NiwatNeural` *(Microsoft Edge Neural TTS - เสียงผู้ชาย)*
* **โมเดลสำรอง (Secondary Fallback)**:
  * `gTTS (Google Text-to-Speech Engine)` *(Google Translate TTS)*

### 1.3 การแปลงเสียงตัวละคร (RVC Voice Conversion)
* **สถาปัตยกรรมหลัก**: `RVC v2 (Retrieval-based Voice Conversion Version 2)`
* **โมเดลสกัดเส้นเสียง Pitch (F0 Estimators)**:
  * `RMVPE` *(Robust Model for Vocal Pitch Estimation - Neural Network)*
  * `Harvest`, `PM (Parselmouth)`, `CREPE`
* **ระบบสืบค้นคุณลักษณะเสียง (Feature Retrieval)**:
  * `FAISS Indexing (.index file)`
* **โมเดลน้ำหนักเสียงตัวละคร (Model Checkpoints)**:
  * PyTorch Checkpoint (`.pth`) เช่น Anime Girl, Male Narrator หรือไฟล์โมเดลที่ User อัปโหลด
* **ระบบเลื่อนระดับเสียง (Pitch Shifter & DSP)**:
  * `Torchaudio Neural STFT Phase-Vocoder` (ปรับคีย์ -24 ถึง +24 Semitones)

---

## 🔄 2. การรับ-ส่งข้อมูลระหว่าง Component (Data Flow & Parameters)

### 2.1 ข้อมูลที่ส่งเข้าโมเดล TTS (Text-to-Speech)
TTS ทำหน้าที่แปลงตัวหนังสือภาษาไทยให้ออกมาเป็นคลื่นเสียง (Waveform):
1. **`clean_text` / `clean_tts_text`**: ตัวหนังสือภาษาไทยแท้ ๆ ที่ตัดแท็ก `[tag]` ออกแล้ว
2. **`prosody_rate`**: ความเร็วการพูด เช่น `+20%` (ตื่นเต้น) หรือ `-15%` (เศร้า)
3. **`prosody_pitch`**: คีย์เสียงสูง-ต่ำตามอารมณ์ เช่น `+12Hz` หรือ `-8Hz`
4. **`voice`**: เสียงผู้พูดตั้งต้น (`th-TH-PremwadeeNeural`)
5. **`prompt`**: คำสั่งกำกับบริบทอารมณ์ที่ได้จาก LLM

### 2.2 ข้อมูลที่ส่งเข้าโมเดล RVC (Voice Conversion)
RVC เป็นโมเดล Audio-to-Audio (ไม่รับ Text) โดยจะรับเฉพาะ:
1. **`audio_data`**: ข้อมูลคลื่นเสียงดิบ 48,000 Hz ที่เพิ่งสร้างเสร็จจากโมเดล TTS
2. **`speaker_id` / โมเดล RVC**: ไฟล์โมเดล `.pth` และ `.index` ของตัวละครเป้าหมาย
3. **`pitch_shift`**: ระดับการเลื่อนคีย์เสียง (-24 ถึง +24 semitones)
4. **`index_rate`**: อัตราการดึงเอกลักษณ์เสียงต้นแบบ (0.00 – 1.00)
5. **`f0_method`**: อัลกอริทึมแกะรอยระดับเสียง (`rmvpe`)

---

## 📊 3. ตารางเปรียบเทียบ 8 โทนอารมณ์กับค่าพารามิเตอร์ TTS (Prosody Mapping)

| โทนอารมณ์ (Tone) | ตัวอย่างบริบทคำพูด | ความเร็ว (`prosody_rate`) | คีย์เสียง (`prosody_pitch`) |
| :--- | :--- | :---: | :---: |
| **`excited`** | ตื่นเต้น, ดีใจสุดขีด, สุดยอดไปเลย | `+20%` | `+12Hz` |
| **`happy`** | ร่าเริง, ยินดี, ยิ้มแย้ม | `+10%` | `+8Hz` |
| **`angry`** | โกรธ, ดุดัน, เสียงแข็ง | `+15%` | `+10Hz` |
| **`sarcastic`** | ประชดประชัน, แดกดัน | `-5%` | `+6Hz` |
| **`nervous`** | ประหม่า, กังวล, ลังเล | `+8%` | `+5Hz` |
| **`neutral`** | อ่านข่าว, สุภาพ, เป็นทางการ | `+0%` | `+0Hz` |
| **`calm`** | สงบ, นุ่มนวล, ผ่อนคลาย | `-12%` | `-4Hz` |
| **`sad`** | เศร้า, เสียใจ, ตัดพ้อ | `-15%` | `-8Hz` |

---

## 🔍 4. โครงสร้างผลลัพธ์ JSON สำหรับ Debug (`Raw JSON & Diagnostics`)
```json
{
  "engine": "rvc",
  "text": "[excited] ยินดีด้วยนะ! ในที่สุดก็ทำสำเร็จแล้ว [sad] แต่เหนื่อยมากเลย",
  "clean_tts_text": "ยินดีด้วยนะ! ในที่สุดก็ทำสำเร็จแล้ว แต่เหนื่อยมากเลย",
  "tts_chunks": [
    {
      "chunk_index": 1,
      "raw_text": "ยินดีด้วยนะ! ในที่สุดก็ทำสำเร็จแล้ว",
      "clean_text": "ยินดีด้วยนะ! ในที่สุดก็ทำสำเร็จแล้ว",
      "tone": "excited",
      "prosody_rate": "+20%",
      "prosody_pitch": "+12Hz",
      "voice": "th-TH-PremwadeeNeural",
      "char_length": 34
    },
    {
      "chunk_index": 2,
      "raw_text": "แต่เหนื่อยมากเลย",
      "clean_text": "แต่เหนื่อยมากเลย",
      "tone": "sad",
      "prosody_rate": "-15%",
      "prosody_pitch": "-8Hz",
      "voice": "th-TH-PremwadeeNeural",
      "char_length": 16
    }
  ],
  "prompt": "อ่านออกเสียงโดยปรับอารมณ์ตามแต่ละท่อน...",
  "segments": [ ... ],
  "model_used": "gemini-3.5-flash",
  "fallback": false,
  "fallback_reason": null,
  "latency_ms": 312.4,
  "clauses_count": 2,
  "timestamp": "2026-08-17T16:50:00.000000"
}
```

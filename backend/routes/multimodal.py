from fastapi import APIRouter, File, UploadFile, Form
import shutil
import os
import uuid

from services.audio_model import analyze_audio, reset_buffer
from services.face_model import analyze_face
from services.llm_service import generate_multimodal_report

router = APIRouter(prefix="/multimodal")

@router.post("/")
async def multimodal_analyze(
    phq9_score: int = Form(0),
    gad7_score: int = Form(0),
    audio_file: UploadFile = File(...)
):
    # Reset audio smoothing buffer for fresh session chunks
    reset_buffer()
    
    # ── 1. Save uploaded audio temporarily
    temp_filename = f"temp_{uuid.uuid4()}.wav"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    try:
        # ── 2. Audio analysis
        audio_result = analyze_audio(temp_filename)
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    # ── 3. Facial analysis (captures from webcam)
    face_result = analyze_face()

    # ── 4. Risk assessment
    if phq9_score > 14 or gad7_score > 14:
        risk = "high"
    elif phq9_score > 8:
        risk = "moderate"
    else:
        risk = "low"

    # ── 5. Combined result
    result = {
        "phq9_score": phq9_score,
        "gad7_score": gad7_score,
        "audio_emotion": audio_result.get("emotion"),
        "audio_energy": audio_result.get("speech_energy", "unknown"),
        "audio_stability": audio_result.get("stability", "unknown"),
        "audio_behavior": audio_result.get("behavior_flag", ""),
        "facial_emotion": face_result.get("emotion"),
        "facial_confidence": face_result.get("confidence"),
        "facial_behavior": face_result.get("behavior_flag", ""),
        "risk_level": risk,
    }

    result["report"] = generate_multimodal_report(result)

    return result

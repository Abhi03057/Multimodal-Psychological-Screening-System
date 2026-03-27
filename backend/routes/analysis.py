from fastapi import APIRouter, File, UploadFile, Form
import shutil
import os
import uuid

from services.audio_model import analyze_audio
from services.llm_service import generate_report

router = APIRouter(prefix="/analysis")

@router.post("/")
async def analyze(
    phq9_score: int = Form(0),
    gad7_score: int = Form(0),
    audio_file: UploadFile = File(...)
):
    # Save the uploaded file temporarily
    temp_filename = f"temp_{uuid.uuid4()}.wav"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)
        
    try:
        # Run audio analysis
        audio_result = analyze_audio(temp_filename)
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    if phq9_score > 14 or gad7_score > 14:
        risk = "high"
    elif phq9_score > 8:
        risk = "moderate"
    else:
        risk = "low"

    result = {
        "phq9_score": phq9_score,
        "gad7_score": gad7_score,
        "audio_features": {
            "energy": audio_result.get("speech_energy", "unknown"),
            "pause_rate": audio_result.get("stability", "unknown")
        },
        "facial_emotion": "unknown", # Placeholder for future feature
        "risk_level": risk,
        "detected_emotion": audio_result.get("emotion")
    }

    result["report"] = generate_report(result)

    return result
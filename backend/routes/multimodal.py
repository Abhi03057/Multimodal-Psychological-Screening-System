import shutil, os, uuid
from fastapi import APIRouter, File, UploadFile, Form
from services.audio_model import analyze_audio, reset_buffer
from services.face_model import analyze_single_frame
from services.llm_service import generate_multimodal_report

router = APIRouter(prefix="/multimodal")

@router.post("/")
async def multimodal_analyze(
    phq9_score: int = Form(0),
    gad7_score: int = Form(0),
    audio_file: UploadFile = File(...)
):
    temp_filename = f"temp_{uuid.uuid4()}.wav"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    try:
        audio_result = analyze_audio(temp_filename)
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    reset_buffer()

    total = phq9_score + gad7_score
    if total >= 14:
        risk_level = "high"
    elif total >= 8:
        risk_level = "moderate"
    else:
        risk_level = "low"

    data = {
        "phq9_score": phq9_score,
        "gad7_score": gad7_score,
        "audio_emotion": audio_result.get("emotion", "unknown"),
        "audio_energy": audio_result.get("speech_energy", "unknown"),
        "audio_stability": audio_result.get("stability", "unknown"),
        "facial_emotion": "unknown",
        "facial_confidence": 0.0,
        "risk_level": risk_level,
    }
    report = generate_multimodal_report(data)
    return {**data, "report": report}

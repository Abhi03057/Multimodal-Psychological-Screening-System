from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse
import shutil
import os
import uuid
import cv2
import numpy as np
import traceback

from services.audio_model import analyze_audio, reset_buffer
from services.face_model import analyze_single_frame, FACE_BEHAVIOR_MAP
from services.llm_service import generate_multimodal_report

router = APIRouter(prefix="/live")

@router.post("/analyze")
async def live_analyze(
    phq9_score: int = Form(0),
    gad7_score: int = Form(0),
    audio_file: UploadFile = File(...),
    frame_file: UploadFile = File(None),
):
    try:
        reset_buffer()

        # ── 1. Audio analysis
        temp_audio = f"temp_{uuid.uuid4()}.wav"
        with open(temp_audio, "wb") as buf:
            shutil.copyfileobj(audio_file.file, buf)

        try:
            audio_result = analyze_audio(temp_audio)
        finally:
            if os.path.exists(temp_audio):
                os.remove(temp_audio)

        # ── 2. Facial analysis from uploaded frame
        face_emotion = "unknown"
        face_confidence = 0.0
        face_behavior = "no frame provided"

        if frame_file is not None:
            try:
                contents = await frame_file.read()
                nparr = np.frombuffer(contents, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    emotion, conf, scores = analyze_single_frame(frame)
                    face_emotion = emotion
                    face_confidence = float(round(conf, 2))
                    face_behavior = FACE_BEHAVIOR_MAP.get(emotion, "expression under analysis")
            except Exception as e:
                face_emotion = "error"
                face_behavior = str(e)

        # ── 3. Risk assessment
        if phq9_score > 14 or gad7_score > 14:
            risk = "high"
        elif phq9_score > 8:
            risk = "moderate"
        else:
            risk = "low"

        result = {
            "phq9_score": phq9_score,
            "gad7_score": gad7_score,
            "audio_emotion": audio_result.get("emotion", "unknown"),
            "audio_energy": audio_result.get("speech_energy", "unknown"),
            "audio_stability": audio_result.get("stability", "unknown"),
            "audio_behavior": audio_result.get("behavior_flag", ""),
            "facial_emotion": face_emotion,
            "facial_confidence": face_confidence,
            "facial_behavior": face_behavior,
            "risk_level": risk,
        }

        result["report"] = generate_multimodal_report(result)
        return result

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "detail": traceback.format_exc()},
        )


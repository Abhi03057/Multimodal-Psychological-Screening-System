import shutil, os, uuid, traceback
from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from collections import Counter
import cv2
import numpy as np

from services.audio_model import analyze_audio, reset_buffer
from services.face_model import analyze_single_frame, FACE_BEHAVIOR_MAP
from services.llm_service import generate_multimodal_report, generate_llm_report, generate_llm_report_v2
from config.questions import GUIDED_QUESTIONS, BASELINE_GROUP, STRESS_GROUP, VALID_QUESTION_TYPES

router = APIRouter(prefix="/live")

ENERGY_NORM_MAP    = {"low": 0.2, "medium": 0.5, "high": 0.8, "unknown": 0.3}
STABILITY_NORM_MAP = {"stable": 0.1, "moderate": 0.5, "variable": 0.9, "unknown": 0.5}
AUDIO_STRESS_MAP   = {"angry":0.9,"fear":0.85,"sad":0.7,"disgust":0.65,"surprise":0.4,
                      "neutral":0.1,"calm":0.1,"happy":0.05,"uncertain":0.3,"no speech detected":0.0}
FACE_DISTRESS_MAP  = {"angry":0.85,"fear":0.9,"sad":0.75,"disgust":0.7,"surprise":0.4,
                      "happy":0.05,"neutral":0.15,"calm":0.1,"unknown":0.3,"error":0.3}


class ChunkResult(BaseModel):
    """Schema for a single analysis chunk in session history."""
    audio_emotion:    str   = "unknown"
    audio_energy:     str   = "unknown"
    audio_stability:  str   = "unknown"
    facial_emotion:   str   = "unknown"
    facial_confidence: float = 0.0
    risk_level:       str   = "low"
    question_type:    str   = "unknown"
    voice_score:      float = Field(default=0.5, ge=0.0, le=1.0)
    face_score:       float = Field(default=0.5, ge=0.0, le=1.0)
    emotion_probs:    dict  = {}


class EndSessionRequest(BaseModel):
    """Schema for the end-session aggregation request."""
    phq9_score: int = 0
    gad7_score: int = 0
    history: List[dict] = []


def compute_voice_score(energy, stability, audio_emotion):
    """Compute a normalized voice stress score in [0, 1]."""
    e = ENERGY_NORM_MAP.get(energy, 0.3)
    s = STABILITY_NORM_MAP.get(stability, 0.5)
    a = AUDIO_STRESS_MAP.get(audio_emotion, 0.3)
    return round(min(0.4*e + 0.3*s + 0.3*a, 1.0), 4)


def compute_face_score(facial_emotion, confidence, emotion_probs):
    """Compute a normalized facial distress score in [0, 1]."""
    if emotion_probs:
        face_base = sum(FACE_DISTRESS_MAP.get(k, 0.3)*v for k,v in emotion_probs.items()) * 1.5
    else:
        face_base = FACE_DISTRESS_MAP.get(facial_emotion, 0.3)
    conf_adj = 0.4 * min(float(confidence), 1.0)
    return round(min(0.3*face_base + conf_adj, 1.0), 4)


def _safe_mean(values, default=0.5):
    """Compute mean with division-by-zero protection."""
    return round(sum(values)/len(values), 4) if values else default


def _safe_std(values, default=0.0):
    """Compute population standard deviation safely."""
    if not values:
        return default
    mean = sum(values) / len(values)
    var  = sum((x - mean)**2 for x in values) / len(values)
    return round(var**0.5, 4)


def _majority_vote(items, default="unknown"):
    """Return the most common item, or default if list is empty."""
    filtered = [x for x in items if x]
    return Counter(filtered).most_common(1)[0][0] if filtered else default


def aggregate_by_question(chunks):
    """Group chunks by question_type and compute per-group statistics."""
    groups = {}
    for c in chunks:
        qt = c.get("question_type", "unknown")
        if qt not in groups:
            groups[qt] = {"voice_scores": [], "face_scores": [], "confidences": []}
        groups[qt]["voice_scores"].append(c.get("voice_score", 0.5))
        groups[qt]["face_scores"].append(c.get("face_score", 0.5))
        groups[qt]["confidences"].append(c.get("facial_confidence", 0.0))
    result = {}
    for qt, g in groups.items():
        result[qt] = {
            "avg_voice":      _safe_mean(g["voice_scores"]),
            "avg_face":       _safe_mean(g["face_scores"]),
            "avg_confidence": _safe_mean(g["confidences"]),
            "count":          len(g["voice_scores"]),
        }
    return result


def compute_reactivity(per_question):
    """Compute emotional reactivity as elevation from baseline to stress."""
    baseline_scores = [
        (per_question[qt]["avg_voice"] + per_question[qt]["avg_face"]) / 2.0
        for qt in per_question if qt in BASELINE_GROUP
    ]
    stress_scores = [
        (per_question[qt]["avg_voice"] + per_question[qt]["avg_face"]) / 2.0
        for qt in per_question if qt in STRESS_GROUP
    ]
    b = _safe_mean(baseline_scores, 0.3)
    s = _safe_mean(stress_scores,   0.5)
    return round(min(max(s - b, 0.0) * 1.0, 1.0), 4)


def compute_fusion_and_confidence(phq9, gad7, voice_norm, face_norm, reactivity_norm):
    """Compute final fused risk score, confidence, and alignment.
    FinalScore = 0.30*PHQ_norm + 0.25*GAD_norm + 0.20*voice + 0.15*face + 0.10*reactivity
    """
    phq9_norm = round(min(phq9 / 27.0, 1.0), 4)
    gad7_norm = round(min(gad7 / 21.0, 1.0), 4)
    final_score = round(
        0.30*phq9_norm + 0.25*gad7_norm +
        0.20*voice_norm + 0.15*face_norm + 0.10*reactivity_norm, 4
    )
    if phq9 >= 14 or final_score >= 0.65:
        risk_level = "High"
    elif phq9 >= 8 or final_score >= 0.4:
        risk_level = "Moderate"
    else:
        risk_level = "Low"
    modal_diff = abs(voice_norm - face_norm)
    confidence_score = round(max(1.0 - modal_diff*2, 0.0), 2)
    alignment_level  = "High" if confidence_score >= 0.75 else ("Moderate" if confidence_score >= 0.35 else "Low")
    return {
        "phq9_score": phq9, "gad7_score": gad7,
        "phq9_norm": phq9_norm, "gad7_norm": gad7_norm,
        "voice_norm": voice_norm, "face_norm": face_norm,
        "reactivity_score": reactivity_norm,
        "final_score": final_score,
        "confidence_score": confidence_score,
        "alignment_level": alignment_level,
        "risk_level": risk_level,
    }


@router.get("/questions")
async def get_questions():
    """Return the guided session prompts to the frontend."""
    return {"questions": GUIDED_QUESTIONS}


@router.post("/analyze")
async def live_analyze(
    phq9_score:    int        = Form(0),
    gad7_score:    int        = Form(0),
    audio_file:    UploadFile = File(...),
    frame_file:    Optional[UploadFile] = File(None),
    question_type: str        = Form("unknown"),
):
    temp_audio = f"temp_{uuid.uuid4()}.wav"
    with open(temp_audio, "wb") as buf:
        shutil.copyfileobj(audio_file.file, buf)

    # Facial analysis
    facial_emotion    = "unknown"
    facial_confidence = 0.0
    facial_behavior   = "no frame provided"
    facial_probs      = {}

    if frame_file:
        try:
            img_bytes = await frame_file.read()
            img_arr   = np.frombuffer(img_bytes, dtype=np.uint8)
            img       = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img is not None:
                face_result       = analyze_single_frame(img)
                facial_emotion    = face_result.get("emotion", "unknown")
                facial_confidence = float(face_result.get("confidence", 0.0))
                facial_behavior   = face_result.get("behavior", "expression under analysis")
                facial_probs      = {k: float(v) for k,v in face_result.get("probs", {}).items()}
        except Exception:
            facial_behavior = "error"

    # Audio analysis
    try:
        audio_result    = analyze_audio(temp_audio)
        audio_emotion   = audio_result.get("emotion", "unknown")
        audio_energy    = audio_result.get("speech_energy", "unknown")
        audio_stability = audio_result.get("stability", "unknown")
        audio_behavior  = audio_result.get("behavior_flag", "")
    except Exception:
        audio_emotion = audio_energy = audio_stability = audio_behavior = "unknown"
    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

    total = phq9_score + gad7_score
    risk_level = "high" if total >= 14 else ("moderate" if total >= 8 else "low")

    voice_s = compute_voice_score(audio_energy, audio_stability, audio_emotion)
    face_s  = compute_face_score(facial_emotion, facial_confidence, facial_probs)

    data = {
        "phq9_score": phq9_score, "gad7_score": gad7_score,
        "audio_emotion": audio_emotion, "audio_energy": audio_energy,
        "audio_stability": audio_stability, "audio_behavior": audio_behavior,
        "facial_emotion": facial_emotion, "facial_confidence": facial_confidence,
        "facial_behavior": facial_behavior,
        "risk_level": risk_level,
        "voice_score": voice_s, "face_score": face_s,
    }

    try:
        report = generate_multimodal_report(data)
        data["report"] = report
        return data
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "detail": traceback.format_exc()})


@router.post("/end-session")
async def end_session(req: EndSessionRequest):
    """Aggregate accumulated live session data and generate a full LLM report."""
    try:
        history = req.history
        chunks_analyzed = len(history)

        if not history:
            aggregated = compute_fusion_and_confidence(req.phq9_score, req.gad7_score, 0.5, 0.5, 0.0)
            report = generate_llm_report({**aggregated, "audio_emotion": "unknown", "facial_emotion": "unknown"})
            return {"phq9_score": req.phq9_score, "gad7_score": req.gad7_score,
                    "aggregated": aggregated, "chunks_analyzed": 0, "report": report}

        voice_scores = [c.get("voice_score", compute_voice_score(
            c.get("audio_energy","unknown"), c.get("audio_stability","unknown"), c.get("audio_emotion","unknown")
        )) for c in history]
        face_scores  = [c.get("face_score", 0.5) for c in history]

        voice_norm = _safe_mean(voice_scores, 0.5)
        face_norm  = _safe_mean(face_scores,  0.5)

        # Check if guided session (has varied question types)
        qtypes = {c.get("question_type", "unknown") for c in history}
        is_guided = any(qt in VALID_QUESTION_TYPES and qt != "unknown" for qt in qtypes)

        per_question = aggregate_by_question(history)
        reactivity_norm = compute_reactivity(per_question) if is_guided else 0.0

        aggregated = compute_fusion_and_confidence(
            req.phq9_score, req.gad7_score, voice_norm, face_norm, reactivity_norm
        )

        audio_emotions  = [c.get("audio_emotion",  "") for c in history]
        facial_emotions = [c.get("facial_emotion", "") for c in history]
        dominant_audio  = _majority_vote(audio_emotions)
        dominant_facial = _majority_vote(facial_emotions)
        emotion_variability = round(_safe_std(voice_scores, 0.0), 4)

        report_inputs = {
            **aggregated,
            "per_question": per_question,
            "reactivity_score": reactivity_norm,
            "chunks_analyzed": chunks_analyzed,
            "dominant_audio_emotion":  dominant_audio,
            "dominant_facial_emotion": dominant_facial,
            "emotion_variability":     emotion_variability,
        }

        if is_guided:
            report = generate_llm_report_v2(report_inputs, per_question)
            return {"phq9_score": req.phq9_score, "gad7_score": req.gad7_score,
                    "aggregated": aggregated, "per_question": per_question,
                    "reactivity_score": reactivity_norm, "chunks_analyzed": chunks_analyzed,
                    "report": report}
        else:
            report_data = {
                **aggregated,
                "audio_emotion":   dominant_audio,
                "audio_energy":    _majority_vote([c.get("audio_energy","medium") for c in history]),
                "audio_stability": _majority_vote([c.get("audio_stability","moderate") for c in history]),
                "facial_emotion":  dominant_facial,
                "facial_confidence": _safe_mean([c.get("facial_confidence",0.0) for c in history]),
            }
            report = generate_llm_report(report_data)
            return {"phq9_score": req.phq9_score, "gad7_score": req.gad7_score,
                    "aggregated": aggregated, "chunks_analyzed": chunks_analyzed,
                    "report": report}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "detail": traceback.format_exc()})
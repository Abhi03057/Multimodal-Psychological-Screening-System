"""
Live Session Routes — Guided Multimodal Screening
===================================================
Endpoints:
  GET  /live/questions    → serve guided prompts to frontend
  POST /live/analyze      → per-chunk audio/video analysis (+ question_type)
  POST /live/end-session  → aggregation, reactivity, fusion, LLM report

Mathematical definitions are documented inline at each computation step.
"""

from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from collections import Counter
import shutil
import os
import uuid
import cv2
import numpy as np
import traceback

from services.audio_model import analyze_audio, reset_buffer
from services.face_model import analyze_single_frame, FACE_BEHAVIOR_MAP
from services.llm_service import generate_multimodal_report, generate_llm_report, generate_llm_report_v2
from config.questions import GUIDED_QUESTIONS, BASELINE_GROUP, STRESS_GROUP, VALID_QUESTION_TYPES

router = APIRouter(prefix="/live")


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS — Score Mappings
# ═══════════════════════════════════════════════════════════════════════════════

# Voice score component mappings (energy_norm, speech_instability)
ENERGY_NORM_MAP = {"low": 0.25, "medium": 0.50, "high": 0.80, "very low": 0.10, "unknown": 0.50}
STABILITY_NORM_MAP = {"stable": 0.15, "moderate": 0.45, "variable": 0.80, "unstable": 0.85, "none": 0.50, "unknown": 0.50}

# Audio emotion → stress contribution (pitch_variability proxy)
AUDIO_STRESS_MAP = {
    "angry": 0.85, "fear": 0.90, "sad": 0.70, "disgust": 0.65,
    "nervous": 0.80, "hesitant": 0.60, "surprise": 0.55,
    "happy": 0.20, "calm": 0.10, "neutral": 0.30, "confident": 0.20,
    "uncertain": 0.40, "unknown": 0.40, "no speech detected": 0.40,
}

# Facial emotion → distress weight for face_score
FACE_DISTRESS_MAP = {
    "happy": 0.10, "neutral": 0.30, "calm": 0.15,
    "surprise": 0.40, "sad": 0.70, "disgust": 0.75,
    "angry": 0.85, "fear": 0.90,
    "unknown": 0.40, "error": 0.40,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class ChunkResult(BaseModel):
    """Schema for a single analysis chunk in session history."""
    audio_emotion: str = "unknown"
    audio_energy: str = "unknown"
    audio_stability: str = "unknown"
    facial_emotion: str = "unknown"
    facial_confidence: float = 0.0
    risk_level: str = "low"
    # ── New fields (optional for backward compat) ──
    question_type: str = "unknown"
    voice_score: float = Field(default=0.0, ge=0.0, le=1.0)
    face_score: float = Field(default=0.0, ge=0.0, le=1.0)
    emotion_probs: Dict[str, float] = {}


class EndSessionRequest(BaseModel):
    """Schema for the end-session aggregation request."""
    phq9_score: int = 0
    gad7_score: int = 0
    history: List[ChunkResult] = []


# ═══════════════════════════════════════════════════════════════════════════════
#  SCORE COMPUTATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_voice_score(energy: str, stability: str, audio_emotion: str) -> float:
    """
    Compute a normalized voice stress score in [0, 1].

    Formula:
      voice_score = 0.4 * energy_norm + 0.3 * pitch_variability + 0.3 * speech_instability

    Where:
      energy_norm        = mapped from energy label (low/medium/high)
      pitch_variability  = mapped from audio emotion (stress proxy)
      speech_instability = mapped from stability label (stable/moderate/variable)
    """
    energy_norm = ENERGY_NORM_MAP.get(energy.lower(), 0.50)
    pitch_var = AUDIO_STRESS_MAP.get(audio_emotion.lower(), 0.40)
    instability = STABILITY_NORM_MAP.get(stability.lower(), 0.50)

    score = 0.4 * energy_norm + 0.3 * pitch_var + 0.3 * instability
    return round(max(0.0, min(1.0, score)), 4)


def compute_face_score(facial_emotion: str, confidence: float, emotion_probs: dict) -> float:
    """
    Compute a normalized facial distress score in [0, 1].

    If emotion_probs is available (from DeepFace):
      face_base = Σ (prob_i * weight_i) for each emotion
      face_score = face_base * confidence

    Otherwise (fallback):
      face_score = distress_weight(dominant_emotion) * max(confidence, 0.3)
    """
    if emotion_probs and len(emotion_probs) > 0:
        # Normalize probs to [0, 1] (DeepFace returns 0-100)
        total = sum(emotion_probs.values())
        if total > 1.5:  # Likely 0-100 scale
            probs = {k: v / 100.0 for k, v in emotion_probs.items()}
        else:
            probs = emotion_probs

        face_base = sum(
            probs.get(emo, 0.0) * FACE_DISTRESS_MAP.get(emo.lower(), 0.40)
            for emo in probs
        )
        face_score = face_base * max(confidence, 0.3)
    else:
        # Fallback: use dominant emotion only
        weight = FACE_DISTRESS_MAP.get(facial_emotion.lower(), 0.40)
        face_score = weight * max(confidence, 0.3)

    return round(max(0.0, min(1.0, face_score)), 4)


# ═══════════════════════════════════════════════════════════════════════════════
#  AGGREGATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_mean(values: list, default: float = 0.0) -> float:
    """Compute mean with division-by-zero protection."""
    return round(sum(values) / len(values), 4) if values else default


def _safe_std(values: list, default: float = 0.0) -> float:
    """Compute population standard deviation safely."""
    if len(values) < 2:
        return default
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return round(variance ** 0.5, 4)


def _majority_vote(items: list, default: str = "unknown") -> str:
    """Return the most common item, or default if list is empty."""
    filtered = [x for x in items if x and x != "unknown"]
    if not filtered:
        return default
    return Counter(filtered).most_common(1)[0][0]


def aggregate_by_question(chunks: List[ChunkResult]) -> dict:
    """
    Group chunks by question_type and compute per-group statistics.

    Returns:
      {
        "baseline": {"avg_voice": 0.35, "avg_face": 0.20, "avg_confidence": 0.72, "count": 3},
        "stress":   {"avg_voice": 0.68, "avg_face": 0.55, "avg_confidence": 0.81, "count": 4},
        ...
      }
    """
    groups: Dict[str, dict] = {}

    for chunk in chunks:
        qt = chunk.question_type if chunk.question_type in VALID_QUESTION_TYPES else "unknown"
        if qt not in groups:
            groups[qt] = {"voice_scores": [], "face_scores": [], "confidences": []}

        groups[qt]["voice_scores"].append(chunk.voice_score)
        groups[qt]["face_scores"].append(chunk.face_score)
        groups[qt]["confidences"].append(chunk.facial_confidence)

    result = {}
    for qt, data in groups.items():
        result[qt] = {
            "avg_voice": _safe_mean(data["voice_scores"]),
            "avg_face": _safe_mean(data["face_scores"]),
            "avg_confidence": _safe_mean(data["confidences"]),
            "count": len(data["voice_scores"]),
        }
    return result


def compute_reactivity(per_question: dict) -> float:
    """
    Compute emotional reactivity as the elevation from baseline to stress.

    Formula:
      baseline_score = mean(avg_voice + avg_face) for baseline_group
      stress_score   = mean(avg_voice + avg_face) for stress_group
      stress_variance = variance(stress_group combined scores)
      reactivity     = (stress_score - baseline_score) + stress_variance

    Edge cases:
      - No baseline → use global average as proxy
      - No stress group → reactivity = 0
    """
    # Collect baseline scores
    baseline_scores = []
    for qt in BASELINE_GROUP:
        if qt in per_question:
            g = per_question[qt]
            baseline_scores.append((g["avg_voice"] + g["avg_face"]) / 2.0)

    # Collect stress scores
    stress_scores = []
    stress_raw = []  # For variance computation
    for qt in STRESS_GROUP:
        if qt in per_question:
            g = per_question[qt]
            combined = (g["avg_voice"] + g["avg_face"]) / 2.0
            stress_scores.append(combined)
            stress_raw.append(g["avg_voice"])
            stress_raw.append(g["avg_face"])

    if not stress_scores:
        return 0.0

    stress_score = _safe_mean(stress_scores)

    if baseline_scores:
        baseline_score = _safe_mean(baseline_scores)
    else:
        # Fallback: use global average across ALL question types
        all_scores = []
        for g in per_question.values():
            all_scores.append((g["avg_voice"] + g["avg_face"]) / 2.0)
        baseline_score = _safe_mean(all_scores, 0.3)

    # Stress variance adds a penalty for emotional instability under stress
    stress_variance = _safe_std(stress_raw, 0.0) if len(stress_raw) >= 2 else 0.0

    reactivity = (stress_score - baseline_score) + stress_variance
    return round(max(0.0, min(1.0, reactivity)), 4)


def compute_fusion_and_confidence(
    phq9: int, gad7: int,
    voice_norm: float, face_norm: float,
    reactivity_norm: float
) -> dict:
    """
    Compute final fused risk score, confidence, and alignment.

    Fusion formula (weighted multimodal):
      FinalScore = 0.30*PHQ_norm + 0.25*GAD_norm + 0.20*Voice_norm
                 + 0.15*Face_norm + 0.10*Reactivity_norm

    Confidence (cross-modal consistency):
      values = [PHQ_norm, GAD_norm, Voice_norm, Face_norm, Reactivity_norm]
      consistency = std_dev(values)
      confidence  = 1 - consistency

    Alignment:
      consistency < 0.10 → "High"
      consistency < 0.25 → "Moderate"
      else                → "Low"

    Risk level:
      final_score > 0.65 or PHQ-9 > 14 or GAD-7 > 14 → "High"
      final_score > 0.40 or PHQ-9 > 8  or GAD-7 > 8  → "Moderate"
      else                                             → "Low"
    """
    phq_norm = min(phq9 / 27.0, 1.0)
    gad_norm = min(gad7 / 21.0, 1.0)

    # ── Weighted fusion ──
    final_score = round(
        0.30 * phq_norm
        + 0.25 * gad_norm
        + 0.20 * voice_norm
        + 0.15 * face_norm
        + 0.10 * reactivity_norm,
        4
    )
    final_score = max(0.0, min(1.0, final_score))

    # ── Confidence from cross-modal consistency ──
    values = [phq_norm, gad_norm, voice_norm, face_norm, reactivity_norm]
    consistency = _safe_std(values, 0.0)
    confidence = round(max(0.0, min(1.0, 1.0 - consistency)), 4)

    # ── Alignment level ──
    if consistency < 0.10:
        alignment = "High"
    elif consistency < 0.25:
        alignment = "Moderate"
    else:
        alignment = "Low"

    # ── Risk level ──
    if phq9 > 14 or gad7 > 14 or final_score > 0.65:
        risk_level = "High"
    elif phq9 > 8 or gad7 > 8 or final_score > 0.40:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    return {
        "phq9_score": phq9,
        "gad7_score": gad7,
        "phq9_norm": round(phq_norm, 4),
        "gad7_norm": round(gad_norm, 4),
        "voice_norm": round(voice_norm, 4),
        "face_norm": round(face_norm, 4),
        "reactivity_score": round(reactivity_norm, 4),
        "final_score": round(final_score, 4),
        "confidence_score": confidence,
        "alignment_level": alignment,
        "risk_level": risk_level,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT: GET /live/questions
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/questions")
async def get_questions():
    """Return the guided session prompts to the frontend."""
    return {"questions": GUIDED_QUESTIONS}


# ═══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT: POST /live/analyze
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/analyze")
async def live_analyze(
    phq9_score: int = Form(0),
    gad7_score: int = Form(0),
    audio_file: UploadFile = File(...),
    frame_file: UploadFile = File(None),
    question_type: str = Form("unknown"),  # NEW: guided question label
):
    try:
        reset_buffer()

        # ── 1. Audio analysis (unchanged) ──
        temp_audio = f"temp_{uuid.uuid4()}.wav"
        with open(temp_audio, "wb") as buf:
            shutil.copyfileobj(audio_file.file, buf)

        try:
            audio_result = analyze_audio(temp_audio)
        finally:
            if os.path.exists(temp_audio):
                os.remove(temp_audio)

        # ── 2. Facial analysis from uploaded frame (unchanged) ──
        face_emotion = "unknown"
        face_confidence = 0.0
        face_behavior = "no frame provided"
        emotion_probs = {}

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
                    emotion_probs = {k: round(float(v), 2) for k, v in scores.items()} if scores else {}
            except Exception as e:
                face_emotion = "error"
                face_behavior = str(e)

        # ── 3. Risk assessment (unchanged) ──
        if phq9_score > 14 or gad7_score > 14:
            risk = "high"
        elif phq9_score > 8:
            risk = "moderate"
        else:
            risk = "low"

        # ── 4. Compute derived scores (NEW) ──
        audio_emotion = audio_result.get("emotion", "unknown")
        audio_energy = audio_result.get("speech_energy", "unknown")
        audio_stability = audio_result.get("stability", "unknown")

        voice_score = compute_voice_score(audio_energy, audio_stability, audio_emotion)
        face_score = compute_face_score(face_emotion, face_confidence, emotion_probs)

        # Validate question_type
        qt = question_type if question_type in VALID_QUESTION_TYPES else "unknown"

        result = {
            "phq9_score": phq9_score,
            "gad7_score": gad7_score,
            "audio_emotion": audio_emotion,
            "audio_energy": audio_energy,
            "audio_stability": audio_stability,
            "audio_behavior": audio_result.get("behavior_flag", ""),
            "facial_emotion": face_emotion,
            "facial_confidence": face_confidence,
            "facial_behavior": face_behavior,
            "risk_level": risk,
            # ── New fields ──
            "question_type": qt,
            "voice_score": voice_score,
            "face_score": face_score,
            "emotion_probs": emotion_probs,
        }

        result["report"] = generate_multimodal_report(result)
        return result

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "detail": traceback.format_exc()},
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT: POST /live/end-session
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/end-session")
async def end_session(req: EndSessionRequest):
    """
    Aggregate accumulated live session data and generate a full LLM report.

    Logic path selection:
      - If any chunk has question_type != "unknown" → GUIDED path (new)
      - Otherwise → LEGACY path (backward compatible)
    """
    try:
        history = req.history

        # ── Determine if this is a guided session ──
        has_guided = any(h.question_type not in ("unknown", "") for h in history)

        if not history:
            # No chunks — questionnaire-only report
            report_inputs = compute_fusion_and_confidence(
                req.phq9_score, req.gad7_score,
                voice_norm=0.5, face_norm=0.5, reactivity_norm=0.0
            )
            report = generate_llm_report_v2(report_inputs, per_question={})
            return {
                "phq9_score": req.phq9_score,
                "gad7_score": req.gad7_score,
                "aggregated": {
                    "audio_emotion": "unknown",
                    "facial_emotion": "unknown",
                },
                "chunks_analyzed": 0,
                "report": report,
            }

        if has_guided:
            # ════════════════════════════════════════════
            #  GUIDED PATH: question-aware aggregation
            # ════════════════════════════════════════════

            # Step 1: Group by question_type
            per_question = aggregate_by_question(history)

            # Step 2: Compute global averages
            all_voice = [h.voice_score for h in history if h.voice_score > 0]
            all_face = [h.face_score for h in history if h.face_score > 0]
            voice_norm = _safe_mean(all_voice, 0.5)
            face_norm = _safe_mean(all_face, 0.5)

            # Step 3: Compute reactivity
            reactivity = compute_reactivity(per_question)

            # Step 4: Compute fusion, confidence, alignment
            report_inputs = compute_fusion_and_confidence(
                req.phq9_score, req.gad7_score,
                voice_norm, face_norm, reactivity
            )

            # Step 5: Gather extra context for LLM
            all_audio_emotions = [h.audio_emotion for h in history if h.audio_emotion != "unknown"]
            all_face_emotions = [h.facial_emotion for h in history if h.facial_emotion != "unknown"]
            dominant_audio = _majority_vote(all_audio_emotions)
            dominant_face = _majority_vote(all_face_emotions)

            # Emotion variability = std_dev of voice scores across chunks
            emotion_variability = _safe_std(all_voice, 0.0) if len(all_voice) >= 2 else 0.0

            report_inputs["dominant_audio_emotion"] = dominant_audio
            report_inputs["dominant_facial_emotion"] = dominant_face
            report_inputs["emotion_variability"] = round(emotion_variability, 4)

            report = generate_llm_report_v2(report_inputs, per_question)

            return {
                "phq9_score": req.phq9_score,
                "gad7_score": req.gad7_score,
                "aggregated": {
                    "audio_emotion": dominant_audio,
                    "facial_emotion": dominant_face,
                },
                "per_question": per_question,
                "reactivity_score": reactivity,
                "chunks_analyzed": len(history),
                "report": report,
            }

        else:
            # ════════════════════════════════════════════
            #  LEGACY PATH: backward compatible averaging
            # ════════════════════════════════════════════
            audio_emotions = [h.audio_emotion for h in history if h.audio_emotion != "unknown"]
            facial_emotions = [h.facial_emotion for h in history if h.facial_emotion != "unknown"]
            energies = [h.audio_energy for h in history if h.audio_energy != "unknown"]
            stabilities = [h.audio_stability for h in history if h.audio_stability != "unknown"]
            face_confs = [h.facial_confidence for h in history if h.facial_confidence > 0]

            session_data = {
                "phq9_score": req.phq9_score,
                "gad7_score": req.gad7_score,
                "audio_emotion": _majority_vote(audio_emotions),
                "audio_energy": _majority_vote(energies, "medium"),
                "audio_stability": _majority_vote(stabilities, "moderate"),
                "facial_emotion": _majority_vote(facial_emotions),
                "facial_confidence": _safe_mean(face_confs, 0.0),
            }

            report = generate_llm_report(session_data)

            return {
                "phq9_score": req.phq9_score,
                "gad7_score": req.gad7_score,
                "aggregated": session_data,
                "chunks_analyzed": len(history),
                "report": report,
            }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "detail": traceback.format_exc()},
        )

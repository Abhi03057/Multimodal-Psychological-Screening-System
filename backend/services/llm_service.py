import os
import requests
from dotenv import load_dotenv

# Load .env from project root (one level up from backend/)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

GROQ_API_KEY = os.getenv("GROK_API_KEY", "").strip()
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _build_system_prompt():
    return """You are an AI assistant generating a structured psychological screening report based strictly on precomputed multimodal inputs.

CRITICAL RULES:
* You are NOT a medical professional.
* DO NOT diagnose any condition.
* DO NOT override or reinterpret the provided risk score.
* DO NOT invent or assume missing data.
* ONLY interpret the structured inputs given.
* Use cautious, evidence-based language (e.g., "indications", "signals", "elevated risk").

OUTPUT REQUIREMENTS:
* Use clear section headings (with emoji prefixes)
* Keep language concise and professional
* Avoid repeating raw numbers excessively
* Ensure logical flow and consistency

EDGE CASE HANDLING:
* If behavioral data is weak or inconsistent, explicitly mention reduced confidence.
* If questionnaire scores dominate, state that conclusions rely primarily on self-reported data.
* If alignment_level is low, highlight conflicting signals."""


def _build_user_prompt(data: dict) -> str:
    return f"""INPUT DATA:
* PHQ-9 Score: {data['phq9_score']} (normalized: {data['phq9_norm']:.2f})
* GAD-7 Score: {data['gad7_score']} (normalized: {data['gad7_norm']:.2f})
* Voice Stress Score: {data['voice_norm']:.2f} (0-1, aggregated from pitch, energy, and speech patterns)
* Facial Emotion Score: {data['face_norm']:.2f} (0-1, weighted across detected emotions)
* Final Risk Score: {data['final_score']:.2f} (precomputed using weighted multimodal fusion)
* Risk Level: {data['risk_level']}
* Confidence Score: {data['confidence_score']:.2f} (derived from cross-modal consistency)
* Alignment Level: {data['alignment_level']}

TASK:
Generate a professional psychological screening report with the following 7 sections:

1. Summary:
* Clearly state the overall risk level.
* Briefly explain what the final score indicates.
* Mention which components (questionnaire vs behavioral) contributed most.

2. Questionnaire Insights:
* Interpret PHQ-9 and GAD-7 scores individually.
* Mention severity levels.
* Avoid any diagnostic claims.

3. Multimodal Behavioral Analysis:
* Interpret voice stress score (e.g., signs of tension, instability, calmness).
* Interpret facial emotion score (emotional valence and intensity).
* Explain whether behavioral signals reinforce or contradict questionnaire results.

4. Consistency & Confidence Analysis:
* Use the alignment_level to explain agreement or mismatch between modalities.
* Interpret the confidence_score:
  * High → strong agreement between signals
  * Low → conflicting or uncertain signals
* Explicitly mention if inconsistencies reduce reliability.

5. Risk Interpretation:
* Explain WHY the system assigned this risk level.
* Reference dominant contributing factors (e.g., high PHQ-9, elevated voice stress).
* Do NOT change or question the provided risk level.

6. Recommendations:
* Based on risk level:
  * Low → general well-being suggestions
  * Moderate → monitoring and self-care strategies
  * High → recommend consulting a licensed professional
* Keep suggestions realistic and non-alarmist.

7. Disclaimer:
* Clearly state that this is a screening tool and not a clinical diagnosis."""


def _compute_report_inputs(data: dict) -> dict:
    """
    Take raw session data and compute normalized scores, risk, confidence, alignment.
    """
    phq9 = data.get("phq9_score", 0)
    gad7 = data.get("gad7_score", 0)

    phq9_norm = min(phq9 / 27.0, 1.0)
    gad7_norm = min(gad7 / 21.0, 1.0)

    # --- Voice stress: map audio features to 0-1 ---
    energy_map = {"low": 0.3, "medium": 0.5, "high": 0.8, "unknown": 0.5}
    stability_map = {"stable": 0.2, "moderate": 0.5, "unstable": 0.8, "unknown": 0.5}
    audio_energy = data.get("audio_energy", "unknown")
    audio_stability = data.get("audio_stability", "unknown")
    audio_emotion = data.get("audio_emotion", "neutral")

    # Emotion-based stress contribution
    stress_emotions = {"angry": 0.8, "fear": 0.9, "sad": 0.7, "disgust": 0.6,
                       "surprise": 0.5, "nervous": 0.8, "hesitant": 0.6}
    calm_emotions = {"happy": 0.2, "calm": 0.1, "neutral": 0.3, "confident": 0.2}
    emotion_stress = stress_emotions.get(audio_emotion, calm_emotions.get(audio_emotion, 0.4))

    voice_norm = round(
        0.3 * energy_map.get(audio_energy, 0.5)
        + 0.3 * stability_map.get(audio_stability, 0.5)
        + 0.4 * emotion_stress,
        2
    )

    # --- Facial emotion score: negative emotions → higher score ---
    facial_emotion = data.get("facial_emotion", "unknown")
    facial_confidence = data.get("facial_confidence", 0.0)
    face_emotion_map = {
        "angry": 0.85, "fear": 0.9, "sad": 0.75, "disgust": 0.7,
        "surprise": 0.4, "happy": 0.1, "neutral": 0.3, "calm": 0.15,
        "unknown": 0.5, "error": 0.5
    }
    face_raw = face_emotion_map.get(facial_emotion, 0.5)
    face_norm = round(face_raw * max(facial_confidence, 0.3), 2)

    # --- Final risk score (weighted fusion) ---
    questionnaire_score = 0.5 * phq9_norm + 0.5 * gad7_norm
    behavioral_score = 0.5 * voice_norm + 0.5 * face_norm
    final_score = round(0.5 * questionnaire_score + 0.5 * behavioral_score, 2)

    # --- Risk level ---
    if phq9 > 14 or gad7 > 14 or final_score > 0.65:
        risk_level = "High"
    elif phq9 > 8 or gad7 > 8 or final_score > 0.4:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    # --- Cross-modal alignment ---
    q_risk = questionnaire_score
    b_risk = behavioral_score
    diff = abs(q_risk - b_risk)
    if diff < 0.15:
        alignment_level = "High"
    elif diff < 0.35:
        alignment_level = "Moderate"
    else:
        alignment_level = "Low"

    # --- Confidence score ---
    confidence_score = round(1.0 - diff, 2)
    confidence_score = max(0.0, min(1.0, confidence_score))

    return {
        "phq9_score": phq9,
        "gad7_score": gad7,
        "phq9_norm": phq9_norm,
        "gad7_norm": gad7_norm,
        "voice_norm": voice_norm,
        "face_norm": face_norm,
        "final_score": final_score,
        "risk_level": risk_level,
        "confidence_score": confidence_score,
        "alignment_level": alignment_level,
    }


def generate_llm_report(session_data: dict) -> str:
    """
    Generate a full screening report using Groq LLM.
    Falls back to template if API fails.
    """
    report_inputs = _compute_report_inputs(session_data)

    if not GROQ_API_KEY:
        return _fallback_report(report_inputs, session_data)

    try:
        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": _build_system_prompt()},
                    {"role": "user", "content": _build_user_prompt(report_inputs)},
                ],
                "temperature": 0.4,
                "max_tokens": 2000,
            },
            timeout=30,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return content.strip()

    except Exception as e:
        print(f"[LLM] Groq API error: {e}")
        return _fallback_report(report_inputs, session_data)


# ═══════════════════════════════════════════════════════════════════════════════
#  V2: GUIDED SESSION REPORT (pre-computed inputs — NO internal recomputation)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_system_prompt_v2():
    return """You are an AI assistant generating a structured psychological screening report based strictly on precomputed multimodal inputs.

CRITICAL RULES:
* You are NOT a medical professional.
* DO NOT diagnose any condition.
* DO NOT override or reinterpret the provided scores — they are precomputed.
* DO NOT invent or assume missing data.
* ONLY interpret the structured inputs given.
* Use cautious, evidence-based language (e.g., "indications", "signals", "elevated risk").

OUTPUT REQUIREMENTS:
* Use clear section headings (with emoji prefixes)
* Keep language concise and professional
* Avoid repeating raw numbers excessively
* Ensure logical flow and consistency
* When reactivity data is available, interpret emotional shifts between baseline and stress conditions

EDGE CASE HANDLING:
* If behavioral data is weak or inconsistent, explicitly mention reduced confidence.
* If questionnaire scores dominate, state that conclusions rely primarily on self-reported data.
* If alignment_level is low, highlight conflicting signals.
* If reactivity_score is near 0, note stable emotional responses across conditions."""


def _build_user_prompt_v2(data: dict, per_question: dict) -> str:
    # Build per-question summary text
    pq_text = ""
    if per_question:
        pq_lines = []
        for qt, stats in per_question.items():
            pq_lines.append(
                f"  - {qt}: voice_avg={stats['avg_voice']:.2f}, "
                f"face_avg={stats['avg_face']:.2f}, "
                f"confidence_avg={stats['avg_confidence']:.2f}, "
                f"segments={stats['count']}"
            )
        pq_text = "\n".join(pq_lines)
    else:
        pq_text = "  (no per-question data available)"

    return f"""INPUT DATA:
* PHQ-9 Score: {data['phq9_score']} (normalized: {data['phq9_norm']:.2f})
* GAD-7 Score: {data['gad7_score']} (normalized: {data['gad7_norm']:.2f})
* Voice Stress Score: {data['voice_norm']:.2f} (0–1, aggregated from energy, pitch variability, and speech instability)
* Facial Emotion Score: {data['face_norm']:.2f} (0–1, weighted across detected emotion probabilities)
* Reactivity Score: {data['reactivity_score']:.2f} (0–1, elevation from baseline to stress conditions)
* Final Risk Score: {data['final_score']:.2f} (precomputed using weighted multimodal fusion)
* Risk Level: {data['risk_level']}
* Confidence Score: {data['confidence_score']:.2f} (derived from cross-modal consistency)
* Alignment Level: {data['alignment_level']}
* Dominant Audio Emotion: {data.get('dominant_audio_emotion', 'unknown')}
* Dominant Facial Emotion: {data.get('dominant_facial_emotion', 'unknown')}
* Emotion Variability: {data.get('emotion_variability', 0):.2f} (std dev of voice scores across session)

Per-Question Breakdown:
{pq_text}

TASK:
Generate a professional psychological screening report with the following 8 sections:

1. 📊 Summary:
* State the overall risk level and what the final score indicates.
* Mention which components (questionnaire vs behavioral vs reactivity) contributed most.

2. 📋 Questionnaire Insights:
* Interpret PHQ-9 and GAD-7 scores individually with severity levels.
* Avoid any diagnostic claims.

3. 🎤 Multimodal Behavioral Analysis:
* Interpret voice stress score (tension, instability, or calmness).
* Interpret facial emotion score (emotional valence and intensity).
* Note whether behavioral signals reinforce or contradict questionnaire results.

4. ⚡ Emotional Reactivity:
* Interpret the reactivity score — how much emotional state shifted from baseline to stress prompts.
* Low reactivity = emotional stability or suppression.
* High reactivity = heightened emotional response to stressors.
* Reference per-question data if available.

5. 🔗 Consistency & Confidence Analysis:
* Use alignment_level and confidence_score to explain agreement/mismatch.
* High confidence = strong cross-modal agreement.
* Low confidence = conflicting or uncertain signals.

6. ⚠️ Risk Interpretation:
* Explain WHY the system assigned this risk level.
* Reference dominant contributing factors.
* Do NOT change or question the provided risk level.

7. 💡 Recommendations:
* Low risk → general well-being suggestions
* Moderate risk → monitoring and self-care strategies
* High risk → recommend consulting a licensed professional

8. ⚕️ Disclaimer:
* Clearly state this is a screening tool, not a clinical diagnosis."""


def generate_llm_report_v2(report_inputs: dict, per_question: dict) -> str:
    """
    Generate a report using Groq LLM with pre-computed guided session inputs.
    Uses the enriched v2 prompt that includes reactivity and per-question data.
    Falls back to template if API fails.
    """
    if not GROQ_API_KEY:
        return _fallback_report_v2(report_inputs, per_question)

    try:
        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": _build_system_prompt_v2()},
                    {"role": "user", "content": _build_user_prompt_v2(report_inputs, per_question)},
                ],
                "temperature": 0.4,
                "max_tokens": 2500,
            },
            timeout=30,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return content.strip()

    except Exception as e:
        print(f"[LLM-v2] Groq API error: {e}")
        return _fallback_report_v2(report_inputs, per_question)


def _fallback_report_v2(inputs: dict, per_question: dict) -> str:
    """Template-based fallback for guided session reports."""
    phq_sev = (
        "minimal" if inputs["phq9_score"] <= 4
        else "mild" if inputs["phq9_score"] <= 9
        else "moderate" if inputs["phq9_score"] <= 14
        else "moderately severe" if inputs["phq9_score"] <= 19
        else "severe"
    )
    gad_sev = (
        "minimal" if inputs["gad7_score"] <= 4
        else "mild" if inputs["gad7_score"] <= 9
        else "moderate" if inputs["gad7_score"] <= 14
        else "severe"
    )

    audio_emo = inputs.get("dominant_audio_emotion", "unknown")
    face_emo = inputs.get("dominant_facial_emotion", "unknown")
    congruence = (
        f"Both audio and facial channels indicate '{audio_emo}' — signals are congruent."
        if audio_emo == face_emo
        else f"Audio detected '{audio_emo}' while facial expression shows '{face_emo}' — signals show divergence."
    )

    reactivity = inputs.get("reactivity_score", 0)
    react_text = (
        "Emotional responses remained relatively stable across baseline and stress conditions, suggesting emotional steadiness or potential suppression."
        if reactivity < 0.2
        else "Moderate emotional shift was observed between baseline and stress conditions, indicating some sensitivity to stressors."
        if reactivity < 0.5
        else "Significant emotional elevation was detected from baseline to stress conditions, suggesting heightened emotional reactivity to stressors."
    )

    rec = (
        "General wellness: maintain regular sleep, exercise, and social connections."
        if inputs["risk_level"] == "Low"
        else "Consider self-monitoring and stress management techniques. If symptoms persist, consider speaking with a counselor."
        if inputs["risk_level"] == "Moderate"
        else "Elevated risk indicators are present. It is recommended to consult a licensed mental health professional for a clinical evaluation."
    )

    return f"""📊 Summary
The overall risk level is {inputs['risk_level']}. The final multimodal score is {inputs['final_score']:.2f}, derived from questionnaire responses, behavioral signals, and emotional reactivity analysis. Cross-modal alignment is {inputs['alignment_level']} with {inputs['confidence_score']:.2f} confidence.

📋 Questionnaire Insights
PHQ-9 Score: {inputs['phq9_score']} (normalized: {inputs['phq9_norm']:.2f}) — {phq_sev} severity.
GAD-7 Score: {inputs['gad7_score']} (normalized: {inputs['gad7_norm']:.2f}) — {gad_sev} severity.

🎤 Multimodal Behavioral Analysis
Voice Stress: {inputs['voice_norm']:.2f} — Dominant audio emotion: '{audio_emo}'.
Facial Emotion: {inputs['face_norm']:.2f} — Dominant expression: '{face_emo}'.
{congruence}

⚡ Emotional Reactivity
Reactivity Score: {reactivity:.2f}
{react_text}

🔗 Consistency & Confidence
Alignment: {inputs['alignment_level']} | Confidence: {inputs['confidence_score']:.2f}
{'Signals across modalities are consistent, supporting reliability.' if inputs['alignment_level'] == 'High' else 'Some divergence between modalities — interpret with caution.' if inputs['alignment_level'] == 'Moderate' else 'Significant divergence detected — reduced confidence in assessment.'}

⚠️ Risk Interpretation
Risk level: {inputs['risk_level']} (score: {inputs['final_score']:.2f}).
{'Self-reported questionnaire scores are the primary contributors.' if inputs['phq9_norm'] + inputs['gad7_norm'] > inputs['voice_norm'] + inputs['face_norm'] else 'Behavioral signals contribute significantly to the risk assessment.'}

💡 Recommendations
{rec}

⚕️ Disclaimer
This is an automated screening tool and does not constitute a medical or clinical diagnosis. Results should be interpreted by qualified professionals. If you are in crisis, contact emergency services or a crisis helpline immediately."""


def _fallback_report(inputs: dict, raw: dict) -> str:
    """Template-based fallback if LLM is unavailable."""
    audio_emotion = raw.get("audio_emotion", "unknown")
    facial_emotion = raw.get("facial_emotion", "unknown")

    if audio_emotion == facial_emotion:
        congruence = f"Both audio and facial channels indicate '{audio_emotion}' — signals are congruent."
    else:
        congruence = f"Audio detected '{audio_emotion}' while facial expression shows '{facial_emotion}' — signals show some divergence."

    return f"""📊 Summary
The overall risk level is {inputs['risk_level']}. The final multimodal score is {inputs['final_score']:.2f}, derived from questionnaire responses and behavioral signals. Cross-modal alignment is {inputs['alignment_level']}.

📋 Questionnaire Insights
PHQ-9 Score: {inputs['phq9_score']} (normalized: {inputs['phq9_norm']:.2f}) — {'minimal' if inputs['phq9_score'] <= 4 else 'mild' if inputs['phq9_score'] <= 9 else 'moderate' if inputs['phq9_score'] <= 14 else 'moderately severe' if inputs['phq9_score'] <= 19 else 'severe'} severity.
GAD-7 Score: {inputs['gad7_score']} (normalized: {inputs['gad7_norm']:.2f}) — {'minimal' if inputs['gad7_score'] <= 4 else 'mild' if inputs['gad7_score'] <= 9 else 'moderate' if inputs['gad7_score'] <= 14 else 'severe'} severity.

🎤 Multimodal Behavioral Analysis
Voice Stress: {inputs['voice_norm']:.2f} — Audio emotion detected as '{audio_emotion}' with {raw.get('audio_energy', 'unknown')} energy and {raw.get('audio_stability', 'unknown')} stability.
Facial Emotion: {inputs['face_norm']:.2f} — Detected '{facial_emotion}' with {raw.get('facial_confidence', 0):.0%} confidence.
{congruence}

🔗 Consistency & Confidence
Alignment: {inputs['alignment_level']} | Confidence: {inputs['confidence_score']:.2f}
{'Signals across modalities are consistent, supporting reliability.' if inputs['alignment_level'] == 'High' else 'Some divergence between modalities — interpret with caution.' if inputs['alignment_level'] == 'Moderate' else 'Significant divergence detected — reduced confidence in assessment.'}

⚠️ Risk Interpretation
Risk level: {inputs['risk_level']} (score: {inputs['final_score']:.2f}).
{'Self-reported questionnaire scores are the primary contributors.' if inputs['phq9_norm'] + inputs['gad7_norm'] > inputs['voice_norm'] + inputs['face_norm'] else 'Behavioral signals contribute significantly to the risk assessment.'}

💡 Recommendations
{'General wellness: maintain regular sleep, exercise, and social connections. Continue practicing healthy coping strategies.' if inputs['risk_level'] == 'Low' else 'Consider self-monitoring and stress management techniques. If symptoms persist, consider speaking with a counselor.' if inputs['risk_level'] == 'Moderate' else 'Elevated risk indicators are present. It is recommended to consult a licensed mental health professional for a clinical evaluation.'}

⚕️ Disclaimer
This is an automated screening tool and does not constitute a medical or clinical diagnosis. Results should be interpreted by qualified professionals. If you are in crisis, contact emergency services or a crisis helpline immediately."""


# ── Legacy functions (kept for backward compatibility) ──

def generate_report(data):
    phq9 = data["phq9_score"]
    risk = data["risk_level"]
    energy = data["audio_features"]["energy"]
    pause = data["audio_features"]["pause_rate"]
    emotion = data["facial_emotion"]

    return f"""🧠 Psychological Screening Report

-------------------------------------

📊 Assessment Summary:
- PHQ-9 Score: {phq9}
- Risk Level: {risk.upper()}

-------------------------------------

🎤 Behavioral Observations:
- Vocal Energy: {energy}
- Pause Pattern: {pause}
- Facial Expression: {emotion}

-------------------------------------

🧾 Interpretation:
The questionnaire responses indicate {risk} psychological risk.
Speech patterns show {energy} energy with {pause} pauses.
Facial expression suggests emotional tone consistent with {emotion} affect.

-------------------------------------

⚠️ Recommendation:
This is a non-clinical screening result.
If symptoms persist, consider consulting a qualified mental health professional.
"""


def generate_multimodal_report(data):
    """Called during live chunks — uses template for speed, LLM for end-session."""
    phq9 = data.get("phq9_score", 0)
    gad7 = data.get("gad7_score", 0)
    risk = data.get("risk_level", "unknown")
    audio_emotion = data.get("audio_emotion", "unknown")
    audio_energy = data.get("audio_energy", "unknown")
    audio_stability = data.get("audio_stability", "unknown")
    facial_emotion = data.get("facial_emotion", "unknown")
    facial_confidence = data.get("facial_confidence", 0.0)

    if audio_emotion == facial_emotion:
        congruence = f"Both channels indicate '{audio_emotion}' — congruent."
    else:
        congruence = f"Audio: '{audio_emotion}' vs Face: '{facial_emotion}' — incongruent."

    return f"""📊 PHQ-9: {phq9} | GAD-7: {gad7} | Risk: {risk.upper()}
🎤 Audio: {audio_emotion.upper()} | Energy: {audio_energy} | Stability: {audio_stability}
📷 Face: {facial_emotion.upper()} | Confidence: {facial_confidence:.0%}
🔗 {congruence}"""
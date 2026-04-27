import os
import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

GROQ_API_KEY = os.getenv("GROK_API_KEY", "").strip()
GROQ_MODEL   = "llama-3.3-70b-versatile"
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"


def _severity(score):
    for t, label in ((4,"minimal"),(9,"mild"),(14,"moderate"),(19,"moderately severe")):
        if score <= t:
            return label
    return "severe"


def _build_system_prompt():
    return (
        "You are an AI assistant generating a structured psychological screening report "
        "based strictly on precomputed multimodal inputs.\n\nCRITICAL RULES:\n"
        "* Do not diagnose or prescribe treatment.\n"
        "* Base all statements on the provided numerical scores.\n"
        "* Use professional, empathetic language."
    )


def _compute_report_inputs(data):
    phq9 = data.get("phq9_score", 0)
    gad7 = data.get("gad7_score", 0)
    phq9_norm = round(min(phq9 / 27.0, 1.0), 4)
    gad7_norm = round(min(gad7 / 21.0, 1.0), 4)
    audio_energy    = data.get("audio_energy",    "unknown")
    audio_stability = data.get("audio_stability", "unknown")
    audio_emotion   = data.get("audio_emotion",   "neutral")
    energy_map    = {"low": 0.2, "medium": 0.5, "high": 0.8, "unknown": 0.3}
    stability_map = {"stable": 0.1, "moderate": 0.5, "variable": 0.9, "unknown": 0.5}
    stress_emotions = {"angry","fear","sad","disgust","surprise","nervous","hesitant"}
    calm_emotions   = {"happy","calm","neutral","confident"}
    e_score = energy_map.get(audio_energy, 0.3)
    s_score = stability_map.get(audio_stability, 0.5)
    a_score = 0.9 if audio_emotion in stress_emotions else (0.1 if audio_emotion in calm_emotions else 0.4)
    voice_norm = round(min(0.4*e_score + 0.3*s_score + 0.3*a_score, 1.0), 4)
    facial_emotion    = data.get("facial_emotion", "unknown")
    facial_confidence = float(data.get("facial_confidence", 0.0))
    face_distress = {"angry":0.85,"fear":0.9,"sad":0.75,"disgust":0.7,"surprise":0.4,
                     "happy":0.05,"neutral":0.15,"calm":0.1,"unknown":0.3,"error":0.3}
    face_base = face_distress.get(facial_emotion, 0.3)
    face_norm = round(min(0.85*face_base + 0.15*(1 - min(facial_confidence,1.0)), 1.0), 4)
    final_score = round(0.3*phq9_norm + 0.2*gad7_norm + 0.3*voice_norm + 0.15*face_norm + 0.05*abs(phq9_norm-gad7_norm), 4)
    if phq9 >= 14 or final_score >= 0.65:
        risk_level = "High"
    elif phq9 >= 8 or final_score >= 0.4:
        risk_level = "Moderate"
    else:
        risk_level = "Low"
    modal_diff = abs(voice_norm - face_norm)
    confidence_score = round(max(1.0 - modal_diff*2, 0.0), 2)
    alignment_level = "High" if confidence_score >= 0.75 else ("Moderate" if confidence_score >= 0.35 else "Low")
    return {"phq9_score":phq9,"gad7_score":gad7,"phq9_norm":phq9_norm,"gad7_norm":gad7_norm,
            "voice_norm":voice_norm,"face_norm":face_norm,"final_score":final_score,
            "risk_level":risk_level,"confidence_score":confidence_score,"alignment_level":alignment_level}


def _build_user_prompt(data):
    return (
        f"INPUT DATA:\nPHQ-9: {data['phq9_score']} (norm {data['phq9_norm']:.2f})\n"
        f"GAD-7: {data['gad7_score']} (norm {data['gad7_norm']:.2f})\n"
        f"Voice Stress: {data['voice_norm']}\nFacial Score: {data['face_norm']}\n"
        f"Final Risk Score: {data['final_score']}\nRisk Level: {data['risk_level']}\n"
        f"Confidence: {data['confidence_score']}\nAlignment: {data['alignment_level']}\n\n"
        "TASK: Generate a 7-section professional psychological screening report."
    )


def _build_user_prompt_v2(data, per_question):
    lines = []
    if per_question:
        for qt, s in per_question.items():
            lines.append(f"  {qt}: voice={s['avg_voice']:.2f} face={s['avg_face']:.2f} n={s['count']}")
    pq = "\n".join(lines) or "  (none)"
    return (
        f"PHQ-9={data.get('phq9_score',0)} GAD-7={data.get('gad7_score',0)} "
        f"Risk={data.get('risk_level')} Score={data.get('final_score',0):.2f}\n"
        f"Per-question:\n{pq}\n\nTASK: Generate an 8-section professional screening report."
    )


def _fallback_report(inputs, raw):
    phq9  = inputs.get("phq9_score", 0)
    gad7  = inputs.get("gad7_score", 0)
    risk  = inputs.get("risk_level", "Low")
    final = inputs.get("final_score", 0)
    align = inputs.get("alignment_level", "Moderate")
    conf  = inputs.get("confidence_score", 0)
    a_emo = inputs.get("audio_emotion", "unknown")
    f_emo = inputs.get("facial_emotion", "unknown")
    a_en  = inputs.get("audio_energy", "unknown")
    a_st  = inputs.get("audio_stability", "unknown")
    f_conf = float(inputs.get("facial_confidence", 0))
    cross = (f"Both channels: '{a_emo}' - congruent." if a_emo == f_emo
             else f"Audio: '{a_emo}' vs Face: '{f_emo}' - divergent.")
    rec = {"Low":"Maintain regular sleep, exercise, and social connections.",
           "Moderate":"Consider stress management techniques."
           }.get(risk, "Consult a licensed mental health professional.")
    return (
        f"Summary: Risk {risk}. Score: {final:.2f}. Alignment: {align}.\n\n"
        f"PHQ-9: {phq9} ({_severity(phq9)}). GAD-7: {gad7} ({_severity(gad7)}).\n\n"
        f"Behavioral: {cross} Energy: {a_en}, stability: {a_st}. Face: {f_conf:.0%}.\n\n"
        f"Recommendations: {rec}\n\n"
        f"Disclaimer: Automated screening only - not a clinical diagnosis."
    )


def _fallback_report_v2(inputs, per_question):
    phq9       = inputs.get("phq9_score", 0)
    gad7       = inputs.get("gad7_score", 0)
    audio_dom  = inputs.get("dominant_audio_emotion", "unknown")
    face_dom   = inputs.get("dominant_facial_emotion", "unknown")
    reactivity = inputs.get("reactivity_score", 0)
    risk       = inputs.get("risk_level", "Low")
    final      = inputs.get("final_score", 0)
    cross = (f"Both: '{audio_dom}' - congruent." if audio_dom == face_dom
             else f"Audio '{audio_dom}' vs Face '{face_dom}'.")
    react_note = ("Stable." if reactivity < 0.2 else
                  "Moderate shift." if reactivity < 0.5 else "Significant elevation.")
    rec = {"Low":"Maintain healthy habits.","Moderate":"Consider counseling."
           }.get(risk, "Consult a mental health professional.")
    return (
        f"Risk: {risk}. Score: {final:.2f}.\n"
        f"PHQ-9: {phq9} ({_severity(phq9)}). GAD-7: {gad7} ({_severity(gad7)}).\n"
        f"{cross}\nReactivity: {reactivity:.2f}. {react_note}\n{rec}\n"
        f"Disclaimer: Automated screening only."
    )


def generate_llm_report(session_data):
    inputs = _compute_report_inputs(session_data)
    if not GROQ_API_KEY:
        return _fallback_report(inputs, None)
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": GROQ_MODEL,
               "messages": [{"role":"system","content":_build_system_prompt()},
                             {"role":"user","content":_build_user_prompt(inputs)}],
               "temperature": 0.4, "max_tokens": 2000}
    try:
        r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[LLM] Groq API error: {e}")
        return _fallback_report(inputs, None)


def generate_llm_report_v2(report_inputs, per_question):
    if not GROQ_API_KEY:
        return _fallback_report_v2(report_inputs, per_question)
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": GROQ_MODEL,
               "messages": [{"role":"system","content":_build_system_prompt()},
                             {"role":"user","content":_build_user_prompt_v2(report_inputs, per_question)}],
               "temperature": 0.4, "max_tokens": 2500}
    try:
        r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[LLM-v2] Groq API error: {e}")
        return _fallback_report_v2(report_inputs, per_question)


def generate_report(data):
    risk  = data.get("risk_level", "unknown").upper()
    phq9  = data.get("phq9_score", 0)
    feats = data.get("audio_features", {})
    energy = feats.get("energy", "unknown")
    pause  = feats.get("pause_rate", "unknown")
    facial = data.get("facial_emotion", "unknown")
    return (
        f"Screening Report\nPHQ-9: {phq9} | Risk: {risk}\n"
        f"Energy: {energy} | Pause: {pause} | Face: {facial}\n"
        f"Non-clinical result. Consult a professional if symptoms persist."
    )


def generate_multimodal_report(data):
    """Called during live chunks - template for speed, LLM for end-session."""
    phq9  = data.get("phq9_score", 0)
    gad7  = data.get("gad7_score", 0)
    risk  = data.get("risk_level", "unknown")
    a_emo = data.get("audio_emotion", "unknown")
    a_en  = data.get("audio_energy",  "unknown")
    a_st  = data.get("audio_stability", "unknown")
    f_emo = data.get("facial_emotion", "unknown")
    f_conf = float(data.get("facial_confidence", 0.0))
    cross = (f"Both: '{a_emo}' - congruent." if a_emo == f_emo
             else f"Audio: '{a_emo}' vs Face: '{f_emo}' - incongruent.")
    return (
        f"PHQ-9: {phq9} | GAD-7: {gad7} | Risk: {risk}\n"
        f"Audio: {a_emo} | Energy: {a_en} | Stability: {a_st}\n"
        f"Face: {f_emo} | Confidence: {f_conf:.0%}\n{cross}"
    )
def generate_report(data):
    phq9 = data["phq9_score"]
    risk = data["risk_level"]
    energy = data["audio_features"]["energy"]
    pause = data["audio_features"]["pause_rate"]
    emotion = data["facial_emotion"]

    return f"""
🧠 Psychological Screening Report

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
Speech patterns show {energy} energy with {pause} pauses, which may reflect communication style and cognitive load.
Facial expression suggests emotional tone consistent with {emotion} affect.

-------------------------------------

⚠️ Recommendation:
This is a non-clinical screening result. 
If symptoms persist, consider consulting a qualified mental health professional.
"""


def generate_multimodal_report(data):
    phq9 = data["phq9_score"]
    gad7 = data["gad7_score"]
    risk = data["risk_level"]
    audio_emotion = data.get("audio_emotion", "unknown")
    audio_energy = data.get("audio_energy", "unknown")
    audio_stability = data.get("audio_stability", "unknown")
    audio_behavior = data.get("audio_behavior", "")
    facial_emotion = data.get("facial_emotion", "unknown")
    facial_confidence = data.get("facial_confidence", 0.0)
    facial_behavior = data.get("facial_behavior", "")

    # Congruence check
    if audio_emotion == facial_emotion:
        congruence = f"Both audio and facial channels indicate '{audio_emotion}' — signals are congruent."
    else:
        congruence = f"Audio detected '{audio_emotion}' while facial expression shows '{facial_emotion}' — signals are incongruent, which may indicate masking or mixed emotions."

    return f"""
🧠 Multimodal Psychological Screening Report

═════════════════════════════════════════════

📊 Assessment Summary:
- PHQ-9 Score: {phq9}
- GAD-7 Score: {gad7}
- Risk Level: {risk.upper()}

═════════════════════════════════════════════

🎤 Audio Analysis:
- Detected Emotion: {audio_emotion.upper()}
- Vocal Energy: {audio_energy}
- Speech Stability: {audio_stability}
- Signal: {audio_behavior}

═════════════════════════════════════════════

📷 Facial Analysis:
- Detected Emotion: {facial_emotion.upper()}
- Confidence: {facial_confidence:.0%}
- Signal: {facial_behavior}

═════════════════════════════════════════════

🔗 Cross-Modal Analysis:
{congruence}

═════════════════════════════════════════════

🧾 Interpretation:
The questionnaire responses indicate {risk} psychological risk.
Speech analysis reveals {audio_energy} vocal energy with {audio_stability} stability, suggesting {audio_behavior}.
Facial expression analysis indicates {facial_behavior} with {facial_confidence:.0%} confidence.

═════════════════════════════════════════════

⚠️ Recommendation:
This is a non-clinical screening result.
If symptoms persist, consider consulting a qualified mental health professional.
"""
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
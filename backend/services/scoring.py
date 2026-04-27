import random

def analyze_data():
    risk = random.choice(("low", "medium", "high"))
    energy = random.choice(("low", "medium", "high"))
    pause  = random.choice(("low", "medium", "high"))
    emotion = random.choice(("neutral", "sad", "happy"))
    return {
        "audio_features": {"energy": energy, "pause_rate": pause},
        "facial_emotion": emotion,
    }

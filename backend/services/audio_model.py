import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Load model ONCE (important)
model = load_model("models/audio_model.keras", compile=False)

# Label mapping (must match training)
label_map = {
    0: "happy",
    1: "sad",
    2: "angry",
    3: "neutral",
    4: "fear"
}

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3)

    # Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel)
    mel_db = np.resize(mel_db, (128, 128))

    # Behavioral features
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    silence = np.sum(y == 0) / len(y)

    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches)

    return mel_db, [rms, zcr, pitch, silence]


def predict_audio(file_path):
    mel, feat = extract_features(file_path)

    mel = np.array(mel)[np.newaxis, ..., np.newaxis]
    feat = np.array(feat)[np.newaxis, ...]

    prediction = model.predict([mel, feat])
    label = label_map[np.argmax(prediction)]

    return {
        "emotion": label,
        "confidence": float(np.max(prediction))
    }
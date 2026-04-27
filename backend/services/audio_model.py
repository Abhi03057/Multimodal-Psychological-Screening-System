import numpy as np
import librosa
import pickle
from collections import deque, Counter
from tensorflow.keras.models import load_model
import os

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model   = load_model(os.path.join(MODEL_DIR, "audio_model_best.keras"))
with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

SMOOTHING_WINDOW       = 3
SWITCH_THRESHOLD       = 1
prediction_buffer      = deque(maxlen=3)
confidence_buffer      = deque(maxlen=1)
current_emotion        = "uncertain"
RMS_SILENCE_THRESH     = 0.002
ZCR_NOISE_THRESH       = 0.15
MIN_CONFIDENCE         = 0.2
SKEPTICAL_EMOTIONS     = frozenset({"fear", "angry", "disgust"})
SKEPTICAL_MIN_CONFIDENCE = 0.75
SKEPTICAL_MIN_RMS      = 0.05

BEHAVIOR_MAP = {
    "neutral":          "no strong emotional signal detected",
    "calm":             "relaxed and composed tone",
    "happy":            "positive affect detected",
    "sad":              "low mood or withdrawal indicators",
    "angry":            "heightened stress or frustration",
    "fear":             "anxiety indicators detected",
    "disgust":          "aversion or discomfort signal",
    "surprise":         "alertness or unexpected stimulus",
    "uncertain":        "signal ambiguous - insufficient cues",
    "no speech detected": "no vocal input detected",
}


def load_audio(file_path, duration=3, sr=22050):
    audio, sr = librosa.load(file_path, sr=sr)
    target = sr * duration
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    return audio, sr


def is_silence(audio):
    rms = float(np.mean(librosa.feature.rms(y=audio)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=audio)))
    return rms < RMS_SILENCE_THRESH and zcr < ZCR_NOISE_THRESH


def extract_features(audio, sr, max_len=130):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)
    mel  = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel  = librosa.power_to_db(mel)
    mel  = np.mean(mel.T, axis=0)
    zcr  = float(np.mean(librosa.feature.zero_crossing_rate(y=audio)))
    rms  = float(np.mean(librosa.feature.rms(y=audio)))
    feat = np.concatenate([mfcc, mel])
    if len(feat) < max_len:
        feat = np.pad(feat, (0, max_len - len(feat)))
    else:
        feat = feat[:max_len]
    return feat, zcr, rms


def energy_label(rms):
    if rms < 0.02:  return "low"
    if rms < 0.05:  return "medium"
    return "high"


def stability_label(zcr):
    if zcr < 0.05:  return "stable"
    if zcr < 0.1:   return "moderate"
    return "variable"


def apply_correction(emotion, confidence, rms, zcr):
    """Returns (corrected_emotion, corrected_confidence).
    When a skeptical emotion is downgraded to neutral,
    confidence is reset to a reasonable value.
    """
    if emotion in SKEPTICAL_EMOTIONS:
        if confidence < SKEPTICAL_MIN_CONFIDENCE or rms < SKEPTICAL_MIN_RMS:
            corrections = {"angry": 0.07, "fear": 0.04, "surprise": 0.02}
            if rms < corrections.get(emotion, SKEPTICAL_MIN_RMS):
                return "neutral", max(0.5, confidence * 0.55)
    return emotion, confidence


def smooth_prediction(emotion, confidence):
    global current_emotion
    prediction_buffer.append(emotion)
    confidence_buffer.append(confidence)
    counts = Counter(prediction_buffer)
    top, count = counts.most_common(1)[0]
    if count >= SWITCH_THRESHOLD:
        current_emotion = top
    avg_conf = float(np.mean(list(confidence_buffer)))
    return current_emotion, round(avg_conf, 2)


def analyze_audio(file_path):
    audio, sr = load_audio(file_path)

    if is_silence(audio):
        return {
            "emotion":       "no speech detected",
            "confidence":    0.0,
            "speech_energy": "very low",
            "stability":     "none",
            "behavior_flag": BEHAVIOR_MAP["no speech detected"],
            "trend":         "-",
            "_raw_emotion":  "no speech detected",
            "_raw_conf":     0.0,
            "_rms":          0.0,
            "_zcr":          0.0,
        }

    feat, zcr, rms = extract_features(audio, sr)
    feat_scaled = scaler.transform(feat[np.newaxis, :])
    preds       = model.predict(feat_scaled.reshape(1, -1), verbose=0)
    raw_conf    = float(np.max(preds[0]))
    raw_idx     = int(np.argmax(preds[0]))
    raw_emotion = le.inverse_transform([raw_idx])[0]

    emotion, confidence = apply_correction(raw_emotion, raw_conf, rms, zcr)
    if confidence < MIN_CONFIDENCE:
        emotion    = "uncertain"
        confidence = min(confidence, 0.85)

    prev_emotion         = current_emotion
    emotion, confidence  = smooth_prediction(emotion, confidence)
    trend = f"{prev_emotion} -> {emotion}" if prev_emotion != emotion else emotion

    return {
        "emotion":       emotion,
        "confidence":    round(confidence, 3),
        "speech_energy": energy_label(rms),
        "stability":     stability_label(zcr),
        "behavior_flag": BEHAVIOR_MAP.get(emotion, "signal under analysis"),
        "trend":         trend,
        "_raw_emotion":  raw_emotion,
        "_raw_conf":     round(raw_conf, 4),
        "_rms":          rms,
        "_zcr":          zcr,
    }


def reset_buffer():
    global current_emotion
    prediction_buffer.clear()
    confidence_buffer.clear()
    current_emotion = "uncertain"

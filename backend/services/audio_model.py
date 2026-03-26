import numpy as np
import librosa
import pickle
from collections import deque, Counter
from tensorflow.keras.models import load_model

# ══════════════════════════════════════════════════════
#  LOAD MODEL ASSETS
# ══════════════════════════════════════════════════════
model = load_model("models/audio_model_best.keras")

with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# ══════════════════════════════════════════════════════
#  TEMPORAL SMOOTHING BUFFER
# ══════════════════════════════════════════════════════
SMOOTHING_WINDOW = 5   # rolling window size
SWITCH_THRESHOLD = 3   # votes needed to change reported emotion

prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)
confidence_buffer = deque(maxlen=SMOOTHING_WINDOW)
current_emotion   = "uncertain"


# ══════════════════════════════════════════════════════
#  THRESHOLDS — tuned for real-world laptop mic
# ══════════════════════════════════════════════════════

# Silence gate
RMS_SILENCE_THRESH = 0.01
ZCR_NOISE_THRESH   = 0.08   # high ZCR + low RMS = background noise

# After correction, confidence must exceed this to avoid "uncertain"
# Kept LOW intentionally — let neutral through easily
MIN_CONFIDENCE = 0.40

# Skeptical emotions (model hallucinates these on normal speech)
# Must clear BOTH bars to be trusted
SKEPTICAL_EMOTIONS        = {"angry", "fear", "disgust"}
SKEPTICAL_MIN_CONFIDENCE  = 0.75   # post-calibration
SKEPTICAL_MIN_RMS         = 0.05   # must be genuinely energetic speech

# Behavior flag map
BEHAVIOR_MAP = {
    "neutral"           : "no strong emotional signal detected",
    "calm"              : "relaxed and composed tone",
    "happy"             : "positive affect detected",
    "sad"               : "low mood or withdrawal indicators",
    "angry"             : "heightened stress or frustration",
    "fear"              : "anxiety indicators detected",
    "disgust"           : "aversion or discomfort signal",
    "surprise"          : "alertness or unexpected stimulus",
    "uncertain"         : "signal ambiguous — insufficient cues",
    "no speech detected": "no vocal input detected",
}


# ══════════════════════════════════════════════════════
#  AUDIO LOADING
# ══════════════════════════════════════════════════════
def load_audio(file_path, duration=3, sr=22050):
    audio, _ = librosa.load(file_path, sr=sr)
    target = sr * duration
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]
    return audio


# ══════════════════════════════════════════════════════
#  SILENCE / NOISE GATE
# ══════════════════════════════════════════════════════
def is_silence(audio):
    rms = float(np.mean(librosa.feature.rms(y=audio)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
    if rms < RMS_SILENCE_THRESH:
        return True, rms, zcr
    if rms < 0.02 and zcr > ZCR_NOISE_THRESH:
        return True, rms, zcr
    return False, rms, zcr


# ══════════════════════════════════════════════════════
#  FEATURE EXTRACTION
# ══════════════════════════════════════════════════════
def extract_features(audio, sr=22050, max_len=130):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)

    spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    spec = librosa.power_to_db(spec)
    if spec.shape[1] < max_len:
        spec = np.pad(spec, ((0, 0), (0, max_len - spec.shape[1])))
    else:
        spec = spec[:, :max_len]

    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
    rms = float(np.mean(librosa.feature.rms(y=audio)))

    return mfcc, spec, [zcr, rms]


# ══════════════════════════════════════════════════════
#  ENERGY + STABILITY LABELS
# ══════════════════════════════════════════════════════
def energy_label(rms):
    if rms < 0.02:   return "low"
    elif rms < 0.05: return "medium"
    else:            return "high"

def stability_label(zcr):
    if zcr < 0.05:   return "stable"
    elif zcr < 0.10: return "moderate"
    else:            return "variable"


# ══════════════════════════════════════════════════════
#  REAL-WORLD CORRECTION  (runs BEFORE confidence gate)
#
#  THE KEY FIX: In the old code, the confidence gate ran
#  first, turning everything into "uncertain". Then correction
#  ran on "uncertain" and did nothing — so the buffer filled
#  with "uncertain".
#
#  Correct order: correction → gate → buffer
# ══════════════════════════════════════════════════════
def apply_correction(emotion, confidence, rms, zcr):
    """
    Returns (corrected_emotion, corrected_confidence).
    When a skeptical emotion is downgraded to neutral,
    confidence is reset to a reasonable floor (0.55)
    so it won't be killed by the MIN_CONFIDENCE gate.
    """

    # Rule 1: Skeptical emotions need strong energy + high confidence
    if emotion in SKEPTICAL_EMOTIONS:
        if confidence < SKEPTICAL_MIN_CONFIDENCE or rms < SKEPTICAL_MIN_RMS:
            return "neutral", max(confidence, 0.55)

    # Rule 2: angry + stable ZCR = conversational tone, not angry
    if emotion == "angry" and zcr < 0.07:
        return "neutral", max(confidence, 0.55)

    # Rule 3: fear + low/medium energy = model misfire
    if emotion == "fear" and rms < 0.04:
        return "neutral", max(confidence, 0.55)

    # Rule 4: surprise at very low energy is implausible
    if emotion == "surprise" and rms < 0.02:
        return "neutral", max(confidence, 0.50)

    # happy, neutral, calm, sad pass through unchanged
    return emotion, confidence


# ══════════════════════════════════════════════════════
#  TEMPORAL SMOOTHING ENGINE
# ══════════════════════════════════════════════════════
def smooth_prediction(emotion, confidence):
    global current_emotion

    prediction_buffer.append(emotion)
    confidence_buffer.append(confidence)

    counts = Counter(prediction_buffer)
    dominant_emotion, dominant_count = counts.most_common(1)[0]

    if dominant_count >= SWITCH_THRESHOLD:
        current_emotion = dominant_emotion

    avg_confidence = float(np.mean(list(confidence_buffer)))
    return current_emotion, round(avg_confidence, 2)


# ══════════════════════════════════════════════════════
#  MAIN ANALYSIS FUNCTION
#
#  PIPELINE ORDER:
#  load → silence gate → features → model → calibrate
#  → CORRECTION → confidence gate → smooth → output
# ══════════════════════════════════════════════════════
def analyze_audio(file_path):
    audio = load_audio(file_path)

    # ── 1. Silence / Noise Gate
    silent, rms_raw, zcr_raw = is_silence(audio)
    if silent:
        return {
            "emotion"      : "no speech detected",
            "confidence"   : 0.0,
            "speech_energy": "very low",
            "stability"    : "none",
            "behavior_flag": BEHAVIOR_MAP["no speech detected"],
            "trend"        : "–",
        }

    # ── 2. Feature Extraction
    mfcc, spec, extra = extract_features(audio)
    zcr_val, rms_val  = extra

    # ── 3. Model Prediction
    spec_input  = spec[np.newaxis, ..., np.newaxis]
    mfcc_input  = scaler.transform([mfcc])
    extra_input = np.array(extra).reshape(1, -1)

    pred_probs  = model.predict([spec_input, mfcc_input, extra_input], verbose=0)[0]
    raw_conf    = float(np.max(pred_probs))
    raw_class   = np.argmax(pred_probs)
    raw_emotion = le.inverse_transform([raw_class])[0]

    # ── 4. Confidence Calibration
    calibrated_conf = min(raw_conf * 0.85, 0.95)

    # ── 5. Real-World Correction  ← BEFORE GATE
    corrected_emotion, corrected_conf = apply_correction(
        raw_emotion, calibrated_conf, rms_val, zcr_val
    )

    # ── 6. Confidence Gate  ← AFTER CORRECTION
    if corrected_conf < MIN_CONFIDENCE:
        corrected_emotion = "uncertain"

    # ── 7. Temporal Smoothing
    stable_emotion, stable_conf = smooth_prediction(corrected_emotion, corrected_conf)

    # ── 8. Build Output
    energy    = energy_label(rms_val)
    stability = stability_label(zcr_val)
    flag      = BEHAVIOR_MAP.get(stable_emotion, "signal under analysis")

    if raw_emotion != corrected_emotion and corrected_emotion != stable_emotion:
        trend = f"{raw_emotion} → {corrected_emotion} → {stable_emotion}"
    elif raw_emotion != stable_emotion:
        trend = f"{raw_emotion} → {stable_emotion}"
    else:
        trend = stable_emotion

    return {
        "emotion"      : stable_emotion,
        "confidence"   : stable_conf,
        "speech_energy": energy,
        "stability"    : stability,
        "behavior_flag": flag,
        "trend"        : trend,
        # debug (remove once stable)
        "_raw_emotion" : raw_emotion,
        "_raw_conf"    : round(raw_conf, 3),
        "_rms"         : round(rms_val, 4),
        "_zcr"         : round(zcr_val, 4),
    }


# ══════════════════════════════════════════════════════
#  UTILITY
# ══════════════════════════════════════════════════════
def reset_buffer():
    global current_emotion
    prediction_buffer.clear()
    confidence_buffer.clear()
    current_emotion = "uncertain"
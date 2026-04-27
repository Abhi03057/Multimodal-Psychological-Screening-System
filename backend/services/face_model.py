import cv2
import numpy as np
from collections import Counter
from deepface import DeepFace

CAPTURE_FRAMES    = 5
FRAME_INTERVAL_MS = 500
CAMERA_INDEX      = 1

FACE_BEHAVIOR_MAP = {
    "happy":   "positive facial affect detected",
    "sad":     "signs of low mood or sadness in expression",
    "angry":   "facial tension or frustration indicators",
    "surprise": "surprised or startled expression",
    "fear":    "anxious or fearful facial expression",
    "disgust": "aversion or discomfort in expression",
    "neutral": "neutral facial expression",
    "unknown": "face not detected or unclear expression",
}

def analyze_single_frame(frame):
    try:
        results = DeepFace.analyze(
            img_path=frame,
            actions=["emotion"],
            enforce_detection=False,
            silent=True
        )
        if isinstance(results, list):
            results = results[0]
        dominant = results.get("dominant_emotion", "unknown")
        probs     = results.get("emotion", {})
        confidence = float(probs.get(dominant, 0)) / 100.0
        behavior  = FACE_BEHAVIOR_MAP.get(dominant, "expression under analysis")
        return {
            "emotion":     dominant,
            "confidence":  round(confidence, 3),
            "behavior":    behavior,
            "probs":       {k: v/100.0 for k, v in probs.items()},
        }
    except Exception:
        return {
            "emotion":    "unknown",
            "confidence": 0.0,
            "behavior":   "face not detected or unclear expression",
            "probs":      {},
        }

def analyze_face(camera_index=CAMERA_INDEX, num_frames=CAPTURE_FRAMES):
    cap = cv2.VideoCapture(camera_index)
    emotions = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if ret:
            result = analyze_single_frame(frame)
            emotions.append(result["emotion"])
    cap.release()
    if not emotions:
        return {"emotion": "unknown", "confidence": 0.0, "behavior": "face not detected or unclear expression"}
    dominant = Counter(emotions).most_common(1)[0][0]
    behavior = FACE_BEHAVIOR_MAP.get(dominant, "expression under analysis")
    return {"emotion": dominant, "confidence": 0.8, "behavior": behavior}

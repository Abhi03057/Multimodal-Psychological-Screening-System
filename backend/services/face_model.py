import cv2
import numpy as np
from collections import Counter
from deepface import DeepFace

# ══════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════
CAPTURE_FRAMES     = 5       # Number of frames to sample over the analysis window
FRAME_INTERVAL_MS  = 500     # Milliseconds between frame captures
CAMERA_INDEX       = 1       # DroidCam virtual camera (change to 0 for built-in webcam)

# Behavior flag map (mirrors audio_model.py style)
FACE_BEHAVIOR_MAP = {
    "happy"    : "positive facial affect detected",
    "sad"      : "signs of low mood or sadness in expression",
    "angry"    : "facial tension or frustration indicators",
    "surprise" : "surprised or startled expression",
    "fear"     : "anxious or fearful facial expression",
    "disgust"  : "aversion or discomfort in expression",
    "neutral"  : "neutral facial expression",
    "unknown"  : "face not detected or unclear expression",
}


def analyze_single_frame(frame):
    """Run DeepFace emotion analysis on a single BGR frame.
    Returns (emotion, confidence) or ('unknown', 0.0) on failure."""
    try:
        results = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
            silent=True
        )
        if isinstance(results, list):
            results = results[0]
        
        emotion = results["dominant_emotion"]
        confidence = results["emotion"][emotion] / 100.0
        return emotion, confidence, results["emotion"]
    except Exception:
        return "unknown", 0.0, {}


def analyze_face(camera_index=CAMERA_INDEX, num_frames=CAPTURE_FRAMES):
    """
    Capture multiple frames from webcam, run emotion detection on each,
    and return the dominant emotion via majority vote.
    
    Returns dict with: emotion, confidence, all_emotions, behavior_flag
    """
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        return {
            "emotion"       : "unknown",
            "confidence"    : 0.0,
            "all_emotions"  : {},
            "behavior_flag" : "camera not available",
        }
    
    emotions_detected = []
    confidences = []
    all_emotion_scores = {}
    
    try:
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                continue
            
            emotion, conf, scores = analyze_single_frame(frame)
            emotions_detected.append(emotion)
            confidences.append(conf)
            
            # Accumulate scores for averaging
            for emo, score in scores.items():
                all_emotion_scores[emo] = all_emotion_scores.get(emo, 0) + score
            
            # Small delay between frames
            cv2.waitKey(FRAME_INTERVAL_MS)
    finally:
        cap.release()
    
    if not emotions_detected:
        return {
            "emotion"       : "unknown",
            "confidence"    : 0.0,
            "all_emotions"  : {},
            "behavior_flag" : "no frames captured",
        }
    
    # Majority vote for dominant emotion
    counts = Counter(emotions_detected)
    dominant_emotion = counts.most_common(1)[0][0]
    
    # Average confidence
    avg_confidence = round(float(np.mean(confidences)), 2)
    
    # Average emotion scores
    num = len(emotions_detected)
    avg_scores = {k: round(v / num, 2) for k, v in all_emotion_scores.items()}
    
    flag = FACE_BEHAVIOR_MAP.get(dominant_emotion, "expression under analysis")
    
    return {
        "emotion"       : dominant_emotion,
        "confidence"    : avg_confidence,
        "all_emotions"  : avg_scores,
        "behavior_flag" : flag,
    }

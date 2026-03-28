"""Quick smoke test for the guided session computation functions."""
from routes.live import (
    compute_voice_score, compute_face_score, compute_reactivity,
    compute_fusion_and_confidence
)

# Test voice_score
vs = compute_voice_score("high", "variable", "angry")
print(f"Voice score (high energy, variable, angry): {vs}")
assert 0.7 < vs <= 1.0, f"Expected high voice score, got {vs}"

vs2 = compute_voice_score("low", "stable", "calm")
print(f"Voice score (low energy, stable, calm): {vs2}")
assert 0.0 <= vs2 < 0.3, f"Expected low voice score, got {vs2}"

# Test face_score
fs = compute_face_score("angry", 0.9, {"angry": 80, "neutral": 10, "happy": 5, "sad": 5})
print(f"Face score (angry, 0.9 conf): {fs}")
assert 0.4 < fs <= 1.0, f"Expected elevated face score, got {fs}"

fs2 = compute_face_score("happy", 0.85, {"happy": 85, "neutral": 10, "sad": 5})
print(f"Face score (happy, 0.85 conf): {fs2}")
assert 0.0 <= fs2 < 0.3, f"Expected low face score, got {fs2}"

# Test reactivity
per_q = {
    "baseline": {"avg_voice": 0.35, "avg_face": 0.20, "avg_confidence": 0.72, "count": 3},
    "stress":   {"avg_voice": 0.70, "avg_face": 0.60, "avg_confidence": 0.80, "count": 4},
    "anxiety":  {"avg_voice": 0.65, "avg_face": 0.55, "avg_confidence": 0.75, "count": 3},
}
r = compute_reactivity(per_q)
print(f"Reactivity: {r}")
assert r > 0.2, f"Expected positive reactivity, got {r}"

# Edge case: no stress group
r0 = compute_reactivity({"baseline": per_q["baseline"]})
print(f"Reactivity (no stress group): {r0}")
assert r0 == 0.0, f"Expected 0 reactivity, got {r0}"

# Test fusion
fusion = compute_fusion_and_confidence(12, 10, 0.55, 0.45, r)
score = fusion["final_score"]
risk = fusion["risk_level"]
conf = fusion["confidence_score"]
align = fusion["alignment_level"]
print(f"Fusion: score={score}, risk={risk}, conf={conf}, align={align}")
assert 0 < score < 1
assert risk in ("Low", "Moderate", "High")
assert 0 < conf <= 1

print("\n=== ALL TESTS PASSED ===")

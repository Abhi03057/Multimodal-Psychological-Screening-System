from fastapi import APIRouter
from services.scoring import analyze_data
from services.llm_service import generate_report

router = APIRouter(prefix="/analysis")  # 🔴 THIS WAS MISSING

@router.post("/")
def analyze(data: dict):
    phq9 = data.get("phq9_score", 0)
    gad7 = data.get("gad7_score", 0)

    analysis = analyze_data()

    if phq9 > 14 or gad7 > 14:
        risk = "high"
    elif phq9 > 8:
        risk = "moderate"
    else:
        risk = "low"

    result = {
        "phq9_score": phq9,
        "gad7_score": gad7,
        "audio_features": analysis.get("audio_features", {}),
        "facial_emotion": analysis.get("facial_emotion", "unknown"),
        "risk_level": risk
    }

    result["report"] = generate_report(result)

    return result
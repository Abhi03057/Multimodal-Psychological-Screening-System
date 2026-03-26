from fastapi import APIRouter

router = APIRouter(prefix="/questionnaire")

@router.post("/")
def submit_questionnaire(data: dict):
    score = sum(data.values())

    if score <= 4:
        level = "minimal"
    elif score <= 9:
        level = "mild"
    elif score <= 14:
        level = "moderate"
    elif score <= 19:
        level = "moderately severe"
    else:
        level = "severe"

    return {
        "total_score": score,
        "level": level
    }
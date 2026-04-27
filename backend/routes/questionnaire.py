from fastapi import APIRouter

router = APIRouter(prefix="/questionnaire")

@router.post("/")
async def submit_questionnaire(data: dict):
    total = sum(data.get(f"q{i}", 0) for i in range(1, 10))
    if total <= 4:
        level = "minimal"
    elif total <= 9:
        level = "mild"
    elif total <= 14:
        level = "moderate"
    elif total <= 19:
        level = "moderately severe"
    else:
        level = "severe"
    return {"total_score": total, "level": level}

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.analysis import router as analysis_router
from routes.questionnaire import router as questionnaire_router
from routes.multimodal import router as multimodal_router
from routes.live import router as live_router

app = FastAPI(
    title="Psychological Screening System API",
    description="Multimodal system for psychological screening combining text and audio analysis.",
    version="1.0.0"
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production to specific frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include modules
app.include_router(analysis_router)
app.include_router(questionnaire_router)
app.include_router(multimodal_router)
app.include_router(live_router)

@app.get("/")
def read_root():
    return {"status": "Backend is running", "message": "Welcome to the Psychological Screening System API."}

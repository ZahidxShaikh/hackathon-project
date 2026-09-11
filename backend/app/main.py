"""FastAPI application for the local manganese prospectivity dashboard."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.prospectivity import router as prospectivity_router


app = FastAPI(title="Balaghat Manganese Prospectivity API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)
app.include_router(prospectivity_router)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

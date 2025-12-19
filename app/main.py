"""FastAPI application for Xynenyx LLM Service."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Xynenyx LLM Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    """Readiness check endpoint."""
    return {"status": "ready"}


# TODO: Add completion, embedding, provider endpoints
# This will be implemented in Phase 2


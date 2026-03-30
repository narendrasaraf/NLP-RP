"""
app/main.py
-----------
FastAPI application entry point.

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Interactive docs available at:
    http://localhost:8000/docs     (Swagger UI)
    http://localhost:8000/redoc   (ReDoc)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "Cognitive Regulation API",
    description = (
        "Predictive Cognitive State Modeling for Adaptive Game Difficulty Regulation.\n\n"
        "Computes CII from gameplay telemetry + NLP features and adjusts difficulty "
        "to maintain optimal player engagement (flow state)."
    ),
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ---------------------------------------------------------------------------
# CORS — allow Streamlit dashboard and game clients
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # Restrict in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ---------------------------------------------------------------------------
# Register routes
# ---------------------------------------------------------------------------

app.include_router(router, prefix="/api/v1")


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

@app.get("/", tags=["System"])
def root():
    return {
        "system"     : "Cognitive Regulation API",
        "version"    : "1.0.0",
        "docs"       : "/docs",
        "health"     : "/api/v1/health",
        "process"    : "POST /api/v1/process",
    }


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host    = settings.api_host,
        port    = settings.api_port,
        reload  = settings.debug,
    )

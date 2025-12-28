"""FastAPI application for Xynenyx LLM Service."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import settings
from app.routers import completions, embeddings, providers
from app.schemas.errors import create_error_response
from app.middleware.logging import LoggingMiddleware
import logging

# Configure structured logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    # Could initialize connections, warm up models, etc.
    yield
    # Shutdown
    # Could close connections, cleanup resources, etc.


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="LLM service for Xynenyx with multi-provider support",
    lifespan=lifespan,
)

# Add logging middleware (before CORS to capture all requests)
app.add_middleware(LoggingMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Include routers
app.include_router(completions.router)
app.include_router(embeddings.router)
app.include_router(providers.router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    """Readiness check endpoint with dependency verification."""
    checks = {}
    all_ready = True

    # Check Supabase connection
    try:
        from app.tracking.usage import UsageTracker
        tracker = UsageTracker()
        # Simple query to verify connection
        result = tracker.client.table("llm_usage").select("id").limit(1).execute()
        checks["supabase"] = "ready"
    except Exception as e:
        logger.error(f"Supabase connection check failed: {e}")
        checks["supabase"] = f"error: {str(e)}"
        all_ready = False

    # Check OpenAI API key (if enabled)
    if settings.openai_enabled:
        if not settings.openai_api_key or settings.openai_api_key.startswith("sk-your"):
            checks["openai"] = "error: API key not configured"
            all_ready = False
        else:
            checks["openai"] = "ready"

    status_code = 200 if all_ready else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if all_ready else "not ready",
            "checks": checks,
        },
    )


# Error handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    errors = exc.errors()
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=create_error_response(
            detail="Validation error",
            status_code=422,
            code="VALIDATION_ERROR",
            errors=errors,
        ),
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            detail=exc.detail,
            status_code=exc.status_code,
            code=f"HTTP_{exc.status_code}",
        ),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=create_error_response(
            detail="Internal server error",
            status_code=500,
            code="INTERNAL_SERVER_ERROR",
        ),
    )

"""FastAPI application entrypoint and lifespan lifecycle manager."""

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import settings
from app.services.predictor import FraudPredictor

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("fraud_api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle manager."""
    logger.info("Initializing Real-Time Fraud Detection Microservice...")
    try:
        predictor = FraudPredictor.get_instance()
        predictor.load_model()
        logger.info(
            "Model successfully loaded. Version: %s, Optimal Threshold: %.4f",
            predictor.model_version,
            predictor.optimal_threshold,
        )
    except Exception as err:
        logger.error("Failed to load model on startup: %s", err, exc_info=True)
    yield
    logger.info("Shutting down Fraud Detection Microservice...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Attach server processing time in milliseconds to every response header."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    return response


@app.get("/", summary="Root Information")
def root_endpoint():
    """Service status and navigational links."""
    predictor = FraudPredictor.get_instance()
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "model_loaded": predictor.is_loaded,
        "documentation": "/docs",
        "health_probes": {
            "liveness": "/health/live",
            "readiness": "/health/ready",
        },
        "observability": "/metrics",
    }


# Include API routes
app.include_router(router)


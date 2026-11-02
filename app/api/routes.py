"""FastAPI route handlers for predictions, health probes, and Prometheus metrics."""

from fastapi import APIRouter, HTTPException, Response, status
from app.core.config import settings
from app.schemas.transaction import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelMetadataResponse,
)
from app.services.predictor import FraudPredictor

router = APIRouter()


@router.post(
    "/v1/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Real-time transaction fraud scoring",
    description="Scores a single transaction against the trained gradient boosting model in sub-10ms.",
)
def predict_transaction(request: PredictionRequest) -> PredictionResponse:
    """Evaluate fraud risk on a single transaction."""
    predictor = FraudPredictor.get_instance()
    try:
        return predictor.predict_single(request)
    except RuntimeError as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(err),
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Inference error: {err}",
        )


@router.post(
    "/v1/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Vectorized batch transaction scoring",
    description="Processes up to 500 transactions concurrently with optimized vectorized matrix evaluation.",
)
def predict_batch_transactions(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Score a batch of transactions concurrently."""
    predictor = FraudPredictor.get_instance()
    try:
        return predictor.predict_batch(request)
    except RuntimeError as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(err),
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch inference error: {err}",
        )


@router.get(
    "/health/live",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Kubernetes Liveness Probe",
    description="Signals if the HTTP process is running. Kubernetes restarts the container if this fails.",
)
def liveness_probe() -> HealthResponse:
    """Verify that process is responsive."""
    predictor = FraudPredictor.get_instance()
    return HealthResponse(
        status="live",
        model_loaded=predictor.is_loaded,
        model_version=predictor.model_version,
        environment=settings.ENVIRONMENT,
    )


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Kubernetes Readiness Probe",
    description="Signals if the container is ready to receive network traffic (i.e. model pipeline is in memory).",
)
def readiness_probe() -> HealthResponse:
    """Verify that model is fully loaded and ready for production inference."""
    predictor = FraudPredictor.get_instance()
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not yet initialized.",
        )
    return HealthResponse(
        status="ready",
        model_loaded=True,
        model_version=predictor.model_version,
        environment=settings.ENVIRONMENT,
    )


@router.get(
    "/model/metadata",
    response_model=ModelMetadataResponse,
    status_code=status.HTTP_200_OK,
    summary="Model Architecture & Training Metadata",
    description="Returns production model version, algorithm, trained metrics, and features list.",
)
def get_model_metadata() -> ModelMetadataResponse:
    """Expose model provenance and evaluation audit metrics."""
    predictor = FraudPredictor.get_instance()
    if not predictor.is_loaded or not predictor.metadata:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata is unavailable.",
        )
    return ModelMetadataResponse(
        model_name=predictor.metadata.get("model_name", "realtime-fraud-detector"),
        model_version=predictor.model_version,
        algorithm=predictor.metadata.get("algorithm", "HistGradientBoostingClassifier"),
        created_at=predictor.metadata.get("created_at", "unknown"),
        optimal_threshold=predictor.optimal_threshold,
        features=predictor.feature_columns,
        metrics=predictor.metadata.get("metrics", {}),
    )


@router.get(
    "/metrics",
    summary="Prometheus Metrics Endpoint",
    description="Exposes application and inference metrics in standard Prometheus exposition format.",
)
def metrics_endpoint() -> Response:
    """Scraped by Prometheus server running in Kubernetes cluster."""
    predictor = FraudPredictor.get_instance()
    content = predictor.get_prometheus_metrics()
    return Response(content=content, media_type="text/plain; version=0.0.4; charset=utf-8")


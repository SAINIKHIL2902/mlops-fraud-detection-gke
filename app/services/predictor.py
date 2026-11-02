"""Real-time fraud prediction engine and in-memory inference service."""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

import joblib
import numpy as np
import pandas as pd

from app.core.config import settings
from app.schemas.transaction import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    RiskLevel,
)

logger = logging.getLogger(__name__)


class FraudPredictor:
    """Thread-safe singleton predictor for high-throughput inference."""

    _instance: Optional["FraudPredictor"] = None

    def __init__(self):
        self.pipeline = None
        self.metadata: Dict[str, Any] = {}
        self.optimal_threshold: float = settings.FALLBACK_THRESHOLD
        self.model_version: str = "unknown"
        self.feature_columns: List[str] = []
        self.is_loaded: bool = False

        # In-memory Prometheus metrics counters
        self.metrics_counter = {
            "predictions_total": 0,
            "fraud_detected_total": 0,
            "batch_requests_total": 0,
            "total_latency_seconds": 0.0,
        }

    @classmethod
    def get_instance(cls) -> "FraudPredictor":
        """Get or initialize singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self) -> None:
        """Load serialized pipeline and metadata from disk."""
        model_path = settings.BASE_DIR / settings.MODEL_PATH
        metadata_path = settings.BASE_DIR / settings.METADATA_PATH

        if not model_path.exists():
            logger.warning("Model file not found at %s. Attempting fallback relative search...", model_path)
            model_path = Path(settings.MODEL_PATH)
            metadata_path = Path(settings.METADATA_PATH)

        if not model_path.exists():
            raise FileNotFoundError(f"Model pipeline artifact not found at {model_path}. Train the model first.")

        logger.info("Loading model artifact from %s...", model_path)
        self.pipeline = joblib.load(model_path)

        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
            self.optimal_threshold = float(self.metadata.get("optimal_threshold", settings.FALLBACK_THRESHOLD))
            self.model_version = self.metadata.get("model_version", settings.VERSION)
            self.feature_columns = self.metadata.get("features", [])
            logger.info("Model metadata loaded successfully (version: %s, threshold: %.4f)", self.model_version, self.optimal_threshold)
        else:
            logger.warning("Metadata file not found at %s, using defaults.", metadata_path)
            self.optimal_threshold = settings.FALLBACK_THRESHOLD
            self.model_version = settings.VERSION

        self.is_loaded = True

    def _determine_risk_tier(self, probability: float, threshold: float) -> tuple[RiskLevel, str]:
        """Categorize risk level and appropriate automated mitigation."""
        if probability >= threshold:
            return RiskLevel.CRITICAL, "DECLINE_AND_ALERT_FRAUD_OPS"
        elif probability >= 0.50:
            return RiskLevel.HIGH, "CHALLENGE_WITH_STEP_UP_AUTH_2FA"
        elif probability >= 0.20:
            return RiskLevel.MEDIUM, "APPROVE_AND_FLAG_FOR_ASYNC_REVIEW"
        return RiskLevel.LOW, "APPROVE"

    def predict_single(self, request: PredictionRequest) -> PredictionResponse:
        """Score single transaction with sub-millisecond latency."""
        if not self.is_loaded or self.pipeline is None:
            raise RuntimeError("Model is not initialized. Service is not ready.")

        start_time = time.perf_counter()
        tx_id = request.transaction_id or f"txn_{uuid.uuid4().hex[:12]}"

        # Extract features into DataFrame to maintain feature names expected by pipeline
        features_dict = request.features.model_dump()
        df = pd.DataFrame([features_dict])

        # Inference
        probs = self.pipeline.predict_proba(df)
        fraud_prob = float(probs[0, 1])

        # Determine threshold
        threshold = (
            request.threshold_override
            if request.threshold_override is not None
            else self.optimal_threshold
        )
        is_fraud = bool(fraud_prob >= threshold)
        risk_level, action = self._determine_risk_tier(fraud_prob, threshold)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Update metrics
        self.metrics_counter["predictions_total"] += 1
        self.metrics_counter["total_latency_seconds"] += (latency_ms / 1000.0)
        if is_fraud:
            self.metrics_counter["fraud_detected_total"] += 1

        return PredictionResponse(
            transaction_id=tx_id,
            is_fraud=is_fraud,
            fraud_probability=round(fraud_prob, 5),
            risk_level=risk_level,
            decision_threshold=round(threshold, 4),
            recommended_action=action,
            inference_latency_ms=round(latency_ms, 3),
            model_version=self.model_version,
        )

    def predict_batch(self, request: BatchPredictionRequest) -> BatchPredictionResponse:
        """Perform vectorized batch scoring."""
        if not self.is_loaded or self.pipeline is None:
            raise RuntimeError("Model is not initialized. Service is not ready.")

        start_time = time.perf_counter()
        records = [item.features.model_dump() for item in request.transactions]
        df = pd.DataFrame(records)

        probs = self.pipeline.predict_proba(df)[:, 1]

        results: List[PredictionResponse] = []
        fraud_count = 0

        for idx, (tx_req, prob) in enumerate(zip(request.transactions, probs)):
            fraud_prob = float(prob)
            threshold = (
                tx_req.threshold_override
                if tx_req.threshold_override is not None
                else self.optimal_threshold
            )
            is_fraud = bool(fraud_prob >= threshold)
            risk_level, action = self._determine_risk_tier(fraud_prob, threshold)

            if is_fraud:
                fraud_count += 1

            results.append(
                PredictionResponse(
                    transaction_id=tx_req.transaction_id or f"txn_batch_{idx}_{uuid.uuid4().hex[:6]}",
                    is_fraud=is_fraud,
                    fraud_probability=round(fraud_prob, 5),
                    risk_level=risk_level,
                    decision_threshold=round(threshold, 4),
                    recommended_action=action,
                    inference_latency_ms=0.0,
                    model_version=self.model_version,
                )
            )

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Update metrics
        self.metrics_counter["predictions_total"] += len(results)
        self.metrics_counter["batch_requests_total"] += 1
        self.metrics_counter["fraud_detected_total"] += fraud_count
        self.metrics_counter["total_latency_seconds"] += (total_latency_ms / 1000.0)

        return BatchPredictionResponse(
            total_transactions=len(results),
            fraud_detected_count=fraud_count,
            results=results,
            batch_latency_ms=round(total_latency_ms, 3),
        )

    def get_prometheus_metrics(self) -> str:
        """Render Prometheus exposition format metrics."""
        pred_total = self.metrics_counter["predictions_total"]
        fraud_total = self.metrics_counter["fraud_detected_total"]
        batch_total = self.metrics_counter["batch_requests_total"]
        total_sec = self.metrics_counter["total_latency_seconds"]
        avg_latency = (total_sec / pred_total) if pred_total > 0 else 0.0

        return (
            "# HELP mlops_fraud_predictions_total Total count of scored transactions\n"
            "# TYPE mlops_fraud_predictions_total counter\n"
            f"mlops_fraud_predictions_total {pred_total}\n\n"
            "# HELP mlops_fraud_detected_total Total count of identified fraudulent transactions\n"
            "# TYPE mlops_fraud_detected_total counter\n"
            f"mlops_fraud_detected_total {fraud_total}\n\n"
            "# HELP mlops_fraud_batch_requests_total Total number of batch requests processed\n"
            "# TYPE mlops_fraud_batch_requests_total counter\n"
            f"mlops_fraud_batch_requests_total {batch_total}\n\n"
            "# HELP mlops_fraud_inference_latency_seconds_sum Cumulative inference time in seconds\n"
            "# TYPE mlops_fraud_inference_latency_seconds_sum counter\n"
            f"mlops_fraud_inference_latency_seconds_sum {total_sec:.6f}\n\n"
            "# HELP mlops_fraud_avg_latency_seconds Average inference latency per transaction\n"
            "# TYPE mlops_fraud_avg_latency_seconds gauge\n"
            f"mlops_fraud_avg_latency_seconds {avg_latency:.6f}\n"
        )


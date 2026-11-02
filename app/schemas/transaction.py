"""Pydantic schemas for transaction validation and prediction contracts."""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class RiskLevel(str, Enum):
    """Categorical risk tiers for transaction decisioning."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TransactionFeatures(BaseModel):
    """Features required by the fraud detection pipeline."""

    amount: float = Field(
        ...,
        gt=0.0,
        description="Transaction amount in currency units (must be > 0.0)",
        examples=[124.50],
    )
    hour_of_day: int = Field(
        ...,
        ge=0,
        le=23,
        description="Local hour of day when transaction occurred (0 to 23)",
        examples=[14],
    )
    distance_from_home_km: float = Field(
        ...,
        ge=0.0,
        description="Physical distance in kilometers from cardholder billing location",
        examples=[12.4],
    )
    distance_from_last_tx_km: float = Field(
        ...,
        ge=0.0,
        description="Distance in kilometers from previous transaction location",
        examples=[2.1],
    )
    tx_velocity_1h: int = Field(
        ...,
        ge=0,
        description="Count of transactions by this cardholder within the past 60 minutes",
        examples=[1],
    )
    tx_amount_ratio_to_mean: float = Field(
        ...,
        gt=0.0,
        description="Ratio of current amount to cardholder's 30-day historical average amount",
        examples=[1.05],
    )
    is_foreign_tx: int = Field(
        ...,
        ge=0,
        le=1,
        description="1 if foreign or cross-border transaction, 0 if domestic",
        examples=[0],
    )
    is_high_risk_merchant: int = Field(
        ...,
        ge=0,
        le=1,
        description="1 if high-risk MCC (crypto, jewelry, wire transfer, casino), 0 otherwise",
        examples=[0],
    )
    device_trust_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Risk engine trust score for the client device (0.0 untrusted, 1.0 highly trusted)",
        examples=[0.88],
    )


class PredictionRequest(BaseModel):
    """Request payload for single transaction inference."""

    transaction_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the transaction (UUID or gateway reference)",
        examples=["txn_9a87f2e1b4c3"],
    )
    features: TransactionFeatures
    threshold_override: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional custom threshold override for specialized risk profiles",
    )


class PredictionResponse(BaseModel):
    """Inference decision and audit metadata."""

    transaction_id: str
    is_fraud: bool
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    decision_threshold: float
    recommended_action: str
    inference_latency_ms: float
    model_version: str


class BatchPredictionRequest(BaseModel):
    """Batch inference request payload."""

    transactions: List[PredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of transactions to score in batch (maximum 500 per call)",
    )


class BatchPredictionResponse(BaseModel):
    """Batch inference response payload."""

    total_transactions: int
    fraud_detected_count: int
    results: List[PredictionResponse]
    batch_latency_ms: float


class HealthResponse(BaseModel):
    """Kubernetes liveness and readiness probe response."""

    status: str
    model_loaded: bool
    model_version: str
    environment: str


class ModelMetadataResponse(BaseModel):
    """Detailed model metrics and architecture specifications."""

    model_name: str
    model_version: str
    algorithm: str
    created_at: str
    optimal_threshold: float
    features: List[str]
    metrics: Dict[str, Any]


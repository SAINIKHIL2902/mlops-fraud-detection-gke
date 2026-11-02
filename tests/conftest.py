"""Pytest fixtures and test client configuration."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.predictor import FraudPredictor


@pytest.fixture(scope="session", autouse=True)
def initialize_model():
    """Ensure model is loaded before test execution."""
    predictor = FraudPredictor.get_instance()
    if not predictor.is_loaded:
        predictor.load_model()
    return predictor


@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_legitimate_transaction():
    """Realistic low-risk domestic transaction payload."""
    return {
        "transaction_id": "txn_legit_001",
        "features": {
            "amount": 24.50,
            "hour_of_day": 14,
            "distance_from_home_km": 3.2,
            "distance_from_last_tx_km": 1.1,
            "tx_velocity_1h": 1,
            "tx_amount_ratio_to_mean": 0.95,
            "is_foreign_tx": 0,
            "is_high_risk_merchant": 0,
            "device_trust_score": 0.95,
        },
    }


@pytest.fixture
def sample_fraudulent_transaction():
    """Realistic high-risk anomaly transaction payload."""
    return {
        "transaction_id": "txn_fraud_001",
        "features": {
            "amount": 1850.00,
            "hour_of_day": 3,
            "distance_from_home_km": 420.0,
            "distance_from_last_tx_km": 380.0,
            "tx_velocity_1h": 8,
            "tx_amount_ratio_to_mean": 6.8,
            "is_foreign_tx": 1,
            "is_high_risk_merchant": 1,
            "device_trust_score": 0.12,
        },
    }


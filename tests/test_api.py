"""Functional and integration tests for FastAPI endpoints."""

from fastapi import status


def test_root_endpoint(client):
    """Test root status endpoint."""
    response = client.get("/")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "service" in data
    assert data["model_loaded"] is True
    assert "health_probes" in data


def test_liveness_probe(client):
    """Test Kubernetes liveness probe."""
    response = client.get("/health/live")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == "live"


def test_readiness_probe(client):
    """Test Kubernetes readiness probe."""
    response = client.get("/health/ready")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == "ready"
    assert response.json()["model_loaded"] is True


def test_model_metadata(client):
    """Test model metadata and provenance endpoint."""
    response = client.get("/model/metadata")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "model_version" in data
    assert "optimal_threshold" in data
    assert len(data["features"]) > 0
    assert "test_roc_auc" in data["metrics"]


def test_predict_legitimate_transaction(client, sample_legitimate_transaction):
    """Verify legitimate transaction receives LOW risk decision."""
    response = client.post("/v1/predict", json=sample_legitimate_transaction)
    assert response.status_code == status.HTTP_200_OK
    result = response.json()
    assert result["transaction_id"] == "txn_legit_001"
    assert result["is_fraud"] is False
    assert result["risk_level"] in ["LOW", "MEDIUM"]
    assert result["fraud_probability"] < 0.50
    assert "inference_latency_ms" in result
    assert result["inference_latency_ms"] < 100.0  # sub-100ms requirement


def test_predict_fraudulent_transaction(client, sample_fraudulent_transaction):
    """Verify fraudulent anomaly transaction receives CRITICAL alert."""
    response = client.post("/v1/predict", json=sample_fraudulent_transaction)
    assert response.status_code == status.HTTP_200_OK
    result = response.json()
    assert result["transaction_id"] == "txn_fraud_001"
    assert result["is_fraud"] is True
    assert result["risk_level"] == "CRITICAL"
    assert result["fraud_probability"] >= result["decision_threshold"]
    assert result["recommended_action"] == "DECLINE_AND_ALERT_FRAUD_OPS"


def test_predict_validation_errors(client):
    """Verify Pydantic input validation rejects malformed requests."""
    # Negative amount
    bad_payload = {
        "features": {
            "amount": -50.0,
            "hour_of_day": 12,
            "distance_from_home_km": 5.0,
            "distance_from_last_tx_km": 2.0,
            "tx_velocity_1h": 1,
            "tx_amount_ratio_to_mean": 1.0,
            "is_foreign_tx": 0,
            "is_high_risk_merchant": 0,
            "device_trust_score": 0.8,
        }
    }
    response = client.post("/v1/predict", json=bad_payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    # Invalid hour (> 23)
    bad_payload["features"]["amount"] = 50.0
    bad_payload["features"]["hour_of_day"] = 28
    response = client.post("/v1/predict", json=bad_payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_predict_batch_transactions(client, sample_legitimate_transaction, sample_fraudulent_transaction):
    """Test batch prediction endpoint with mixed transactions."""
    batch_payload = {
        "transactions": [
            sample_legitimate_transaction,
            sample_fraudulent_transaction,
        ]
    }
    response = client.post("/v1/predict/batch", json=batch_payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["total_transactions"] == 2
    assert data["fraud_detected_count"] == 1
    assert len(data["results"]) == 2
    assert "batch_latency_ms" in data


def test_prometheus_metrics(client):
    """Verify Prometheus metrics text exposition format."""
    response = client.get("/metrics")
    assert response.status_code == status.HTTP_200_OK
    text = response.text
    assert "mlops_fraud_predictions_total" in text
    assert "mlops_fraud_detected_total" in text
    assert "mlops_fraud_inference_latency_seconds_sum" in text


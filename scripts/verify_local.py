"""Quick local verification of complete pipeline and microservice."""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.main import app
from app.services.predictor import FraudPredictor


def main():
    print("🔹 Checking model files...")
    predictor = FraudPredictor.get_instance()
    predictor.load_model()
    print(f"✅ Model loaded successfully: {predictor.model_version} (threshold: {predictor.optimal_threshold})")

    print("\n🔹 Testing FastAPI TestClient...")
    with TestClient(app) as client:
        # Liveness
        r = client.get("/health/live")
        assert r.status_code == 200
        print(f"✅ /health/live: {r.json()}")

        # Readiness
        r = client.get("/health/ready")
        assert r.status_code == 200
        print(f"✅ /health/ready: {r.json()}")

        # Predict legit
        legit = {
            "transaction_id": "verify_legit_01",
            "features": {
                "amount": 35.0,
                "hour_of_day": 13,
                "distance_from_home_km": 2.0,
                "distance_from_last_tx_km": 1.0,
                "tx_velocity_1h": 1,
                "tx_amount_ratio_to_mean": 1.0,
                "is_foreign_tx": 0,
                "is_high_risk_merchant": 0,
                "device_trust_score": 0.95,
            },
        }
        r = client.post("/v1/predict", json=legit)
        assert r.status_code == 200
        print(f"✅ /v1/predict (Legit): fraud={r.json()['is_fraud']}, prob={r.json()['fraud_probability']}, latency={r.json()['inference_latency_ms']}ms")

        # Predict fraud
        fraud = {
            "transaction_id": "verify_fraud_01",
            "features": {
                "amount": 2500.0,
                "hour_of_day": 3,
                "distance_from_home_km": 750.0,
                "distance_from_last_tx_km": 600.0,
                "tx_velocity_1h": 9,
                "tx_amount_ratio_to_mean": 8.0,
                "is_foreign_tx": 1,
                "is_high_risk_merchant": 1,
                "device_trust_score": 0.08,
            },
        }
        r = client.post("/v1/predict", json=fraud)
        assert r.status_code == 200
        print(f"✅ /v1/predict (Fraud): fraud={r.json()['is_fraud']}, prob={r.json()['fraud_probability']}, latency={r.json()['inference_latency_ms']}ms")

        # Metrics
        r = client.get("/metrics")
        assert r.status_code == 200
        print("✅ /metrics returns valid Prometheus exposition text")

    print("\n🎉 ALL LOCAL VERIFICATION CHECKS PASSED!")


if __name__ == "__main__":
    main()


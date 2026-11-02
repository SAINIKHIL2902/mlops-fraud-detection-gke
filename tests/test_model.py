"""Unit tests for ML model pipeline inference, performance, and validation."""

import time
import pandas as pd
from pipeline.data_generator import FEATURE_COLUMNS


def test_model_loaded_and_schema(initialize_model):
    """Verify that predictor loaded pipeline with correct feature schema."""
    predictor = initialize_model
    assert predictor.is_loaded is True
    assert predictor.pipeline is not None
    assert predictor.optimal_threshold > 0.0
    assert len(predictor.feature_columns) == len(FEATURE_COLUMNS)


def test_model_inference_latency_benchmark(initialize_model):
    """Verify single prediction inference executes well below SLA (sub-20ms)."""
    predictor = initialize_model

    sample_features = {
        "amount": 75.20,
        "hour_of_day": 15,
        "distance_from_home_km": 15.0,
        "distance_from_last_tx_km": 4.0,
        "tx_velocity_1h": 2,
        "tx_amount_ratio_to_mean": 1.1,
        "is_foreign_tx": 0,
        "is_high_risk_merchant": 0,
        "device_trust_score": 0.85,
    }
    df = pd.DataFrame([sample_features])

    # Warmup
    _ = predictor.pipeline.predict_proba(df)

    # Benchmark 50 iterations
    latencies = []
    for _ in range(50):
        start = time.perf_counter()
        probs = predictor.pipeline.predict_proba(df)
        latencies.append((time.perf_counter() - start) * 1000.0)

    median_latency = sorted(latencies)[len(latencies) // 2]
    assert median_latency < 20.0, f"Median latency {median_latency:.2f}ms exceeded 20ms SLA"
    assert 0.0 <= float(probs[0, 1]) <= 1.0


def test_probability_distribution(initialize_model):
    """Verify probability outputs are normalized between 0.0 and 1.0."""
    predictor = initialize_model
    test_cases = [
        {"amount": 10.0, "hour_of_day": 12, "distance_from_home_km": 1.0, "distance_from_last_tx_km": 0.5, "tx_velocity_1h": 0, "tx_amount_ratio_to_mean": 0.8, "is_foreign_tx": 0, "is_high_risk_merchant": 0, "device_trust_score": 0.99},
        {"amount": 5000.0, "hour_of_day": 3, "distance_from_home_km": 500.0, "distance_from_last_tx_km": 450.0, "tx_velocity_1h": 10, "tx_amount_ratio_to_mean": 9.0, "is_foreign_tx": 1, "is_high_risk_merchant": 1, "device_trust_score": 0.05},
    ]
    df = pd.DataFrame(test_cases)
    probs = predictor.pipeline.predict_proba(df)
    assert probs.shape == (2, 2)
    # Probabilities must sum to 1
    assert abs(probs[0].sum() - 1.0) < 1e-5
    assert abs(probs[1].sum() - 1.0) < 1e-5
    # High risk case fraud prob must be greater than low risk case
    assert probs[1, 1] > probs[0, 1]


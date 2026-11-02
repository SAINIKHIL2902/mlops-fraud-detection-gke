"""Model Training and Evaluation Pipeline for Real-time Fraud Detection.

Handles:
- Data generation and stratified splitting
- Handling extreme class imbalance via class weighting
- Fitting an end-to-end preprocessing + gradient boosting pipeline
- Decision threshold calibration to balance False Positives vs False Negatives
- Exporting the production model artifact (.joblib) and rich metadata (.json)
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure root directory is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from pipeline.data_generator import (
    generate_transaction_dataset,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    beta: float = 1.2,
) -> Tuple[float, float]:
    """Find threshold that maximizes F-beta score (penalizing false negatives more).

    In fraud detection, missing a fraud (FN) is typically 5-10x more costly
    than a false alert (FP). Setting beta > 1 gives more weight to recall.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    # Avoid zero division
    numerator = (1 + beta**2) * (precisions * recalls)
    denominator = ((beta**2) * precisions) + recalls
    f_scores = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator != 0,
    )
    best_idx = np.argmax(f_scores[:-1])
    return float(thresholds[best_idx]), float(f_scores[best_idx])


def train_and_evaluate(
    output_dir: Path = Path("models"),
    n_samples: int = 60_000,
    fraud_ratio: float = 0.02,
) -> Dict[str, Any]:
    """Execute complete training and evaluation lifecycle."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating synthetic realistic transaction dataset (%d rows, %.1f%% fraud)...", n_samples, fraud_ratio * 100)
    df = generate_transaction_dataset(n_samples=n_samples, fraud_ratio=fraud_ratio, random_seed=42)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Stratified 70% Train, 15% Validation, 15% Test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    logger.info("Split sizes -> Train: %d, Val: %d, Test: %d", len(X_train), len(X_val), len(X_test))

    # Pipeline: RobustScaler handles large transaction amounts and extreme distances
    pipeline = Pipeline(
        steps=[
            ("scaler", RobustScaler()),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    class_weight="balanced",
                    max_iter=150,
                    learning_rate=0.08,
                    max_leaf_nodes=31,
                    min_samples_leaf=20,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=10,
                ),
            ),
        ]
    )

    logger.info("Training HistGradientBoostingClassifier pipeline...")
    pipeline.fit(X_train, y_train)

    # Threshold calibration on Validation set
    val_probs = pipeline.predict_proba(X_val)[:, 1]
    optimal_threshold, best_val_f_score = find_optimal_threshold(y_val, val_probs, beta=1.2)
    logger.info("Optimal threshold calibrated on validation set: %.4f (Val F-beta: %.4f)", optimal_threshold, best_val_f_score)

    # Evaluate on held-out Test set
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    test_pred_calibrated = (test_probs >= optimal_threshold).astype(int)
    test_pred_default = (test_probs >= 0.50).astype(int)

    roc_auc = float(roc_auc_score(y_test, test_probs))
    pr_auc = float(average_precision_score(y_test, test_probs))
    cm = confusion_matrix(y_test, test_pred_calibrated).tolist()
    report = classification_report(y_test, test_pred_calibrated, output_dict=True)

    metrics = {
        "test_roc_auc": round(roc_auc, 4),
        "test_pr_auc": round(pr_auc, 4),
        "test_f1_calibrated": round(f1_score(y_test, test_pred_calibrated), 4),
        "test_f1_default_0_5": round(f1_score(y_test, test_pred_default), 4),
        "confusion_matrix": {
            "true_negatives": cm[0][0],
            "false_positives": cm[0][1],
            "false_negatives": cm[1][0],
            "true_positives": cm[1][1],
        },
        "classification_report": report,
    }

    logger.info("Evaluation Results:")
    logger.info("  ROC-AUC:       %.4f", metrics["test_roc_auc"])
    logger.info("  PR-AUC:        %.4f", metrics["test_pr_auc"])
    logger.info("  F1 Score (Calibrated): %.4f", metrics["test_f1_calibrated"])
    logger.info("  Confusion Matrix: %s", metrics["confusion_matrix"])

    # Model artifact and metadata export
    model_path = output_dir / "fraud_model_pipeline.joblib"
    metadata_path = output_dir / "model_metadata.json"

    logger.info("Saving model pipeline to %s...", model_path)
    joblib.dump(pipeline, model_path)

    metadata = {
        "model_name": "realtime-fraud-detector",
        "model_version": "1.0.0",
        "algorithm": "RobustScaler + HistGradientBoostingClassifier",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "optimal_threshold": round(optimal_threshold, 4),
        "features": FEATURE_COLUMNS,
        "metrics": metrics,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved metadata to %s", metadata_path)
    return metadata


if __name__ == "__main__":
    train_and_evaluate()

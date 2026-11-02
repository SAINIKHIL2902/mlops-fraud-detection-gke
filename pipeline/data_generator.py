"""Synthetic financial transaction generator simulating realistic fraud patterns.

Simulates standard non-fraud transactions alongside realistic fraud signatures:
- Unusually high transaction amounts compared to user average
- Rapid transaction velocity (multiple transactions in last hour)
- Impossible travel distances (distance_from_last_tx_km)
- Night-time transactions (unusual hours)
- High-risk merchant categories (crypto, wire transfers, jewelry)
"""

from typing import Tuple
import numpy as np
import pandas as pd


def generate_transaction_dataset(
    n_samples: int = 50_000,
    fraud_ratio: float = 0.02,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic transaction dataset with imbalanced fraud labels.

    Args:
        n_samples: Total number of transactions to generate.
        fraud_ratio: Proportion of fraudulent transactions (default 2%).
        random_seed: Random state seed for reproducibility.

    Returns:
        pd.DataFrame containing feature columns and the binary 'is_fraud' target.
    """
    np.random.seed(random_seed)

    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # --- 1. Legitimate Transactions ---
    legit_amount = np.random.exponential(scale=65.0, size=n_legit) + 5.0
    # Occasionally legitimate users make large purchases (long tail)
    big_purchase_mask = np.random.binomial(n=1, p=0.03, size=n_legit).astype(bool)
    legit_amount[big_purchase_mask] += np.random.exponential(scale=350.0, size=big_purchase_mask.sum())

    hour_weights = np.array([
        0.01, 0.01, 0.01, 0.01, 0.01, 0.02,
        0.04, 0.06, 0.07, 0.08, 0.08, 0.07,
        0.08, 0.08, 0.07, 0.06, 0.06, 0.06,
        0.05, 0.04, 0.03, 0.03, 0.02, 0.01,
    ])
    hour_probs = hour_weights / hour_weights.sum()
    legit_hour = np.random.choice(
        np.arange(24),
        size=n_legit,
        p=hour_probs,
    )
    legit_dist_home = np.random.exponential(scale=25.0, size=n_legit)
    # Legitimate business travelers
    traveler_mask = np.random.binomial(n=1, p=0.04, size=n_legit).astype(bool)
    legit_dist_home[traveler_mask] += np.random.exponential(scale=350.0, size=traveler_mask.sum())

    legit_dist_last = np.random.exponential(scale=10.0, size=n_legit)
    legit_velocity_1h = np.random.poisson(lam=1.3, size=n_legit)
    legit_ratio_mean = np.random.lognormal(mean=0.0, sigma=0.5, size=n_legit)
    legit_foreign = np.random.binomial(n=1, p=0.06, size=n_legit)
    legit_high_risk = np.random.binomial(n=1, p=0.05, size=n_legit)
    legit_device_trust = np.clip(np.random.normal(loc=0.82, scale=0.15, size=n_legit), 0.0, 1.0)

    # --- 2. Fraudulent Transactions ---
    # Complex fraud signatures: some mimic normal users, others are high-value busts
    fraud_amount = np.random.exponential(scale=280.0, size=n_fraud) + 20.0
    # Micro-testing attacks (small card validation transactions)
    micro_mask = np.random.binomial(n=1, p=0.15, size=n_fraud).astype(bool)
    fraud_amount[micro_mask] = np.random.uniform(1.0, 5.0, size=micro_mask.sum())

    fraud_hour = np.random.choice(np.arange(24), size=n_fraud)
    fraud_dist_home = np.random.exponential(scale=120.0, size=n_fraud) + 10.0
    fraud_dist_last = np.random.exponential(scale=80.0, size=n_fraud) + 5.0
    fraud_velocity_1h = np.random.poisson(lam=3.8, size=n_fraud)
    fraud_ratio_mean = np.random.lognormal(mean=0.9, sigma=0.8, size=n_fraud)
    fraud_foreign = np.random.binomial(n=1, p=0.35, size=n_fraud)
    fraud_high_risk = np.random.binomial(n=1, p=0.38, size=n_fraud)
    fraud_device_trust = np.clip(np.random.normal(loc=0.40, scale=0.22, size=n_fraud), 0.0, 1.0)

    # Combine
    amounts = np.concatenate([legit_amount, fraud_amount])
    hours = np.concatenate([legit_hour, fraud_hour])
    dist_home = np.concatenate([legit_dist_home, fraud_dist_home])
    dist_last = np.concatenate([legit_dist_last, fraud_dist_last])
    velocity = np.concatenate([legit_velocity_1h, fraud_velocity_1h])
    ratio_mean = np.concatenate([legit_ratio_mean, fraud_ratio_mean])
    foreign = np.concatenate([legit_foreign, fraud_foreign])
    high_risk = np.concatenate([legit_high_risk, fraud_high_risk])
    device_trust = np.concatenate([legit_device_trust, fraud_device_trust])
    labels = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])

    df = pd.DataFrame(
        {
            "amount": np.round(amounts, 2),
            "hour_of_day": hours.astype(int),
            "distance_from_home_km": np.round(dist_home, 2),
            "distance_from_last_tx_km": np.round(dist_last, 2),
            "tx_velocity_1h": velocity.astype(int),
            "tx_amount_ratio_to_mean": np.round(ratio_mean, 3),
            "is_foreign_tx": foreign.astype(int),
            "is_high_risk_merchant": high_risk.astype(int),
            "device_trust_score": np.round(device_trust, 4),
            "is_fraud": labels,
        }
    )

    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return df


FEATURE_COLUMNS = [
    "amount",
    "hour_of_day",
    "distance_from_home_km",
    "distance_from_last_tx_km",
    "tx_velocity_1h",
    "tx_amount_ratio_to_mean",
    "is_foreign_tx",
    "is_high_risk_merchant",
    "device_trust_score",
]

TARGET_COLUMN = "is_fraud"

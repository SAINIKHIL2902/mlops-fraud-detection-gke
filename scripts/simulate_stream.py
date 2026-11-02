"""Real-time transaction stream simulator.

Emulates live payment gateway traffic pushing transactions into the FastAPI
microservice to demonstrate real-time scoring, alerting, and metrics scraping.
"""

import argparse
import random
import sys
import time
import uuid
import httpx


def generate_streaming_transaction() -> dict:
    """Generate a pseudo-random real-time transaction event."""
    is_fraud_scenario = random.random() < 0.15  # 15% fraud in simulation for demo

    if is_fraud_scenario:
        return {
            "transaction_id": f"sim_tx_{uuid.uuid4().hex[:8]}",
            "features": {
                "amount": round(random.uniform(250.0, 3500.0), 2),
                "hour_of_day": random.choice([1, 2, 3, 4, 23]),
                "distance_from_home_km": round(random.uniform(150.0, 800.0), 2),
                "distance_from_last_tx_km": round(random.uniform(100.0, 600.0), 2),
                "tx_velocity_1h": random.randint(5, 12),
                "tx_amount_ratio_to_mean": round(random.uniform(3.5, 9.0), 2),
                "is_foreign_tx": 1 if random.random() < 0.7 else 0,
                "is_high_risk_merchant": 1 if random.random() < 0.6 else 0,
                "device_trust_score": round(random.uniform(0.05, 0.35), 4),
            },
        }
    else:
        return {
            "transaction_id": f"sim_tx_{uuid.uuid4().hex[:8]}",
            "features": {
                "amount": round(random.uniform(5.0, 120.0), 2),
                "hour_of_day": random.choice([8, 10, 12, 14, 16, 18, 20]),
                "distance_from_home_km": round(random.uniform(0.5, 25.0), 2),
                "distance_from_last_tx_km": round(random.uniform(0.1, 8.0), 2),
                "tx_velocity_1h": random.choice([0, 1, 2]),
                "tx_amount_ratio_to_mean": round(random.uniform(0.6, 1.4), 2),
                "is_foreign_tx": 0,
                "is_high_risk_merchant": 0,
                "device_trust_score": round(random.uniform(0.75, 0.99), 4),
            },
        }


def stream_traffic(endpoint_url: str, count: int, delay_sec: float) -> None:
    """Send stream of transactions to API."""
    print(f"\n==================================================================")
    print(f"🚀 Streaming {count} real-time transactions to: {endpoint_url}")
    print(f"==================================================================\n")

    total_sent = 0
    fraud_detected = 0
    latencies = []

    with httpx.Client(timeout=5.0) as client:
        # Check health
        try:
            health = client.get(f"{endpoint_url.rsplit('/', 2)[0]}/health/ready")
            if health.status_code != 200:
                print(f"❌ Server not ready: {health.text}")
                sys.exit(1)
        except Exception as e:
            print(f"❌ Failed to connect to {endpoint_url}: {e}")
            print("Make sure the FastAPI server is running (`uvicorn app.main:app --port 8000`).")
            sys.exit(1)

        for i in range(1, count + 1):
            payload = generate_streaming_transaction()
            amount = payload["features"]["amount"]
            tx_id = payload["transaction_id"]

            t0 = time.perf_counter()
            resp = client.post(endpoint_url, json=payload)
            round_trip_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(round_trip_ms)

            if resp.status_code == 200:
                data = resp.json()
                is_fraud = data["is_fraud"]
                prob = data["fraud_probability"]
                risk = data["risk_level"]
                server_lat = data["inference_latency_ms"]

                total_sent += 1
                if is_fraud:
                    fraud_detected += 1
                    status_badge = "🚨 [CRITICAL FRAUD BLOCKED]"
                elif risk in ["MEDIUM", "HIGH"]:
                    status_badge = "⚠️  [SUSPICIOUS / STEP-UP]"
                else:
                    status_badge = "✅ [APPROVED]"

                print(
                    f"[{i:03d}/{count:03d}] {status_badge} "
                    f"ID: {tx_id} | Amt: ${amount:>7.2f} | "
                    f"Prob: {prob:.4f} | Risk: {risk:<8} | "
                    f"Infer: {server_lat:>5.2f}ms | RTT: {round_trip_ms:>5.2f}ms"
                )
            else:
                print(f"[{i:03d}/{count:03d}] ❌ Error {resp.status_code}: {resp.text}")

            if delay_sec > 0:
                time.sleep(delay_sec)

    avg_rtt = sum(latencies) / len(latencies) if latencies else 0.0
    print(f"\n------------------------------------------------------------------")
    print(f"📊 STREAM SUMMARY:")
    print(f"   Total Transactions Processed: {total_sent}")
    print(f"   Fraud Attacks Intercepted:   {fraud_detected} ({fraud_detected / max(total_sent, 1) * 100:.1f}%)")
    print(f"   Average Round-Trip Latency:  {avg_rtt:.2f}ms")
    print(f"------------------------------------------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate real-time transaction traffic")
    parser.add_argument(
        "--url",
        default="http://localhost:8000/v1/predict",
        help="Target prediction endpoint URL",
    )
    parser.add_argument("--count", type=int, default=25, help="Number of transactions to stream")
    parser.add_argument("--delay", type=float, default=0.08, help="Delay between transactions in seconds")
    args = parser.parse_args()

    stream_traffic(endpoint_url=args.url, count=args.count, delay_sec=args.delay)


import argparse
import asyncio
import os
import platform
import statistics
import time
from typing import Any

import httpx

# Realistic sample customer payload matching Pydantic CustomerInput schema
SAMPLE_PAYLOAD: dict[str, Any] = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 844.20,
}


def compute_percentiles(latencies: list[float]) -> dict[str, float]:
    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    if n == 0:
        return {}
    return {
        "min": min(sorted_lat),
        "mean": statistics.mean(sorted_lat),
        "p50": sorted_lat[int(0.50 * (n - 1))],
        "p90": sorted_lat[int(0.90 * (n - 1))],
        "p95": sorted_lat[int(0.95 * (n - 1))],
        "p99": sorted_lat[int(0.99 * (n - 1))],
        "max": max(sorted_lat),
    }


async def send_single_request(client: httpx.AsyncClient, url: str) -> float:
    t0 = time.perf_counter()
    resp = await client.post(url, json=SAMPLE_PAYLOAD, timeout=10.0)
    t1 = time.perf_counter()
    if resp.status_code != 200:
        raise RuntimeError(f"Unexpected status code {resp.status_code}: {resp.text}")
    return (t1 - t0) * 1000.0  # ms


async def run_sequential_benchmark(url: str, n_requests: int) -> list[float]:
    print(f"\n[1/3] Running Sequential Benchmark ({n_requests} requests, concurrency=1)...")
    latencies: list[float] = []
    async with httpx.AsyncClient() as client:
        # Warmup
        for _ in range(10):
            await send_single_request(client, url)

        # Timed test
        for i in range(n_requests):
            latency = await send_single_request(client, url)
            latencies.append(latency)
            if (i + 1) % 25 == 0 or (i + 1) == n_requests:
                print(f"  Progress: {i + 1}/{n_requests} requests completed")
    return latencies


async def run_concurrent_benchmark(url: str, total_requests: int, concurrency: int) -> tuple[list[float], float]:
    print(f"\n[2/3] Running Concurrent Benchmark ({total_requests} requests, concurrency={concurrency})...")
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=concurrency * 2)) as client:
        async def worker():
            async with sem:
                lat = await send_single_request(client, url)
                latencies.append(lat)

        t_start = time.perf_counter()
        tasks = [asyncio.create_task(worker()) for _ in range(total_requests)]
        await asyncio.gather(*tasks)
        t_total = time.perf_counter() - t_start

    return latencies, t_total


def print_system_info():
    print("=" * 65)
    print("            INFERENCE PERFORMANCE BENCHMARK REPORT             ")
    print("=" * 65)
    print(f"Platform:       {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python Version: {platform.python_version()}")
    print(f"CPU Cores:      {os.cpu_count()} logical cores")
    print("=" * 65)


def print_results(
    seq_stats: dict[str, float],
    conc_stats: dict[str, float],
    total_conc_requests: int,
    total_time: float,
    concurrency: int,
):
    throughput = total_conc_requests / total_time if total_time > 0 else 0.0

    print("\n" + "=" * 65)
    print("                      BENCHMARK RESULTS                        ")
    print("=" * 65)
    print(f"{'Metric':<25} | {'Sequential (c=1)':<16} | {'Concurrent (c=' + str(concurrency) + ')':<16}")
    print("-" * 65)
    print(f"{'Mean Latency (ms)':<25} | {seq_stats['mean']:>14.2f} ms | {conc_stats['mean']:>14.2f} ms")
    print(f"{'P50 Latency (ms)':<25} | {seq_stats['p50']:>14.2f} ms | {conc_stats['p50']:>14.2f} ms")
    print(f"{'P90 Latency (ms)':<25} | {seq_stats['p90']:>14.2f} ms | {conc_stats['p90']:>14.2f} ms")
    print(f"{'P95 Latency (ms)':<25} | {seq_stats['p95']:>14.2f} ms | {conc_stats['p95']:>14.2f} ms")
    print(f"{'P99 Latency (ms)':<25} | {seq_stats['p99']:>14.2f} ms | {conc_stats['p99']:>14.2f} ms")
    print(f"{'Min Latency (ms)':<25} | {seq_stats['min']:>14.2f} ms | {conc_stats['min']:>14.2f} ms")
    print(f"{'Max Latency (ms)':<25} | {seq_stats['max']:>14.2f} ms | {conc_stats['max']:>14.2f} ms")
    print("-" * 65)
    print(f"{'Throughput (req/s)':<25} | {1000.0 / seq_stats['mean']:>14.2f} req/s | {throughput:>14.2f} req/s")
    print(f"{'Total Elapsed Time':<25} | {'-':>16} | {total_time:>14.2f} s")
    print("=" * 65)


async def main():
    parser = argparse.ArgumentParser(description="Inference Performance Benchmark")
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict", help="Predict endpoint URL")
    parser.add_argument("--n-sequential", type=int, default=100, help="Number of sequential requests")
    parser.add_argument("--n-concurrent", type=int, default=200, help="Number of concurrent requests")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent client workers")
    args = parser.parse_args()

    print_system_info()
    print(f"Target URL: {args.url}")

    # Step 1: Healthcheck to ensure API is online
    async with httpx.AsyncClient() as client:
        health_url = args.url.replace("/predict", "/health")
        try:
            health_resp = await client.get(health_url, timeout=5.0)
            if health_resp.status_code != 200:
                print(f"Warning: Healthcheck failed with status {health_resp.status_code}")
            else:
                print(f"Service Status: Online ({health_resp.json()})")
        except Exception as e:  # noqa: BLE001
            print(f"\n[ERROR] Could not connect to API at {health_url}: {e}")
            print("Make sure your FastAPI server is running with: uvicorn src.serving.main:app --port 8000")
            return

    # Step 2: Sequential Benchmark
    seq_latencies = await run_sequential_benchmark(args.url, args.n_sequential)
    seq_stats = compute_percentiles(seq_latencies)

    # Step 3: Concurrent Benchmark
    conc_latencies, total_time = await run_concurrent_benchmark(
        args.url, args.n_concurrent, args.concurrency
    )
    conc_stats = compute_percentiles(conc_latencies)

    # Step 4: Report Results
    print("\n[3/3] Aggregating results...")
    print_results(seq_stats, conc_stats, args.n_concurrent, total_time, args.concurrency)


if __name__ == "__main__":
    asyncio.run(main())

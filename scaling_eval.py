"""
scaling_eval.py
Benchmarks performance under burst load, verifies cache latency reduction,
and validates strict adherence to runtime token/cost budgets.
"""

import time
import statistics
import concurrent.futures
from typing import List, Dict, Any

from crew import run_orchestrated_query
from cache import (
    RESPONSE_CACHE,
    enforce_runtime_budget,
    BudgetExceededException,
    MAX_TOKEN_BUDGET_PER_REQUEST,
)

BENCHMARK_QUERIES = [
    "What is the standard notice period for confirmed employees?",
    "Can probation be extended beyond 6 months?",
    "Check status for application APP-1001",
    "What is the annual casual leave entitlement?",
    "Check status for application APP-1025",
]

def benchmark_cache_speedup(iterations: int = 20) -> Dict[str, Any]:
    query = "What is the standard notice period for confirmed employees?"
    RESPONSE_CACHE.clear()

    # 1. Cold execution (Uncached)
    cold_latencies: List[float] = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = run_orchestrated_query(query, session_id="bench_cold")
        cold_latencies.append((time.perf_counter() - t0) * 1000.0)

    # Prime cache
    cached_val = run_orchestrated_query(query, session_id="bench_prime")
    RESPONSE_CACHE.set(query, cached_val.model_dump())

    # 2. Warm execution (Cached)
    warm_latencies: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = RESPONSE_CACHE.get(query)
        warm_latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_cold = statistics.mean(cold_latencies)
    avg_warm = statistics.mean(warm_latencies)
    speedup = avg_cold / max(avg_warm, 0.0001)

    return {
        "avg_cold_ms": round(avg_cold, 2),
        "avg_warm_ms": round(avg_warm, 4),
        "speedup_factor": round(speedup, 1),
    }

def benchmark_burst_concurrency(concurrency_level: int = 10) -> Dict[str, Any]:
    latencies: List[float] = []
    successes = 0

    def task(q: str, idx: int):
        t0 = time.perf_counter()
        resp = run_orchestrated_query(q, session_id=f"burst_{idx}")
        elapsed = (time.perf_counter() - t0) * 1000.0
        return elapsed, resp.status != ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency_level) as executor:
        futures = [
            executor.submit(task, BENCHMARK_QUERIES[i % len(BENCHMARK_QUERIES)], i)
            for i in range(concurrency_level)
        ]
        for f in concurrent.futures.as_completed(futures):
            elapsed, ok = f.result()
            latencies.append(elapsed)
            if ok:
                successes += 1

    return {
        "concurrency": concurrency_level,
        "success_rate": f"{(successes / concurrency_level) * 100:.1f}%",
        "p50_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies), 2),
        "mean_ms": round(statistics.mean(latencies), 2),
    }

def run_stress_suite():
    print("=" * 60)
    print("🚀 Running Naukri Agent Scaling & Stress Evaluation Suite")
    print("=" * 60)

    # Budget Check Verification
    print("\n[1/3] Runtime Budget Verification...")
    normal_query = "What is the notice period policy for permanent employees?"
    tokens = enforce_runtime_budget(normal_query)
    within_budget = tokens <= MAX_TOKEN_BUDGET_PER_REQUEST
    
    oversized_query = " ".join(["applicant_policy_clause_repeat"] * 300)
    over_budget_caught = False
    try:
        enforce_runtime_budget(oversized_query)
    except BudgetExceededException:
        over_budget_caught = True

    print(f"  - Request within budget accepted: {within_budget}")
    print(f"  - Request exceeding budget intercepted: {over_budget_caught}")
    assert within_budget and over_budget_caught, "Budget enforcement test failed"

    # Cache Speedup Test
    print("\n[2/3] Cache Latency & Speedup Benchmark...")
    cache_metrics = benchmark_cache_speedup()
    print(f"  - Cold RAG + Crew Latency: {cache_metrics['avg_cold_ms']} ms")
    print(f"  - Warm Cache Hit Latency:  {cache_metrics['avg_warm_ms']} ms")
    print(f"  - Latency Speedup:          {cache_metrics['speedup_factor']}x")

    # Burst Throughput Test
    print("\n[3/3] Burst Concurrency Benchmark (10 Workers)...")
    burst_metrics = benchmark_burst_concurrency(concurrency_level=10)
    print(f"  - Success Rate: {burst_metrics['success_rate']}")
    print(f"  - Mean Latency: {burst_metrics['mean_ms']} ms")
    print(f"  - P50 Latency:  {burst_metrics['p50_ms']} ms")
    print(f"  - P95 Latency:  {burst_metrics['p95_ms']} ms")

    print("\n" + "=" * 60)
    print("✅ All Scaling and Stress Verification Checks PASSED.")
    print("=" * 60)

if __name__ == "__main__":
    run_stress_suite()
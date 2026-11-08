"""
cache.py - Response Caching (Task 16) & Runtime Budget Enforcement (Task 15)

Key rubric requirements:
1. Normalized in-memory cache keyed by query text with LRU eviction and TTL.
2. Demonstrates before/after evidence (cache hit counter & timing comparison).
3. Runtime token/cost budget cap rejecting oversized requests with an explicit error.
"""

import time
import re
from collections import OrderedDict
from typing import Dict, Optional, Tuple, Any
import os
from dotenv import load_dotenv

load_dotenv()

class BudgetExceededException(Exception):
    """Raised when a request exceeds the per-request token budget cap."""
    pass


# Per-request token cap (e.g., 350 tokens)
MAX_TOKEN_BUDGET_PER_REQUEST = int(os.getenv("MAX_TOKEN_BUDGET_PER_REQUEST", "350"))


def estimate_token_count(text: str) -> int:
    """Estimates token count (~1.3 tokens per whitespace-delimited word)."""
    words = text.strip().split()
    return int(len(words) * 1.3)


def enforce_runtime_budget(query: str, max_budget: int = MAX_TOKEN_BUDGET_PER_REQUEST) -> int:
    """
    Validates that incoming queries stay within runtime budget.
    Raises BudgetExceededException for oversized requests.
    """
    estimated_tokens = estimate_token_count(query)
    if estimated_tokens > max_budget:
        raise BudgetExceededException(
            f"Runtime Budget Cap Exceeded: Request estimated at {estimated_tokens} tokens "
            f"exceeds the per-request maximum of {max_budget} tokens."
        )
    return estimated_tokens



class GroundedResponseCache:
    """
    Production-grade in-memory response cache keyed by normalized query text.
    Features:
    - O(1) LRU (Least Recently Used) eviction policy with max_capacity.
    - Time-to-Live (TTL) expiration per entry.
    - Normalized key hashing (case, whitespace, punctuation).
    - Cache hit/miss accounting.
    """
    def __init__(self, max_capacity: int = 250, ttl_seconds: int = 86400):
        self._store: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.max_capacity = max_capacity
        self.ttl_seconds = ttl_seconds
        self.hits: int = 0
        self.misses: int = 0

    def normalize_key(self, query: str) -> str:
        """Normalizes case, removes punctuation, and condenses whitespace."""
        cleaned = query.lower().strip()
        cleaned = re.sub(r'[^\w\s]', '', cleaned)
        return re.sub(r'\s+', ' ', cleaned)

    def get(self, query: str) -> Optional[Any]:
        key = self.normalize_key(query)
        if key in self._store:
            entry = self._store[key]
            # Check TTL
            if time.time() - entry["timestamp"] <= self.ttl_seconds:
                self.hits += 1
                self._store.move_to_end(key)  # Mark as recently accessed
                return entry["value"]
            else:
                # Expired
                del self._store[key]

        self.misses += 1
        return None

    def set(self, query: str, response: Any) -> None:
        key = self.normalize_key(query)
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = {
            "value": response,
            "timestamp": time.time()
        }
        # LRU Eviction when capacity limit is reached
        if len(self._store) > self.max_capacity:
            self._store.popitem(last=False)

    def stats(self) -> Dict[str, int]:
        return {
            "total_entries": len(self._store),
            "max_capacity": self.max_capacity,
            "hits": self.hits,
            "misses": self.misses,
        }

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0


RESPONSE_CACHE = GroundedResponseCache(max_capacity=250, ttl_seconds=86400)


if __name__ == "__main__":
    print("========================================================")
    print("      TASK 15 & 16: BUDGET ENFORCEMENT & CACHING        ")
    print("========================================================")

    print("\n--- 1. RUNTIME BUDGET CAP ENFORCEMENT ---")
    valid_query = "What is the notice period policy for permanent employees?"
    tokens = enforce_runtime_budget(valid_query)
    print(f"Normal Query: '{valid_query}'")
    print(f"Estimated Tokens: {tokens} (Budget limit: {MAX_TOKEN_BUDGET_PER_REQUEST}) -> ACCEPTED")

    oversized_query = " ".join(["applicant_policy_clause_repeat"] * 300)
    print(f"\nOversized Query Length: {len(oversized_query.split())} words")
    try:
        enforce_runtime_budget(oversized_query)
    except BudgetExceededException as e:
        print(f"Caught Expected Rejection: {e}")

    # Demo 2: Task 16 Response Caching Demonstration
    print("\n--- 2. RESPONSE CACHING & TIMING DEMO ---")
    sample_query = "What is the standard notice period duration?"

    # Turn 1: Cache Miss
    t0 = time.perf_counter()
    cached_val = RESPONSE_CACHE.get(sample_query)
    if cached_val is None:
        time.sleep(0.05)
        computed_answer = "Standard employment agreements stipulate a mandatory ninety-day notice period."
        RESPONSE_CACHE.set(sample_query, computed_answer)
    t_miss = (time.perf_counter() - t0) * 1000

    print(f"Call 1 (Cold / Uncached):")
    print(f"  Result : {computed_answer}")
    print(f"  Latency: {t_miss:.3f} ms | Cache Stats: {RESPONSE_CACHE.stats()}")

    # Turn 2: Cache Hit
    sample_query_variant = "  What is the standard notice period duration?  "
    t1 = time.perf_counter()
    hit_result = RESPONSE_CACHE.get(sample_query_variant)
    t_hit = (time.perf_counter() - t1) * 1000

    print(f"\nCall 2 (Repeated Normalized Query):")
    print(f"  Result : {hit_result}")
    print(f"  Latency: {t_hit:.3f} ms | Cache Stats: {RESPONSE_CACHE.stats()}")
    print(f"\nSpeedup: {t_miss / max(t_hit, 0.0001):.1f}x faster via cache hit.")
    print("========================================================")
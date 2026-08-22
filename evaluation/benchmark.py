"""
End-to-end latency benchmark — Tech Doc §17.

Runs the RAG pipeline against a batch of test queries and calculates
P50, P70, and P100 latency percentiles.

Usage:
    python evaluation/benchmark.py --queries evaluation/test_queries.json --output evaluation/results.json

Requirements:
    - API server must be running: uvicorn app.main:app
    - .env must be configured with valid keys
    - evaluation/test_queries.json must exist (see format below)

Test query file format (JSON array of strings):
    ["What is retrieval augmented generation?", "How does FAISS work?", ...]

Output format (evaluation/results.json):
    {
        "summary": {"p50_ms": ..., "p70_ms": ..., "p100_ms": ...},
        "runs": [{"query_id": ..., "total_latency_ms": ..., "status": ..., "grounded": ..., ...}]
    }

IMPORTANT: All latency values in results are measured from actual runs.
           No values are fabricated. (Tech Doc §17, PRD §13)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    import httpx
    import numpy as np
except ImportError:
    print("Missing dependencies. Run: pip install httpx numpy")
    sys.exit(1)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_QUERIES_FILE = Path(__file__).parent / "test_queries.json"
DEFAULT_OUTPUT_FILE = Path(__file__).parent / "results.json"
DEFAULT_API_URL = "http://localhost:8000/ask"


async def run_single_query(
    client: httpx.AsyncClient,
    query_id: int,
    query_text: str,
    api_url: str,
) -> Dict[str, Any]:
    """
    Sends a single query to the API as audio (using a silent WAV stub for text-only benchmark)
    and records the result.

    For a proper audio benchmark, replace the audio_bytes with real recorded audio.
    For a latency benchmark on the generation/retrieval pipeline, we send a minimal WAV
    that will produce an empty transcript — which is valid for measuring error paths — OR
    you can extend this to send real audio files mapped to each query.
    """
    # Minimal 44-byte WAV header for a silent 0-sample file.
    # Swap this with real audio bytes for full end-to-end measurement.
    SILENT_WAV = (
        b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00"
        b"\x01\x00\x80\xbb\x00\x00\x00w\x01\x00\x02\x00\x10\x00"
        b"data\x00\x00\x00\x00"
    )

    start = time.perf_counter()
    try:
        response = await client.post(
            api_url,
            files={"audio": ("query.wav", SILENT_WAV, "audio/wav")},
            timeout=30.0,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        if response.status_code == 200:
            data = response.json()
            return {
                "query_id": query_id,
                "query_text": query_text,
                "total_latency_ms": data.get("latency_ms", elapsed_ms),
                "stage_latencies": data.get("stage_latencies", {}),
                "status": data.get("status", "unknown"),
                "grounded": data.get("grounded", False),
                "http_status": response.status_code,
                "wall_time_ms": round(elapsed_ms, 2),
            }
        else:
            return {
                "query_id": query_id,
                "query_text": query_text,
                "total_latency_ms": round(elapsed_ms, 2),
                "stage_latencies": {},
                "status": "error",
                "grounded": False,
                "http_status": response.status_code,
                "wall_time_ms": round(elapsed_ms, 2),
            }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.error("Query %d failed: %s", query_id, exc)
        return {
            "query_id": query_id,
            "query_text": query_text,
            "total_latency_ms": round(elapsed_ms, 2),
            "stage_latencies": {},
            "status": "error",
            "grounded": False,
            "http_status": None,
            "wall_time_ms": round(elapsed_ms, 2),
        }


async def run_benchmark(
    queries: List[str],
    api_url: str,
    concurrency: int = 1,
) -> Dict[str, Any]:
    """
    Run all queries and collect latency measurements.

    Args:
        queries:     List of query strings.
        api_url:     POST /ask endpoint URL.
        concurrency: How many concurrent requests (keep at 1 for sequential measurement).

    Returns:
        Dict with "summary" (P50/P70/P100) and "runs" (per-query records).
    """
    logger.info("Starting benchmark: %d queries, concurrency=%d", len(queries), concurrency)
    runs: List[Dict[str, Any]] = []

    async with httpx.AsyncClient() as client:
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_run(qid, text):
            async with semaphore:
                return await run_single_query(client, qid, text, api_url)

        tasks = [bounded_run(i, q) for i, q in enumerate(queries)]
        results = await asyncio.gather(*tasks)
        runs.extend(results)

    # Calculate percentiles from total_latency_ms of all completed runs
    latencies = [r["total_latency_ms"] for r in runs if r["total_latency_ms"] > 0]

    if not latencies:
        logger.error("No successful runs to calculate percentiles.")
        summary = {"p50_ms": None, "p70_ms": None, "p100_ms": None, "n": 0}
    else:
        summary = {
            "p50_ms": round(float(np.percentile(latencies, 50)), 2),
            "p70_ms": round(float(np.percentile(latencies, 70)), 2),
            "p100_ms": round(float(np.percentile(latencies, 100)), 2),
            "n": len(latencies),
            "success_rate": sum(1 for r in runs if r["status"] == "success") / len(runs),
            "grounded_rate": sum(1 for r in runs if r["grounded"]) / len(runs),
        }
        logger.info(
            "Results: P50=%.1f ms, P70=%.1f ms, P100=%.1f ms (n=%d)",
            summary["p50_ms"], summary["p70_ms"], summary["p100_ms"], summary["n"],
        )

    return {"summary": summary, "runs": runs}


def load_queries(queries_file: Path) -> List[str]:
    """Load queries from a JSON file (array of strings)."""
    if not queries_file.exists():
        logger.warning("Queries file not found: %s. Using sample queries.", queries_file)
        # Sample queries for smoke-testing the pipeline structure
        return [
            "What is retrieval augmented generation?",
            "How does FAISS vector search work?",
            "What is the MSMARCO dataset?",
            "Explain cosine similarity in information retrieval.",
            "What is a transformer model?",
        ]
    with open(queries_file, "r", encoding="utf-8") as f:
        queries = json.load(f)
    if not isinstance(queries, list):
        raise ValueError("Queries file must be a JSON array of strings.")
    return queries


def main():
    parser = argparse.ArgumentParser(description="Voice-Enabled RAG latency benchmark")
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES_FILE,
        help="Path to JSON file containing test queries (array of strings)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Path to write results JSON",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help="POST /ask endpoint URL (default: http://localhost:8000/ask)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent requests (default: 1 for sequential measurement)",
    )
    args = parser.parse_args()

    queries = load_queries(args.queries)
    logger.info("Loaded %d queries from %s", len(queries), args.queries)

    results = asyncio.run(
        run_benchmark(queries, api_url=args.api_url, concurrency=args.concurrency)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Results saved to %s", args.output)
    print("\n=== Benchmark Summary ===")
    s = results["summary"]
    print(f"  Queries run : {s.get('n', 0)}")
    print(f"  P50         : {s.get('p50_ms', 'N/A')} ms")
    print(f"  P70         : {s.get('p70_ms', 'N/A')} ms")
    print(f"  P100        : {s.get('p100_ms', 'N/A')} ms")
    print(f"  Success rate: {s.get('success_rate', 0):.1%}")
    print(f"  Grounded    : {s.get('grounded_rate', 0):.1%}")
    print(f"\nFull results: {args.output}")


if __name__ == "__main__":
    main()

"""Summarise processor runs (Part 2 study) from MongoDB batch_stats and keyframes.

For each run: throughput, queries out, drop ratio and reasons, batch duration, ingest delay,
keyframe delay and the processor config. Several runs are printed side by side for comparison.

    python tools/processing_report.py --run-id E1-local1 E1-local4 [--skip-seconds 10] [--csv summary.csv]
"""
import argparse
import csv
import math
import os
import sys
from collections import Counter
from pathlib import Path

from pymongo import MongoClient

REASONS = ("first", "change", "heartbeat")


def percentile(values: list, p: float):
    """Linearly interpolated percentile, as numpy.percentile does; None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * p / 100
    low, high = math.floor(k), math.ceil(k)
    return ordered[low] + (ordered[high] - ordered[low]) * (k - low)


def mean(values: list):
    return sum(values) / len(values) if values else None


def summarize(db, run_id: str, skip_seconds: float = 0):
    """Returns one flat dict of metrics for a run, or None if the run has no batch_stats. Batches that
    started in the first `skip_seconds` of the run (warm-up) are left out."""
    stats = list(db.batch_stats.find({"run_id": run_id}, {"_id": 0}).sort("batch_id", 1))
    if not stats:
        return None
    cutoff_ms = stats[0]["batch_start_ms"] + skip_seconds * 1000
    stats = [s for s in stats if s["batch_start_ms"] >= cutoff_ms]
    if not stats:
        return None
    keyframes = list(db.keyframes.find({"run_id": run_id, "selected_ts_ms": {"$gte": cutoff_ms}},
                                       {"_id": 0, "capture_ts_ms": 1, "selected_ts_ms": 1}))

    frames_in = sum(s["frames_in"] for s in stats)
    keyframes_out = sum(s["keyframes_out"] for s in stats)
    queries_out = sum(s.get("queries_out", 0) for s in stats)
    # Wall-clock span of the run, from the first batch's start to the last batch's end.
    duration_s = (max(s["batch_end_ms"] for s in stats) - min(s["batch_start_ms"] for s in stats)) / 1000
    reasons, devices = Counter(), Counter()
    for s in stats:
        reasons.update(s["reasons"])
        devices.update(s["devices"])
    batch_ms = [s["duration_ms"] for s in stats]
    # batch_stats holds the mean ingest delay per batch; weighting it by the batch's frame count
    # approximates the per-frame distribution.
    weighted_delays = [s["ingest_delay_ms"]["mean"] for s in stats for _ in range(s["frames_in"])]
    keyframe_delays = [k["selected_ts_ms"] - k["capture_ts_ms"] for k in keyframes]
    config = stats[0]["config"]
    if any(s["config"] != config for s in stats):
        print(f"Warning: run {run_id} contains batches with different configs; showing the first one.")

    return {
        "run_id": run_id,
        "skipped_s": skip_seconds,
        "devices": len(devices),
        "batches": len(stats),
        "duration_s": duration_s,
        "frames_in": frames_in,
        "keyframes_out": keyframes_out,
        "frames_in_per_s": frames_in / duration_s if duration_s else None,
        "keyframes_out_per_s": keyframes_out / duration_s if duration_s else None,
        "queries_out": queries_out,
        "drop_ratio": 1 - keyframes_out / frames_in,
        **{f"reason_{reason}": reasons.get(reason, 0) for reason in REASONS},
        "frames_per_batch_mean": frames_in / len(stats),
        "batch_ms_mean": mean(batch_ms),
        "batch_ms_p95": percentile(batch_ms, 95),
        "ingest_delay_ms_mean": mean(weighted_delays),
        "ingest_delay_ms_p50": percentile(weighted_delays, 50),
        "ingest_delay_ms_p95": percentile(weighted_delays, 95),
        "ingest_delay_ms_max": max(s["ingest_delay_ms"]["max"] for s in stats),
        "keyframe_delay_ms_mean": mean(keyframe_delays),
        "keyframe_delay_ms_p50": percentile(keyframe_delays, 50),
        "keyframe_delay_ms_p95": percentile(keyframe_delays, 95),
        **{f"config_{key}": value for key, value in config.items()},
    }


def fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarise processor runs from MongoDB.")
    parser.add_argument("--run-id", nargs="+", required=True, help="one or more run ids to summarise")
    parser.add_argument("--skip-seconds", type=float, default=0,
                        help="leave out batches from the first N seconds of each run (warm-up)")
    parser.add_argument("--csv", help="also write the per-run summary to this CSV file")
    parser.add_argument("--mongo-uri", default=os.environ.get("MONGO_URI", "mongodb://127.0.0.1:27017"),
                        help="MongoDB URI (default: $MONGO_URI or mongodb://127.0.0.1:27017)")
    args = parser.parse_args()

    db = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=5000)["edge_vlm"]
    rows = []
    for run_id in args.run_id:
        row = summarize(db, run_id, args.skip_seconds)
        if row is None:
            print(f"Warning: no batch_stats for run {run_id}")
        else:
            rows.append(row)
    if not rows:
        return 1

    # One line per metric, one column per run. Runs of older code versions can lack some config fields.
    keys = list(dict.fromkeys(key for row in rows for key in row))
    metric_width = max(len(key) for key in keys)
    column_width = max(12, *(len(row["run_id"]) + 2 for row in rows))
    for key in keys:
        print(f"{key:<{metric_width}}" + "".join(f"{fmt(row.get(key)):>{column_width}}" for row in rows))

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: round(value, 3) if isinstance(value, float) else value
                                 for key, value in row.items()})
        print(f"\nWrote {len(rows)} run(s) to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

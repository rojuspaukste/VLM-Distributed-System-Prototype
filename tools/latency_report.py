"""Summarise the VLM answers (evaluation) from MongoDB: latency per run and per number of context frames.

For each (run_id, context_frames_used): the number of answers; the mean, p50 and p95 of the end-to-end,
queue and inference latency; how far back the images reached (context_span_s: oldest context frame to
current frame); and the peak VRAM. Without --run-id, every run in the collection is shown.

    python tools/latency_report.py [--run-id E2-ctx0 E2-ctx4] [--csv results/latency.csv]
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

from pymongo import MongoClient

from processing_report import fmt, mean, percentile  # same helpers as the processing report

LATENCIES = ("end_to_end", "queue", "inference")


def summarize(answers: list) -> list:
    """One row per (run_id, context_frames_used), in the order the runs were requested."""
    groups = defaultdict(list)
    for answer in answers:
        groups[(answer["run_id"], answer["context_frames_used"])].append(answer)
    rows = []
    for (run_id, context_frames), group in groups.items():
        row = {"run_id": run_id, "context_frames": context_frames, "answers": len(group)}
        for name in LATENCIES:
            values = [a["latency_ms"][name] for a in group]
            row[f"{name}_ms_mean"] = mean(values)
            row[f"{name}_ms_p50"] = percentile(values, 50)
            row[f"{name}_ms_p95"] = percentile(values, 95)
        spans = [a["context_span_s"] for a in group if "context_span_s" in a]  # older answers lack it
        row["context_span_s_mean"] = mean(spans)
        row["context_span_s_min"] = min(spans) if spans else None
        row["peak_vram_mb"] = max(a["peak_vram_mb"] for a in group)
        row["model"] = group[0]["model"]
        row["max_pixels"] = group[0]["max_pixels"]
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarise VLM answer latency from MongoDB.")
    parser.add_argument("--run-id", nargs="+", help="run ids to summarise (default: all runs)")
    parser.add_argument("--csv", help="also write the summary to this CSV file")
    parser.add_argument("--mongo-uri", default=os.environ.get("MONGO_URI", "mongodb://127.0.0.1:27017"),
                        help="MongoDB URI (default: $MONGO_URI or mongodb://127.0.0.1:27017)")
    args = parser.parse_args()

    db = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=5000)["edge_vlm"]
    query = {"run_id": {"$in": args.run_id}} if args.run_id else {}
    answers = list(db.answers.find(query, {"_id": 0}).sort("timestamps.inference_end_ms", 1))
    if args.run_id:
        missing = set(args.run_id) - {a["run_id"] for a in answers}
        for run_id in sorted(missing):
            print(f"Warning: no answers for run {run_id}")
        answers.sort(key=lambda a: args.run_id.index(a["run_id"]))  # keep the requested run order
    rows = summarize(answers)
    if not rows:
        print("No answers found.")
        return 1

    # One line per metric, one column per (run, context frames).
    metric_width = max(len(key) for key in rows[0])
    column_width = max(12, *(len(str(row["run_id"])) + 2 for row in rows), *(len(row["model"]) + 2 for row in rows))
    for key in rows[0]:
        print(f"{key:<{metric_width}}" + "".join(f"{fmt(row[key]):>{column_width}}" for row in rows))

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({key: round(value, 1) if isinstance(value, float) else value
                                 for key, value in row.items()})
        print(f"\nWrote {len(rows)} row(s) to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

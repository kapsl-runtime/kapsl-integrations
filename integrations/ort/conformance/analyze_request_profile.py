#!/usr/bin/env python3
"""Summarize bounded engine/adapter timing logs; never apply qualification gates."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

MARKER = "KAPSL_REQUEST_PROFILE "


def statistics_us(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "samples": len(values),
        "mean_us": statistics.mean(values) / 1000,
        "median_us": statistics.median(values) / 1000,
        "p95_us": ordered[int((len(ordered) - 1) * 0.95)] / 1000,
        "max_us": ordered[-1] / 1000,
    }


def analyze(path: Path, warmups: int, requests_per_trial: int) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    seen: set[tuple] = set()
    for line in path.read_text().splitlines():
        if MARKER not in line:
            continue
        # Accept plain logs and a structured logger's JSON message field.
        if line.startswith("{"):
            outer = json.loads(line)
            line = outer.get("message", outer.get("fields", {}).get("message", ""))
        if MARKER not in line:
            continue
        report = json.loads(line.split(MARKER, 1)[1])
        if report.get("schema_version") != 1:
            raise ValueError("unsupported request profile schema")
        identity = (
            report["component"],
            report["process_id"],
            report["model_id"],
            report["replica_id"],
            report["origin_unix_ns"],
        )
        for record in report["records"]:
            unique = (*identity, record["sequence"])
            if unique in seen:
                raise ValueError("duplicate request profile sample")
            seen.add(unique)
            for field in ("sequence", "request_id", "started_ns", "wall_ns"):
                if type(record[field]) is not int or record[field] < 0:
                    raise ValueError(f"invalid {field} in request profile")
            if record["sequence"] >= report["sample_limit"]:
                raise ValueError("request profile exceeds its collection limit")
            if any(
                type(p["wall_ns"]) is not int or p["wall_ns"] < 0
                for p in record["phases"]
            ):
                raise ValueError("invalid phase duration")
            if sum(p["wall_ns"] for p in record["phases"]) > record["wall_ns"]:
                raise ValueError("phase durations exceed enclosing operation")
            groups[(*identity, record["operation"])].append(record)
    if not groups:
        raise ValueError(
            f"no request profiles in {path}; unload the model before stopping the process"
        )
    results = []
    for identity, records in sorted(groups.items()):
        records.sort(key=lambda record: record["sequence"])
        windows: dict[int, list[dict]] = defaultdict(list)
        for ordinal, record in enumerate(records):
            window = (
                -1 if ordinal < warmups else (ordinal - warmups) // requests_per_trial
            )
            windows[window].append(record)
        for window, selected in sorted(windows.items()):
            phases: dict[str, list[int]] = defaultdict(list)
            for record in selected:
                for phase in record["phases"]:
                    phases[phase["name"]].append(phase["wall_ns"])
            results.append(
                {
                    "component": identity[0],
                    "process_id": identity[1],
                    "model_id": identity[2],
                    "replica_id": identity[3],
                    "operation": identity[5],
                    "window": "warmup" if window < 0 else window + 1,
                    "wall": statistics_us([record["wall_ns"] for record in selected]),
                    "phases": {
                        name: statistics_us(values)
                        for name, values in sorted(phases.items())
                    },
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--warmups", type=int, default=40)
    parser.add_argument("--requests-per-trial", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmups < 0 or args.requests_per_trial <= 0:
        parser.error("warmups must be nonnegative and requests-per-trial positive")
    result = {
        "diagnostic_only": True,
        "qualification_passed": False,
        "note": "Windows group operation order; interrupted/cancelled or concurrent workloads need explicit request-ID correlation. Nested phases must not be added to their parent durations.",
        "logs": {
            str(path): analyze(path, args.warmups, args.requests_per_trial)
            for path in args.logs
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

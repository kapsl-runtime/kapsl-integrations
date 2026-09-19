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


def read_profiles(path: Path) -> dict[tuple, list[dict]]:
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
        origin = report["origin_unix_ns"]
        if (
            not isinstance(origin, str)
            or not origin.isascii()
            or not origin.isdecimal()
        ):
            raise ValueError("invalid clock origin")
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
    return groups


def analyze(path: Path, warmups: int, requests_per_trial: int) -> list[dict]:
    groups = read_profiles(path)
    results = []
    for identity, records in sorted(groups.items()):
        records.sort(key=lambda record: record["sequence"])
        windows: dict[int, list[dict]] = defaultdict(list)
        for ordinal, record in enumerate(records):
            window = (
                -1 if ordinal < warmups else (ordinal - warmups) // requests_per_trial
            )
            # Allocator sequence numbers count callbacks, not inferences.
            if identity[0] == "engine.native.allocator":
                window = -2
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
                    "window": (
                        "unassigned_callbacks"
                        if window == -2
                        else "warmup"
                        if window < 0
                        else window + 1
                    ),
                    "wall": statistics_us([record["wall_ns"] for record in selected]),
                    "phases": {
                        name: statistics_us(values)
                        for name, values in sorted(phases.items())
                    },
                }
            )
    return results


def _interval(identity: tuple, record: dict) -> tuple[int, int]:
    origin = int(identity[4])
    if origin < 0:
        raise ValueError("invalid clock origin")
    start = origin + record["started_ns"]
    return start, start + record["wall_ns"]


def _union_ns(intervals: list[tuple[int, int]]) -> int:
    total = 0
    end = None
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end if end is not None else start))
        end = max(stop, end if end is not None else stop)
    return total


def correlate_allocator(path: Path) -> dict:
    """Associate callbacks only with a unique enclosing native infer span.

    Nonzero IDs must also match; zero-ID synchronization callbacks can only use
    timestamp containment. Retain ambiguity/lifecycle gaps instead of guessing.
    """
    groups = read_profiles(path)
    anchors = []
    callbacks = []
    for identity, records in groups.items():
        destination = (
            anchors
            if identity[0] == "engine.native" and identity[5] == "infer"
            else callbacks
            if identity[0] == "engine.native.allocator"
            else None
        )
        if destination is not None:
            for record in records:
                destination.append(
                    {
                        "identity": identity,
                        "record": record,
                        "interval": _interval(identity, record),
                        "callbacks": [],
                    }
                )
    by_owner: dict[tuple, list[dict]] = defaultdict(list)
    by_request: dict[tuple, list[dict]] = defaultdict(list)
    for anchor in anchors:
        owner = anchor["identity"][1:4]
        by_owner[owner].append(anchor)
        by_request[(*owner, anchor["record"]["request_id"])].append(anchor)
    unmatched = []
    for callback in callbacks:
        identity, record = callback["identity"], callback["record"]
        start, end = callback["interval"]
        owner = identity[1:4]
        candidates = (
            by_request[(*owner, record["request_id"])]
            if record["request_id"]
            else by_owner[owner]
        )
        matching = [
            anchor
            for anchor in candidates
            if anchor["interval"][0] <= start and end <= anchor["interval"][1]
        ]
        if len(matching) == 1:
            matching[0]["callbacks"].append(callback)
        else:
            unmatched.append(
                {
                    "process_id": identity[1],
                    "model_id": identity[2],
                    "replica_id": identity[3],
                    "request_id": record["request_id"],
                    "operation": record["operation"],
                    "sequence": record["sequence"],
                    "reason": "ambiguous_enclosing_infer"
                    if matching
                    else "no_enclosing_infer",
                }
            )
    requests = []
    for anchor in sorted(
        anchors, key=lambda row: (row["identity"], row["record"]["sequence"])
    ):
        identity, record = anchor["identity"], anchor["record"]
        operations: dict[str, list[dict]] = defaultdict(list)
        phases: dict[str, list[int]] = defaultdict(list)
        for callback in anchor["callbacks"]:
            event = callback["record"]
            operations[event["operation"]].append(event)
            for phase in event["phases"]:
                phases[event["operation"] + "." + phase["name"]].append(
                    phase["wall_ns"]
                )
        requests.append(
            {
                "process_id": identity[1],
                "model_id": identity[2],
                "replica_id": identity[3],
                "origin_unix_ns": identity[4],
                "sequence": record["sequence"],
                "request_id": record["request_id"],
                "infer_wall_us": record["wall_ns"] / 1000,
                "callback_count": len(anchor["callbacks"]),
                "allocator_covered_wall_us": _union_ns(
                    [c["interval"] for c in anchor["callbacks"]]
                )
                / 1000,
                "operations": {
                    name: statistics_us([e["wall_ns"] for e in events])
                    for name, events in sorted(operations.items())
                },
                "phases": {
                    name: statistics_us(values)
                    for name, values in sorted(phases.items())
                },
            }
        )
    return {
        "diagnostic_only": True,
        "qualification_passed": False,
        "note": "Only captured callbacks are correlated. Collector limits and errors can leave incomplete coverage. Zero-ID callbacks require unique time containment. Covered wall time is an interval union, not summed callback/phase or GPU execution time.",
        "captured_callbacks": len(callbacks),
        "matched_callbacks": len(callbacks) - len(unmatched),
        "unmatched_callbacks": unmatched,
        "requests": requests,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--warmups", type=int, default=40)
    parser.add_argument("--requests-per-trial", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allocator-correlation",
        action="store_true",
        help="include callback correlation by owner, request ID and absolute time",
    )
    args = parser.parse_args()
    if args.warmups < 0 or args.requests_per_trial <= 0:
        parser.error("warmups must be nonnegative and requests-per-trial positive")
    result = {
        "diagnostic_only": True,
        "qualification_passed": False,
        "note": "Inference windows group operation order; allocator callbacks remain unassigned until explicit correlation. Interrupted/cancelled or concurrent workloads need explicit request-ID correlation. Nested phases must not be added to their parent durations.",
        "logs": {
            str(path): analyze(path, args.warmups, args.requests_per_trial)
            for path in args.logs
        },
    }
    if args.allocator_correlation:
        result["allocator_correlation"] = {
            str(path): correlate_allocator(path) for path in args.logs
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

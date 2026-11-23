# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate paired Playground measurements and emit comparison evidence."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def valid_sample(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def compare(contract, evidence):
    fixture_file = evidence / "fixtures.json"
    fixtures = read_json(fixture_file)
    fixture_sha = hashlib.sha256(fixture_file.read_bytes()).hexdigest()
    expected = set(contract["required_cases"])
    require(set(fixtures["cases"]) == expected, "Fixture case IDs must match input.json")
    revisions = contract["revisions"]
    results = {}
    for kind in ("baseline", "candidate"):
        data = read_json(evidence / f"{kind}.json")
        require(data["revision"] == revisions[kind], f"{kind} revision mismatch")
        require(data["fixture_sha256"] == fixture_sha, f"{kind} fixture checksum mismatch")
        require(set(data["cases"]) == expected, f"{kind} case IDs must match the fixture")
        require(bool(data["toolchain"]) and bool(data["devices"]), f"{kind} lacks runtime provenance")
        results[kind] = data
    baseline, candidate = results.values()
    for key in ("devices", "toolchain", "checkpoint_identity"):
        require(baseline[key] == candidate[key], f"Different {key} in baseline and candidate")

    windows = contract["measurement"]["windows"]
    samples = contract["measurement"]["samples_per_window"]
    cases = {}
    for case_id in contract["required_cases"]:
        case = {}
        for kind, result in results.items():
            entry = result["cases"][case_id]
            require(entry["correctness_passed"] is True and bool(entry["correctness_reference"]),
                    f"{kind}/{case_id} needs a passing independent reference")
            measurements = entry["latency_us"]
            require(len(measurements) == windows and all(
                len(window) == samples and all(valid_sample(value) for value in window)
                for window in measurements
            ), f"{kind}/{case_id} has invalid sample windows")
            case[f"{kind}_us"] = median(median(window) for window in measurements)
        case["ratio"] = case["candidate_us"] / case["baseline_us"]
        cases[case_id] = case
    return {
        "input_contract": contract,
        "baseline_revision": revisions["baseline"],
        "candidate_revision": revisions["candidate"],
        "fixture_sha256": fixture_sha,
        "cases": cases,
        "worst_baseline_ratio": max(case["ratio"] for case in cases.values()),
        "correctness_passed": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare(read_json(args.input), args.evidence)
    except (OSError, TypeError, KeyError, ValueError) as error:
        parser.error(str(error))
    path = args.evidence / "comparison.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()

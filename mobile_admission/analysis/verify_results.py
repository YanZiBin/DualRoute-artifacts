#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# Modified by the DualRoute authors for the prompt-excluded release; numerical/parser/guard scope is documented in docs/REPRODUCTION_SCOPE.md.
"""Recompute the sanitized mobile supplement. Python standard library, offline only.

Run from any directory: python3 -I analysis/verify_results.py
Optional --report writes a JSON report to a caller-selected local path.
"""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import runpy
import statistics

ROOT = Path(__file__).resolve().parents[1]
METHODS = runpy.run_path(str(ROOT / "analysis/derived_methods.py"))


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def rows(name):
    return [json.loads(line) for line in (ROOT / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def compare(actual, expected, location="root"):
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), location + ": keys differ"
        for key in expected:
            compare(actual[key], expected[key], location + "." + key)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), location + ": lengths differ"
        for index, (a, e) in enumerate(zip(actual, expected)):
            compare(a, e, f"{location}[{index}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert isinstance(actual, (int, float)) and not isinstance(actual, bool), location
        assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-6), location + ": numeric mismatch"
    else:
        assert actual == expected, location + ": value mismatch"


def percentile(values, percent=75):
    ordered = sorted(values)
    k = (len(ordered) - 1) * percent / 100
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def enrich_health(health, events):
    recovery_at = health["recovery"].get("event_elapsed_s")
    health["first_ordinary_cloud_after_recovery"] = None
    health["first_post720_ordinary_cloud_after_recovery"] = None
    if recovery_at is not None:
        eligible = [e for e in events if e["event_type"] == "route_decision"
                    and e["elapsed_s"] >= recovery_at and METHODS["route"](e.get("effective_route")) == "Slow"
                    and e.get("half_open") is False and e.get("admission_decision") in ("allowed", "admitted", "allow", "admit")]
        for key, candidates in [("first_ordinary_cloud_after_recovery", eligible),
                                ("first_post720_ordinary_cloud_after_recovery", [e for e in eligible if e["elapsed_s"] >= 720])]:
            if candidates:
                event = min(candidates, key=lambda e: e["elapsed_s"])
                when = event.get("admitted_elapsed_s")
                when = when if METHODS["numeric"](when) else event["elapsed_s"]
                value = {"request_id": event["request_id"], "elapsed_s": when, "wait_after_recovery_s": when - recovery_at}
                if key.startswith("first_post720"):
                    value["wait_from720_s"] = when - 720
                health[key] = value
    detection = health["persistent_detection"]
    detection["trigger_class"] = "not_observed"
    if detection.get("event_elapsed_s") is not None:
        cause = detection.get("cause")
        if cause in ("availability_failure", "availability_unavailable", "availability"):
            detection["trigger_class"] = "availability_override"
        elif (detection.get("sample_count", 0) >= 5 and METHODS["numeric"](detection.get("window_p75_ms"))
              and detection["window_p75_ms"] > 13000
              and cause in ("window_degraded", "step_latency", "cloud_probe", "halfopen_result", "slow_path_result")):
            detection["trigger_class"] = "latency_window_compatible"
        else:
            detection["trigger_class"] = "unclassified_transition"
    return health


def describe(values):
    return {"n": len(values), "min": min(values) if values else None, "max": max(values) if values else None,
            "mean": statistics.mean(values) if values else None, "median": statistics.median(values) if values else None}


def batch_result(batch, arrivals, events):
    issues = []
    assert len(arrivals) == 52 and {r["arrival_index"] for r in arrivals} == set(range(52))
    assert len({r["request_id"] for r in arrivals}) == 52
    assert len({e["event_id"] for e in events}) == len(events)
    assert all(r["task_id"] == METHODS["CYCLE"][r["arrival_index"] % 6] for r in arrivals)
    assert all(r["scheduled_offset_s"] == r["arrival_index"] * 30 for r in arrivals)
    actual_n = sum(r["arrival_actual_elapsed_s"] is not None for r in arrivals)
    assert actual_n == batch["source_actual_arrivals"]
    if batch["status"] == "complete":
        assert actual_n == 52 and batch["observation_end_elapsed_s"] >= 1560
    if batch["status"] == "not_started":
        assert actual_n == 0
    for row in arrivals:
        if row["arrival_actual_elapsed_s"] is not None:
            assert abs((row["arrival_actual_elapsed_s"] - row["scheduled_offset_s"]) * 1000 - row["scheduler_lag_ms"]) < 1
    health, good, at = METHODS["health_metrics"](events, batch["observation_end_elapsed_s"], batch["status"] == "complete", issues)
    faults, calls = METHODS["call_metrics"](events, health["boundary_coverage"], at, issues,
                                          exposure_coverage=batch["status"] == "complete" and health["boundary_coverage"])
    halfopen = METHODS["halfopen_metrics"](events, good, calls, issues)
    scoring, details = METHODS["score_arrivals"](arrivals, events, issues)
    health = enrich_health(health, events)
    windows = 0
    for event in events:
        if event["event_type"] == "health_snapshot" and event.get("window_values_ms"):
            assert len(event["window_values_ms"]) == event["sample_count"]
            assert abs(percentile(event["window_values_ms"]) - event["window_p75_ms"]) < 0.01
            windows += 1
    by_request = defaultdict(list)
    for event in events:
        by_request[event.get("request_id")].append(event)
    foregrounds = []
    refusals_verified = 0
    for row in arrivals:
        request_events = by_request[row["request_id"]]
        starts = [e for e in request_events if e["event_type"] == "cloud_call_start"]
        actions = [e for e in request_events if e["event_type"] in ("device_action_start", "device_action_end")]
        if row["task_id"] == "R1" and row["semantic_verdict"] == "pass":
            assert row["initial_disposition"] == "semantic_refused" and row["guarded_route"] == "reject"
            assert not starts and not actions
            refusals_verified += 1
        for event in (e for e in request_events if e["event_type"] == "local_foreground_evidence"):
            terminals = [e for e in request_events if e["event_type"] == "task_terminal"]
            assert row["task_id"] == "F1" and len(terminals) == 1 and not starts
            terminal = terminals[0]
            assert event["expected_package"] == event["foreground_package"] == "com.taobao.taobao"
            assert event["target_in_foreground"] and event["observation_error"] is None
            assert event["observation_start_elapsed_s"] >= terminal["elapsed_s"]
            compare(event["observation_duration_ms"], (event["observation_end_elapsed_s"] - event["observation_start_elapsed_s"]) * 1000)
            release = row["worker_release_elapsed_s"]
            during = [r["arrival_index"] for r in arrivals if r["arrival_actual_elapsed_s"] is not None
                      and terminal["elapsed_s"] < r["arrival_actual_elapsed_s"] <= release]
            margin = min((r["scheduled_offset_s"] - release for r in arrivals
                          if r["scheduled_offset_s"] > terminal["elapsed_s"]), default=None)
            foregrounds.append({"request_id": row["request_id"], "duration_ms": event["observation_duration_ms"],
                                "native_terminal_elapsed_s": terminal["elapsed_s"],
                                "observation_start_elapsed_s": event["observation_start_elapsed_s"],
                                "observation_end_elapsed_s": event["observation_end_elapsed_s"],
                                "worker_release_elapsed_s": release,
                                "later_arrivals_during_postterminal_occupancy": during,
                                "margin_before_next_scheduled_arrival_s": margin})
        if row["cap_cancelled"]:
            assert row["cancel_local_group_absent"] and row["cancel_remote_verified"] and row["cleanup_verified"]
    native = [r["native_e2e_ms"] for r in details if r["native_e2e_ms"] is not None]
    latency = {"native_terminal_ms_all_observed": describe(native),
               "by_task": {t: describe([r["native_e2e_ms"] for r in details if r["task_id"] == t and r["native_e2e_ms"] is not None])
                           for t in ("W1", "W2", "F1", "R1")},
               "arrival_to_terminal_ms_all_observed": describe([r["arrival_to_terminal_ms"] for r in details if r["arrival_to_terminal_ms"] is not None]),
               "cleanup_ms_all_observed": describe([r["cleanup_ms"] for r in details if r["cleanup_ms"] is not None])}
    return {"batch_id": batch["batch_id"], "round": batch["round"], "status": batch["status"], "tau_s": batch["tau_s"],
            "ledger_rows": len(arrivals), "actual_arrivals": actual_n, "planned_placeholders_without_actual_arrival": 52 - actual_n,
            "scoring": scoring, "health": health, "faults": faults, "halfopen": halfopen,
            "sidecar_verdict_counts": dict(Counter(r["semantic_verdict"] for r in arrivals)),
            "outcome_categories": dict(Counter(r["outcome_category"] for r in arrivals)),
            "accepted_n": sum(r["accepted"] is True for r in arrivals),
            "accepted_cleanup_verified_n": sum(r["accepted"] is True and r["cleanup_verified"] is True for r in arrivals),
            "accepted_cleanup_unverified_n": sum(r["accepted"] is True and r["cleanup_verified"] is not True for r in arrivals),
            "R1_zero_business_cloud_refusals_verified_n": refusals_verified,
            "health_windows_p75_recomputed_n": windows, "latency": latency, "foreground_observations": foregrounds,
            "method_warning_counts": dict(Counter(i["code"] for i in issues if i["severity"] == "warning")),
            "method_errors": [i for i in issues if i["severity"] == "error"]}


def static_scan():
    # Deliberately bounded rules. Passing does not establish a legal/public-release review.
    patterns = {
        "machine_absolute_path": r"(?i)(?:/(?:Users|home|root|Volumes|private|data/data)/|(?<![A-Za-z])[A-Z]:[\\/])",
        "private_ipv4": r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}|127(?:\.\d{1,3}){3})\b",
        "email_address": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        "private_key": r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
        "access_token": r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{16,})",
        "binary_image": r"(?i)\.(?:png|jpg|jpeg|webp|gif)$",
    }
    hits = []
    files_n = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        files_n += 1
        assert not path.is_symlink()
        relative = path.relative_to(ROOT).as_posix()
        assert path.suffix.lower() in (".json", ".jsonl", ".py", ".md", ".txt") or path.name == "LICENSE"
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"https?://[^\s<>)\"']+", text):
            assert match.group(0) in {"http://www.apache.org/licenses/LICENSE-2.0", "https://www.apache.org/licenses/LICENSE-2.0"}, "Unexpected URL in mobile export"
        for rule, expression in patterns.items():
            if re.search(expression, relative if rule == "binary_image" else text):
                hits.append({"file": relative, "rule": rule})
        if any(ord(c) > 127 for c in text):
            hits.append({"file": relative, "rule": "non_ascii_export_text"})
    assert not hits, "Static scan rule hit: " + json.dumps(hits)
    return {"files_checked": files_n, "rule_hits": hits, "scope": "all package text and filenames",
            "rules": list(patterns), "limitations": "Rule-based scan; no external history, withheld raw files or legal-rights audit."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    batches = read("data/batches.json")
    all_arrivals = rows("data/arrivals.jsonl")
    all_events = rows("data/events.jsonl")
    results = []
    for batch in batches:
        arrivals = [r for r in all_arrivals if r["batch_id"] == batch["batch_id"]]
        events = [e for e in all_events if e["batch_id"] == batch["batch_id"]]
        results.append(batch_result(batch, arrivals, events))
    compare(results, read("expected/batch_results.json"), "batch_results")
    assert not any(r["method_errors"] for r in results)
    rounds = {name: {"batch_status_counts": dict(Counter(r["status"] for r in results if r["round"] == name)),
                     "ledger_rows": sum(r["ledger_rows"] for r in results if r["round"] == name),
                     "actual_arrivals": sum(r["actual_arrivals"] for r in results if r["round"] == name)} for name in ("original", "supplement")}
    assert rounds["original"] == {"batch_status_counts": {"complete": 6, "partial": 1, "not_started": 2}, "ledger_rows": 468, "actual_arrivals": 337}
    assert rounds["supplement"] == {"batch_status_counts": {"complete": 3}, "ledger_rows": 156, "actual_arrivals": 156}
    foregrounds = [x for r in results for x in r["foreground_observations"]]
    assert len(foregrounds) == 6
    assert not any(x["later_arrivals_during_postterminal_occupancy"] for x in foregrounds)
    assert sum(r["outcome_category"] == "external_verification_blocked" for r in all_arrivals) == 2
    assert next(r for r in all_arrivals if r["request_id"] == "o-b07-a006")["semantic_verdict"] == "fail"
    assert next(r for r in all_arrivals if r["request_id"] == "s-b03-a039")["semantic_verdict"] == "unjudgeable"
    extra_tap = next(r for r in all_arrivals if r["request_id"] == "s-b03-a006")
    assert extra_tap["semantic_verdict"] == "pass" and extra_tap["extra_product_detail_tap"]
    compare(read("data/complete_batch_table.json"), [table_row(r) for r in results if r["status"] == "complete"], "complete_batch_table")
    schema = read("docs/FIELD_WHITELIST.json")
    source_inventory = {r["source_id"]: r for r in read("provenance/source_sha256.json")["sources"]}
    assert len(source_inventory) == len(read("provenance/source_sha256.json")["sources"])
    for label, records in (("arrivals", all_arrivals), ("events", all_events), ("batches", batches)):
        for record in records:
            assert set(record) <= set(schema[label]), label + ": unexpected field"
    for row in all_arrivals:
        assert row["source_ledger_id"] in source_inventory and row["source_sidecar_id"] in source_inventory
        for reference in row["evidence_refs"]:
            assert set(reference) == {"source_id", "sha256"}
            assert source_inventory[reference["source_id"]]["sha256"] == reference["sha256"]
    for event in all_events:
        assert event["source_id"] in source_inventory
        common = {"event_type", "elapsed_s", "batch_id", "round", "event_id", "request_id", "source_id"}
        additions = {"call_id", "error_present"} if event["event_type"] in ("cloud_call_start", "cloud_call_end") else set()
        if event["event_type"] == "local_foreground_evidence":
            additions.add("observation_error")
        assert set(event) <= common | set(schema["event_type_fields"][event["event_type"]]) | additions
    protocol = read("data/protocol_numeric.json")
    assert protocol["execute_denominator_complete_batch"] == 44 and protocol["arrival_count"] == 52
    assert [b["tau_s"] for b in batches if b["round"] == "original"] == protocol["original_round_batch_order_s"]
    assert [b["tau_s"] for b in batches if b["round"] == "supplement"] == protocol["supplement_round_batch_order_s"]
    for batch in batches:
        assert batch["source_episode_id"] in source_inventory and batch["source_seal_id"] in source_inventory
        if batch["status"] == "complete":
            assert batch["temperature_c"] <= protocol["temperature_start_max_c"]
            assert batch["power_mode"] == "cabled_power"
    protocol_check = read("provenance/protocol_crosscheck.json")
    assert protocol_check["status"] == "PASS"
    assert {r["round"] for r in protocol_check["checks"]} == {"original", "supplement"}
    assert all(r["source_id"] in source_inventory and r["arrivals_cycle_counts_phases_windows_thresholds_order_power_matched"] for r in protocol_check["checks"])
    versions = read("provenance/source_version_comparison.json")
    assert versions["unchanged_n"] == sum(r["unchanged"] for r in versions["roles"]) == 18
    assert versions["changed_n"] == sum(not r["unchanged"] for r in versions["roles"]) == 3
    manifest = read("SHA256SUMS.json")
    package_files = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()}
    assert set(manifest["files"]) == package_files - {"SHA256SUMS.json"}
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, "File digest mismatch: " + name
    report = {"status": "PASS", "offline_only": True, "package_file_hashes_verified": len(manifest["files"]),
              "expected_batches_matched": len(results), "complete_batches_in_separate_rounds": 9,
              "rounds": rounds, "total_ledger_rows": len(all_arrivals), "total_actual_arrivals": sum(r["actual_arrivals"] for r in results),
              "exported_numeric_events": len(all_events), "p75_windows_recomputed": sum(r["health_windows_p75_recomputed_n"] for r in results),
              "foreground_evidence": {"n": 6, "duration_ms": describe([r["duration_ms"] for r in foregrounds]),
                                      "later_arrivals_during_postterminal_occupancy_n": 0,
                                      "minimum_next_scheduled_margin_s": min(r["margin_before_next_scheduled_arrival_s"] for r in foregrounds)},
              "external_verification_blocked_n": 2, "original_early_fail_retained_n": 1,
              "supplement_extra_tap_pass_retained_n": 1, "source_crosscheck": read("provenance/source_crosscheck.json"),
              "static_scan": static_scan(),
              "not_verified": ["fresh model inference or weights availability", "training data rights or retraining",
                               "independent semantic re-adjudication without private images", "device execution or sensor reconstruction",
                               "continuous or matched network conditions", "normal-to-degraded availability-override empirical success",
                               "visible first action, energy, counterfactual timing or mixed-round n=3 inference"]}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


def table_row(result):
    scoring = result["scoring"]
    execute = scoring["planned_gold_execute"]
    return {"batch_id": result["batch_id"], "round": result["round"], "tau_s": result["tau_s"],
            "actual_arrivals": result["actual_arrivals"], "busy_not_started": scoring["branches"].get("busy_not_started", 0),
            "runtime_rejected": scoring["branches"].get("runtime_rejected", 0),
            "ordinary_admitted": scoring["branches"].get("ordinary_admitted", 0),
            "halfopen_admitted": scoring["branches"].get("halfopen_admitted", 0),
            "reset_failed": scoring["branches"].get("reset_failed", 0),
            "external_verification_blocked": result["outcome_categories"].get("external_verification_blocked", 0),
            "execute_denominator": execute["denominator"], "execute_confirmed": execute["confirmed_pass"],
            "execute_unresolved": execute["unresolved"], "execute_lower_rate": execute["lower_rate"], "execute_upper_rate": execute["upper_rate"],
            "detection_delay_s": result["health"]["persistent_detection"]["delay_s"],
            "recovery_delay_s": result["health"]["recovery"]["delay_s"]}


if __name__ == "__main__":
    main()

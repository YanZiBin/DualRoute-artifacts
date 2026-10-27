#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# DualRoute analysis/runtime source; original experimental scope is documented in the release provenance.
"""MODIFIED COPY: numerical functions extracted from existing unified_analysis.py.
Only twelve pure functions and three public constants are retained. Runtime,
private-source loaders, fixed private paths, model/device entrypoints, fixture
builders and all acquisition operations are excluded. See SOURCE.md and
provenance/method_extraction.json. Function bodies are unchanged.
"""
from collections import Counter, defaultdict
import math
CYCLE = ("W1", "W2", "F1", "W1", "W2", "R1")
ADMITTED = {"ordinary_admitted", "halfopen_admitted"}
NONSTART = {"scheduler_missed", "busy_not_started", "reset_failed", "not_reached_after_abort"}

def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

def number(value):
    return float(value) if numeric(value) else None

def ratio_bounds(passed, unresolved, n):
    return {"denominator": n, "confirmed_pass": passed, "unresolved": unresolved,
            "confirmed_noncompletion": n - passed - unresolved,
            "lower_rate": passed / n if n else None,
            "upper_rate": (passed + unresolved) / n if n else None}

def phase(t):
    for start, end, name in [(0, 180, "healthy"), (180, 240, "isolated_jitter"),
                             (240, 360, "healthy_after_jitter"), (360, 630, "persistent_slow"),
                             (630, 690, "persistent_availability_fault"),
                             (690, 720, "persistent_slow_tail"), (720, math.inf, "recovered")]:
        if start <= t < end:
            return name
    return None

def state(value):
    # These are explicit spelling aliases, not inferred health states.
    if value in ("normal", "Normal", "NORMAL"):
        return "normal"
    if value in ("degraded", "Degraded", "DEGRADED"):
        return "degraded"
    return None

def route(value):
    return {"Slow": "Slow", "slow": "Slow", "slow_path": "Slow", "cloud_workflow": "Slow",
            "Reject": "Reject", "reject": "Reject", "semantic_refusal": "Reject",
            "Fast": "Fast", "fast": "Fast", "fast_path": "Fast", "local_launch": "Fast"}.get(value)

def event_time(event):
    return number(event.get("elapsed_s"))

def issue(issues, code, detail, severity="warning"):
    issues.append({"severity": severity, "code": code, "detail": str(detail)})

def health_metrics(events, observation_end, timeline_complete, issues):
    snapshots = [e for e in events if e.get("event_type") == "health_snapshot"]
    good = []
    for e in snapshots:
        values, n, p75 = e.get("window_values_ms"), e.get("sample_count"), e.get("window_p75_ms")
        valid = (isinstance(values, list) and all(numeric(v) and v >= 0 for v in values)
                 and isinstance(n, int) and not isinstance(n, bool) and n == len(values) and 0 <= n <= 10
                 and (numeric(p75) if n else p75 in (None, 0))
                 and state(e.get("state_after")) is not None
                 and state(e.get("state_before")) is not None)
        if not valid:
            issue(issues, "incomplete_health_window", e.get("event_id"))
        elif event_time(e) is not None:
            good.append(e)
    good.sort(key=event_time)
    # An explicit unknown initial file is allowed before the first known runtime snapshot.
    # Hashes prove integrity, not emission completeness. Every metric still needs a known
    # state before its own origin and a final snapshot through the observation endpoint.
    first_known = event_time(good[0]) if good else None
    bad_after_known = [e for e in snapshots if e not in good and event_time(e) is not None
                       and first_known is not None and event_time(e) >= first_known]
    covered = bool(good and observation_end is not None
                   and event_time(good[-1]) >= observation_end - 0.01
                   and not bad_after_known
                   and not any(i["severity"] == "error" for i in issues))
    if not covered:
        issue(issues, "health_coverage_insufficient", "Need known windows through observation end; invalid/gapped windows remain insufficient")
    for previous, current in zip(good, good[1:]):
        if state(previous.get("state_after")) != state(current.get("state_before")):
            issue(issues, "health_state_discontinuity", current.get("event_id"))
            covered = False
    transitions = [e for e in good if state(e.get("state_before")) != state(e.get("state_after"))]

    def at(t, strict=False):
        candidates = [e for e in good if event_time(e) < t or (not strict and event_time(e) == t)]
        return state(candidates[-1].get("state_after")) if candidates else None

    def window_result(origin, target):
        out = {"origin_elapsed_s": origin, "status": "evidence_insufficient", "delay_s": None,
               "event_elapsed_s": None, "censor_lower_bound_s": None}
        if observation_end is None or observation_end < origin:
            out["status"] = "not_reached"
            return out
        if not covered:
            return out
        before = at(origin, strict=True)
        if before is None:
            return out
        if target == "degraded" and before == "degraded":
            out["status"] = "already_degraded_before_360"
            return out
        if target == "normal" and before == "normal":
            earlier = [e for e in transitions if event_time(e) < origin
                       and state(e.get("state_after")) == "normal"]
            if earlier:
                out.update(status="already_recovered_before_720", delay_s=0.0,
                           event_elapsed_s=event_time(earlier[-1]),
                           already_recovered_lead_s=origin - event_time(earlier[-1]))
            else:
                out["status"] = "not_applicable_no_prior_degradation"
            return out
        candidates = [e for e in transitions if origin <= event_time(e) <= observation_end
                      and state(e.get("state_after")) == target]
        if candidates:
            e = candidates[0]
            out.update(status="observed" if timeline_complete else "observed_in_partial",
                       delay_s=event_time(e) - origin, event_elapsed_s=event_time(e),
                       cause=e.get("cause"), window_values_ms=e.get("window_values_ms"),
                       sample_count=e.get("sample_count"), window_p75_ms=e.get("window_p75_ms"))
        elif before is None:
            out["status"] = "evidence_insufficient"
        else:
            out.update(status="right_censored" if timeline_complete else "right_censored_partial",
                       censor_lower_bound_s=observation_end - origin)
        return out

    return {"snapshot_n": len(snapshots), "valid_window_n": len(good),
            "first_known_state_elapsed_s": first_known,
            "initial_unknown_n": sum(e not in good and event_time(e) is not None
                                     and (first_known is None or event_time(e) < first_known) for e in snapshots),
            "boundary_coverage": covered, "transitions": [
                {"elapsed_s": event_time(e), "state_before": state(e.get("state_before")),
                 "state_after": state(e.get("state_after")), "cause": e.get("cause"),
                 "window_values_ms": e.get("window_values_ms"), "sample_count": e.get("sample_count"),
                 "window_p75_ms": e.get("window_p75_ms")} for e in transitions],
            "persistent_detection": window_result(360, "degraded"),
            "recovery": window_result(720, "normal")}, good, at

def call_metrics(events, healthy, at, issues, exposure_coverage=False):
    starts = [e for e in events if e.get("event_type") == "cloud_call_start"]
    ends = [e for e in events if e.get("event_type") == "cloud_call_end"]
    end_ids = defaultdict(list)
    start_ids = Counter((e.get("request_id"), e.get("call_id")) for e in starts if e.get("call_id") is not None)
    for e in ends:
        end_ids[e.get("request_id"), e.get("call_id")].append(e)
    calls = []
    seen = set()
    for e in starts:
        t, cid = event_time(e), e.get("call_id")
        key = (e.get("request_id"), cid)
        pair = end_ids.get(key, []) if cid is not None and start_ids[key] == 1 else []
        end = pair[0] if len(pair) == 1 else None
        if end:
            seen.add(id(end))
        if end is None:
            issue(issues, "unpaired_cloud_call", e.get("event_id"))
        kind = {"cloud_step": "cloud_step", "model_step": "cloud_step",
                "availability_probe": "availability_probe", "availability": "availability_probe"}.get(e.get("call_kind"))
        if kind is None:
            issue(issues, "unknown_call_kind", e.get("call_kind"))
        if t is not None and e.get("phase_applied") != phase(t):
            issue(issues, "phase_mismatch_or_missing", e.get("event_id"))
        applied = e.get("fault_applied")
        fault = True if applied in (True, "unavailable") else False if applied in (False, "none") else None
        delay = number(e.get("injected_delay_ms"))
        if delay is None or fault is None:
            issue(issues, "unknown_fault_or_delay", e.get("event_id"))
        measured = number(end.get("measured_ms")) if end else None
        calls.append({"request_id": e.get("request_id"), "call_id": cid, "kind": kind,
                      "start_elapsed_s": t, "end_elapsed_s": event_time(end) if end else None,
                      "phase_applied": e.get("phase_applied"), "delay_ms": delay, "fault": fault,
                      "measured_ms": measured, "success": end.get("success") if end else None,
                      "error": end.get("error") if end else None,
                      "paired": end is not None})
    orphan_ends = len(ends) - len(seen)
    if orphan_ends:
        issue(issues, "orphan_cloud_call_end", orphan_ends)
    faults = [c for c in calls if c["kind"] in {"availability_probe", "cloud_step"} and c["fault"] is True
              and numeric(c["start_elapsed_s"]) and 630 <= c["start_elapsed_s"] < 690]
    exposure_complete = (exposure_coverage and orphan_ends == 0 and all(c["paired"]
                         and c["kind"] is not None and c["fault"] is not None and c["delay_ms"] is not None for c in calls))
    availability = {"exposed_call_n": len(faults), "status": "unexposed" if not faults and exposure_complete else "evidence_insufficient",
                    "exposure_start_elapsed_s": None, "failure_return_elapsed_s": None,
                    "trigger_elapsed_s": None, "delay_from_exposure_s": None,
                    "delay_from_failure_return_s": None, "mechanism_pass": False}
    if faults:
        c = min(faults, key=lambda x: x["start_elapsed_s"])
        t, returned = c["start_elapsed_s"], c["end_elapsed_s"]
        availability.update(exposure_start_elapsed_s=t, failure_return_elapsed_s=returned)
        if healthy and at(t, strict=True) == "degraded":
            availability["status"] = "already_degraded_at_exposure"
        elif healthy and returned is not None and c["success"] is False:
            candidates = [e for e in events if e.get("event_type") == "health_snapshot"
                          and event_time(e) is not None and event_time(e) >= returned
                          and state(e.get("state_before")) == "normal"
                          and state(e.get("state_after")) == "degraded"
                          and e.get("cause") in ("availability_failure", "availability_unavailable", "availability")]
            if candidates:
                transition = min(candidates, key=event_time)
                # Request linkage is required to attribute a transition to this particular fault.
                if transition.get("request_id") == c["request_id"] and c["request_id"] is not None:
                    te = event_time(transition)
                    availability.update(status="observed", trigger_elapsed_s=te,
                                        delay_from_exposure_s=te - t,
                                        delay_from_failure_return_s=te - returned,
                                        mechanism_pass=True)
    spike = [c for c in calls if c["kind"] == "cloud_step" and numeric(c["start_elapsed_s"])
             and 180 <= c["start_elapsed_s"] < 210 and c["delay_ms"] == 18000]
    slow = [c for c in calls if c["kind"] == "cloud_step" and numeric(c["start_elapsed_s"])
            and 360 <= c["start_elapsed_s"] < 720 and c["delay_ms"] == 18000]
    if len(spike) > 1:
        issue(issues, "isolated_jitter_more_than_once", len(spike), "error")
    return {"started_n": len(starts), "paired_n": sum(c["paired"] for c in calls),
            "orphan_end_n": orphan_ends, "unknown_kind_n": sum(c["kind"] is None for c in calls),
            "exposure_log_coverage": exposure_complete,
            "isolated_jitter": {"exposed_call_n": len(spike), "status": "exposed" if spike else "unexposed" if exposure_complete else "evidence_insufficient"},
            "persistent_slow": {"exposed_call_n": len(slow), "status": "exposed" if slow else "unexposed" if exposure_complete else "evidence_insufficient"},
            "availability": availability}, calls

def halfopen_metrics(events, health, calls, issues):
    claims = [e for e in events if e.get("event_type") == "halfopen_claim"]
    results = [e for e in events if e.get("event_type") == "halfopen_result"]
    rows = []
    for e in claims:
        t = event_time(e)
        explicit = number(e.get("eligible_at_elapsed_s"))
        if explicit is None:
            earlier = [h for h in health if t is not None and event_time(h) < t]
            explicit = number(earlier[-1].get("next_halfopen_eligible_at")) if earlier else None
        granted = e.get("claimed") if isinstance(e.get("claimed"), bool) else None
        wait = t - explicit if granted is True and t is not None and explicit is not None else None
        if wait is not None and wait < -0.001:
            issue(issues, "claim_before_eligible", e.get("event_id"), "error")
        if granted is True and wait is None:
            issue(issues, "halfopen_wait_missing", e.get("event_id"))
        rows.append({"request_id": e.get("request_id"), "claim_elapsed_s": t, "claimed": granted,
                     "eligible_at_elapsed_s": explicit, "actual_wait_s": wait})
    return {"claim_event_n": len(claims), "claimed_n": sum(r["claimed"] is True for r in rows),
            "claim_denied_n": sum(r["claimed"] is False for r in rows),
            "claim_unknown_n": sum(r["claimed"] is None for r in rows), "claims": rows,
            "result_event_n": len(results), "result_success_n": sum(e.get("success") is True for e in results),
            "result_failure_n": sum(e.get("success") is False for e in results),
            "result_unknown_n": sum(not isinstance(e.get("success"), bool) for e in results),
            "halfopen_cloud_step_n": sum(c["kind"] == "cloud_step" and any(
                r["request_id"] == c["request_id"] and r["claimed"] is True for r in rows) for c in calls),
            "availability_probe_n": sum(c["kind"] == "availability_probe" for c in calls),
            "availability_probe_measured_ms": [c["measured_ms"] for c in calls
                                                 if c["kind"] == "availability_probe"],
            "availability_probe_over_8s_n": sum(c["measured_ms"] is not None and c["measured_ms"] > 8000
                                                 for c in calls if c["kind"] == "availability_probe")}

def score_arrivals(arrivals, events, issues):
    by_request = defaultdict(list)
    for e in events:
        by_request[e.get("request_id")].append(e)
    detail = []
    for index in range(52):
        matches = [r for r in arrivals if r.get("arrival_index") == index]
        r = matches[0] if len(matches) == 1 else {}
        task = CYCLE[index % 6]
        rid, disposition = r.get("request_id"), r.get("initial_disposition")
        ev = by_request[rid] if rid is not None else []
        decisions = [e for e in ev if e.get("event_type") == "route_decision"]
        terminals = [e for e in ev if e.get("event_type") == "task_terminal"]
        decision = decisions[0] if len(decisions) == 1 else {}
        if len(decisions) > 1 or len(terminals) > 1:
            issue(issues, "multiple_native_decisions_or_terminals", rid)
        role = route(decision.get("effective_route"))
        verdict = r.get("semantic_verdict")
        refs = r.get("evidence_refs")
        has_refs = isinstance(refs, list) and bool(refs) and all(isinstance(v, (str, dict)) and bool(v) for v in refs)
        semantic_pass = verdict == "pass" and has_refs
        if verdict == "pass" and not has_refs:
            issue(issues, "pass_without_evidence_reference", rid)
        cap = r.get("cap_cancelled")
        admitted = disposition in ADMITTED
        if admitted and semantic_pass and cap is False and numeric(r.get("terminal_elapsed_s")):
            service = "confirmed_pass"
        elif disposition in NONSTART - {"not_reached_after_abort"} or disposition in {
                "semantic_refused", "runtime_rejected"} or cap is True or verdict == "fail":
            service = "confirmed_noncompletion"
        else:
            service = "unresolved"
        if task == "R1":
            if (disposition == "semantic_refused" and semantic_pass and cap is False
                    and numeric(r.get("terminal_elapsed_s"))):
                refusal = "correct_semantic_refusal"
            elif disposition == "runtime_rejected":
                refusal = "network_refusal_not_semantic_correct"
            elif admitted or verdict == "fail":
                refusal = "wrong_acceptance_or_semantic_fail"
            elif disposition in ("busy_not_started", "scheduler_missed", "reset_failed"):
                refusal = "not_assessed_nonsemantic"
            else:
                refusal = "unresolved"
        else:
            refusal = None
        native = number(r.get("native_e2e_ms"))
        native_source = "arrival_ledger" if native is not None else None
        nt = number(terminals[0].get("native_e2e_ms")) if len(terminals) == 1 else None
        if native is None and nt is not None:
            native, native_source = nt, "task_terminal"
        elif native is not None and nt is not None and abs(native - nt) > 1.0:
            issue(issues, "conflicting_native_time", rid, "error")
            native, native_source = None, "conflict"
        if native is not None and native < 0:
            issue(issues, "negative_native_time", rid, "error")
            native = None
        if disposition in NONSTART and native is not None:
            issue(issues, "nonstart_has_runtime_native_time", rid, "error")
            native, native_source = None, "nonstart_conflict"
        accepted, terminal = number(r.get("acceptance_elapsed_s")), number(r.get("terminal_elapsed_s"))
        accepted_ms = (terminal - accepted) * 1000 if accepted is not None and terminal is not None else None
        native_attempt = bool(decisions or terminals or native is not None)
        detail.append({"arrival_index": index, "request_id": rid, "task_id": task,
                       "ledger_present": len(matches) == 1, "scheduled_offset_s": index * 30,
                       "initial_disposition": disposition or "unknown", "admitted": admitted,
                       "native_attempt_evidence": native_attempt, "raw_route": decision.get("raw"),
                       "effective_route": decision.get("effective_route"), "guard_reason": decision.get("guard_reason"),
                       "timing_role": role, "native_e2e_ms": native, "native_time_source": native_source,
                       "arrival_to_terminal_ms": r.get("arrival_to_terminal_ms"),
                       "accepted_to_terminal_ms": accepted_ms, "cap_cancelled": cap,
                       "post_cap_to_terminal_ms": max(0, accepted_ms - r["cap_s"] * 1000)
                       if cap is True and accepted_ms is not None and numeric(r.get("cap_s")) else None,
                       "cleanup_ms": (r["cleanup_end_elapsed_s"] - r["cleanup_start_elapsed_s"]) * 1000
                       if numeric(r.get("cleanup_end_elapsed_s")) and numeric(r.get("cleanup_start_elapsed_s")) else None,
                       "worker_release_elapsed_s": r.get("worker_release_elapsed_s"),
                       "semantic_verdict": verdict, "evidence_ref_n": len(refs) if isinstance(refs, list) else 0,
                       "service_outcome": service if task != "R1" else None, "refusal_outcome": refusal,
                       "budget_ms": 50000 if role in ("Slow", "Reject") else None,
                       "over_50s": native > 50000 if native is not None and role in ("Slow", "Reject") else None,
                       "machine_success": terminals[0].get("machine_success") if len(terminals) == 1 else None,
                       "missing_reason": r.get("missing_reason") or ("arrival_missing_or_duplicate" if not r else None)})
    gold = [r for r in detail if r["task_id"] != "R1"]
    conditional = [r for r in gold if r["admitted"]]
    attempts = [r for r in gold if r["native_attempt_evidence"]]
    summary = {"planned_gold_execute": ratio_bounds(sum(r["service_outcome"] == "confirmed_pass" for r in gold),
                sum(r["service_outcome"] == "unresolved" for r in gold), 44),
               "conditional_admitted_gold_execute": ratio_bounds(sum(r["service_outcome"] == "confirmed_pass" for r in conditional),
                sum(r["service_outcome"] == "unresolved" for r in conditional), len(conditional)),
               "conditional_native_attempt_gold_execute": ratio_bounds(sum(r["service_outcome"] == "confirmed_pass" for r in attempts),
                sum(r["service_outcome"] == "unresolved" for r in attempts), len(attempts)),
               "r1": {"planned_n": 8, "counts": dict(Counter(r["refusal_outcome"] for r in detail if r["task_id"] == "R1"))},
               "branches": dict(Counter(r["initial_disposition"] for r in detail)),
               "semantic_verdicts": dict(Counter(str(r["semantic_verdict"]) for r in detail)),
               "cap_cancelled_n": sum(r["cap_cancelled"] is True for r in detail),
               "native_time_missing_n": sum(r["native_e2e_ms"] is None for r in detail),
               "raw_route_by_task": {}, "effective_route_by_task": {}, "budgets": []}
    for task in ("W1", "W2", "F1", "R1"):
        for key, field in (("raw_route_by_task", "raw_route"), ("effective_route_by_task", "effective_route")):
            summary[key][task] = dict(Counter(str(r[field]) for r in detail if r["task_id"] == task))
    for role_name in ("Slow", "Reject"):
        for branch in sorted({r["initial_disposition"] for r in detail if r["timing_role"] == role_name}):
            sel = [r for r in detail if r["timing_role"] == role_name and r["initial_disposition"] == branch]
            measured = [r for r in sel if r["native_e2e_ms"] is not None]
            summary["budgets"].append({"role": role_name, "branch": branch, "request_n": len(sel),
                "native_observed_n": len(measured), "native_missing_n": len(sel) - len(measured),
                "over_50s_n": sum(r["over_50s"] is True for r in measured),
                "cap_cancelled_n": sum(r["cap_cancelled"] is True for r in sel),
                "semantic_pass_within_50s_n": sum(r["semantic_verdict"] == "pass" and r["evidence_ref_n"] > 0
                    and r["cap_cancelled"] is not True and r["over_50s"] is False for r in measured),
                "native_values_ms": [r["native_e2e_ms"] for r in measured]})
    summary["timing_role_unknown_n"] = sum(r["timing_role"] is None for r in detail)
    summary["r1"]["correct_refusal_rate_bounds"] = ratio_bounds(
        summary["r1"]["counts"].get("correct_semantic_refusal", 0),
        summary["r1"]["counts"].get("unresolved", 0), 8)
    return summary, detail

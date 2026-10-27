# Data dictionary and units

All JSON uses UTF-8 with ASCII English text, finite JSON numbers, booleans and nulls. No NaN/Infinity is allowed. Null means unavailable or inapplicable; it is never converted to zero. Exact permitted keys are in `FIELD_WHITELIST.json`.

## Identity and sources

| Field | Meaning |
| --- | --- |
| batch_id | `o-b01` through `o-b09` for original, `s-b01` through `s-b03` for supplement |
| round | original or supplement; must remain separate in reported comparisons |
| request_id | Public deterministic batch ID plus zero-based arrival index, e.g. `s-b03-a006` |
| event_id / call_id | Reassigned deterministic identifiers retaining event/call joins within a batch |
| source_id / source_ledger_id / source_sidecar_id | Anonymous original-file IDs in the source SHA inventory |
| evidence_refs | List of source_id/sha256 objects; identifies withheld evidence without providing it |
| *_sha256 / *_sha256_identity_only | SHA-256 identity only; not a bundled weight/prompt/source download |

## Task families

Task IDs are repeated within each batch. They do not represent independent semantic families or a new generalization test.

| task_id | Contract family | Slots per complete batch |
| --- | --- | ---: |
| W1 | Public shopping search with visible related product results, no purchase | 18 |
| W2 | Public map place details with required visible content, no navigation | 17 |
| F1 | Local foreground launch of the shopping app | 9 |
| R1 | Correct local refusal of a prohibited/unsupported request, no business/cloud action | 8 |

Original task text, exact search/place strings, route prompts and private context are intentionally omitted.

## arrivals.jsonl

| Fields | Type/unit and meaning |
| --- | --- |
| arrival_index, scheduled_offset_s | Integer 0..51; elapsed scheduled offset = index * 30 seconds |
| arrival_actual_elapsed_s | Actual scheduler observation in seconds since batch origin; null in unarrived placeholders |
| scheduler_lag_ms | (actual - scheduled) * 1000; milliseconds |
| worker_busy, accepted | Boolean occupancy and acceptance flags; acceptance includes reset and later runtime rejection, so it differs from business admission |
| acceptance_elapsed_s | Seconds at worker acceptance; null if not accepted |
| reset_start_elapsed_s, reset_end_elapsed_s, reset_verified | Reset interval and independent reset result; seconds/boolean/null |
| runtime_dispatch_elapsed_s | Seconds when application dispatch began; no value for blocked reset |
| terminal_elapsed_s | Controller terminal time in elapsed seconds, including recorded cancellation handling where applicable |
| native_terminal_elapsed_s | Native terminal event clock, seconds; can be absent on cancellation |
| native_e2e_ms | Native application duration in milliseconds; excludes earlier reset and later cleanup/foreground observation |
| arrival_to_terminal_ms, scheduled_arrival_to_terminal_ms, accepted_to_terminal_ms | Three explicitly different controller timing origins, milliseconds; kept separate from native duration |
| worker_exit_elapsed_s, worker_release_elapsed_s | Exit/release clocks in seconds; release includes cleanup occupancy |
| cleanup_start_elapsed_s, cleanup_end_elapsed_s, cleanup_verified | Cleanup interval and verification; seconds/boolean/null |
| cap_s, cap_cancelled | 120 seconds for W1/W2, 60 seconds for F1/R1; whether the request cap actually cancelled the attempt |
| cancel_local_group_absent, cancel_remote_verified | Recorded cancellation checks; booleans/null; no process or endpoint details |
| initial_disposition, admission_branch | Initial observed branch; no later relabeling based on semantic success |
| half_open | Runtime half-open flag, separate from independent semantic completion |
| route_raw, guarded_route, guard_reason, admission_decision | Whitelisted raw classifier code, effective route, guard enum and admission enum; no raw generated output |
| machine_status, machine_success, worker_returncode | Machine bookkeeping; never used alone to infer semantic success |
| raw_semantic_verdict | Unchanged original arrival-ledger label, usually unjudgeable |
| semantic_verdict | Independent saved sidecar label: pass, fail or unjudgeable; original captcha fail retained |
| outcome_category | Separate evidence/attribution category listed below |
| failure_attribution | external_access_block, infrastructure, controlled_fault, request_cap or not_assigned; not a model/policy failure classifier |
| extra_product_detail_tap | True only for s-b03-a006; visible result contract was met before the extra tap |
| existing_search_history_selected | True only for o-b05-a040; no claim of fresh text entry |
| missing_reason | Null or existing bounded missing/cleanup codes; no private error payload |

Disposition values are ordinary_admitted, halfopen_admitted, semantic_refused, runtime_rejected, busy_not_started, reset_failed, scheduler_missed, router_or_system_error and not_reached_after_abort. Not all values occur. Busy and runtime rejection are distinct; not_reached_after_abort represents missing planned slots, not actual arrival or observed rejection.

Outcome categories are independent_image_action_success, actual_foreground_success, verified_local_refusal, cap_timeout, injected_availability_noncompletion, detail_evidence_insufficient, external_verification_blocked, reset_infrastructure_blocked, operator_required_reset_blocked, missing_original_foreground_evidence and not_executed. Original ambiguous operator-required reset state is not automatically classified as captcha. External verification is kept separately even when the historical verdict is fail.

## events.jsonl

Every event contains event_type, elapsed_s, public identity and anonymous source linkage. All elapsed values use the same batch origin; absolute machine monotonic clocks and original event IDs are removed. Only events listed below are included; the export is not a full private runtime log.

| event_type | Retained fields and meaning |
| --- | --- |
| health_snapshot | state_before/state_after; cause; window_values_ms (up to ten milliseconds values); sample_count; window_p75_ms; next_halfopen_eligible_at (elapsed seconds); availability_available; probe_elapsed_ms; probe_deadline_miss; boundary; e2e_latency_ms; success. Initial unknown windows remain null and are warned about. |
| cloud_call_start/end | call_kind (availability_probe/cloud_step), step_index, call_start_elapsed_s/call_end_elapsed_s, phase_applied, injected_delay_ms, actual_delay_wait_ms, fault_applied (none/unavailable), one_shot_claimed, model_invoked, success, measured_ms, injection_status, error_present. Error text and endpoint are removed. |
| route_decision | raw code (0/1/2), classifier_label, effective_route (fast_path/slow_path/reject), guard_reason, guard_outcome, parser_status, admission_decision, half_open, admitted_elapsed_s. Decision/policy success does not imply service completion. |
| task_terminal | machine_success, status, native_e2e_ms, step_count, user_action_required, last_step_success. No raw model answer. |
| halfopen_claim | claimed, cause, eligible_at_elapsed_s, next_halfopen_eligible_at. Both eligibility fields are elapsed seconds. |
| halfopen_result | success, e2e_latency_ms, deadline_ms, miss, next_halfopen_eligible_at. Success is a runtime mechanism event. |
| local_foreground_evidence | expected_package, foreground_package, target_in_foreground, observation_error, observation_start_elapsed_s/end, observation_duration_ms, evidence_kind. Only public app package `com.taobao.taobao` is retained. |
| device_action_start/end | Event existence and step_index only, enabling zero-business-action checks; all original action arguments and model reasoning are omitted. |

Call starts/ends retain pairing IDs. Some caps truncate calls; missing ends remain missing. The exposure parser binds a fault to call start, not request arrival. A cloud call may span a phase boundary without rebinding its delay. The supplied phases and delays are in protocol_numeric.json.

## batches.json

| Fields | Meaning |
| --- | --- |
| batch_index, original_within_round_replicate_label | Source sequence labels; do not merge replicate labels between rounds |
| tau_s | Half-open eligibility interval, seconds |
| status, source_started | complete, partial or not_started; source boolean preserved |
| source_actual_arrivals, planned_arrival_count | Recorded arrival count versus 52 planned; checked against ledger clocks |
| observation_end_elapsed_s | End of observed episode, seconds; partial recovery can be right-censored |
| start_utc, end_utc | UTC time mappings; no account, location or network identity |
| interbatch_wait_s | Difference between previous end and current start within a round; placeholders are not usable interbatch waits |
| temperature_c, power_mode | Batch gate temperature and cabled power enum when available; null for absent gates |
| abort_reason | Existing bounded cleanup code or null; private error strings omitted |
| private_sealed_files_checked | Count of source seal manifest entries locally SHA-verified; does not count independent observations |
| raw_analysis_error_n, raw_analysis_warning_counts | Preserved raw analysis counts, including unresolved raw ledger semantics despite a separate sidecar review |
| scope | complete_batch or planned_slot_bounds_only_not_a_complete_replicate |

## Derived results and expected comparisons

`expected/batch_results.json` stores complete recomputed structures for each batch. `scoring.planned_gold_execute` has denominator, confirmed_pass, unresolved, confirmed_noncompletion, lower_rate and upper_rate. Its full denominator is 44 for every planned ledger, but only complete episodes support full-episode performance. For partial/unstarted batches the range explicitly includes missing planned slots. Conditional admitted/native-attempt results use their own denominator and retain selection limits.

The inherited scoring rule treats busy/reset/runtime-reject/cap/fail as noncompletion, and retained insufficient admitted outcomes as unresolved. A semantic pass requires a nonempty evidence reference and an appropriate branch and terminal. This keeps the historical external-block fail as noncompletion while separately identifying its cause. The new unjudgeable external block contributes to the unresolved upper bound, never confirmed success.

Budget rows separately report native observed/missing values, strict >50 seconds counts, cap counts and sidecar passes within 50 seconds. Fast requests have no 50-second budget row. Latency summaries give count, minimum, maximum, mean and median for observed values only; missing counts are separately preserved. No quantile or mean is imputed for missing native terminals.

Health includes all retained transitions, boundary coverage, initial unknown snapshots, persistent detection and recovery status, delay/censor bound, source window and first ordinary post-recovery cloud request. Fault metrics distinguish exposure, pairing and unavailable calls encountered while already degraded. Half-open metrics keep claim/result counts and probe timings; no business success is inferred from them.

Foreground observation records contain native terminal, observation start/end/duration, release clock, later actual arrival indices during the full post-terminal occupancy, and seconds to the next scheduled arrival. Package SHA comparison is exact; numerical comparisons allow absolute 1e-6 or relative 1e-10 tolerance. Window P75 reconstruction uses 0.01 ms tolerance for logged rounding.

`FIELD_WHITELIST.json` is an allowlist of exported keys, not permission to add arbitrary strings under them. The extractor separately reviewed categorical values and never copies reason/model/action/error text.

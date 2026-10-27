# Data dictionary

All counts and durations retain their original experimental scope. Empty CSV cells become `null` in analysis; JSON `null` is never scored as success or silently dropped.

| File | Unit and core fields | Scope |
| --- | --- | --- |
| `fourlayer_slots.jsonl` | One planned slot: `device`, `run_id`, `task_id`, `layer`, `mode`, `state`, `machine`, `verdict`, nullable `success`, nullable `seconds`, `cloud`, `path` | 168 slots per device. `seconds` is ledger request-to-terminal time, not visible TTFA. Original mode labels Hybrid/Slow-only are reported as Routed/Cloud-only. `cloud` retains the original ledger counter. |
| `excluded_slots.csv` | One unjudgeable slot; task/run identifiers, mode/state, selected attempt and adjudication line | Four exclusions. Original free-text reasons and local source paths omitted; slot IDs allow matching with numeric exports. |
| `energy_pairs.csv` | One active–idle pair: group/condition/repeat, requests per active batch, net mAh per batch/request, active duration in seconds | 36 observations, retained exactly. `active_seconds_request` is active batch duration divided by registered request count. |
| `energy_source_checks.csv` | Active and idle mAh, reconstructed and authority net values, difference and coverage | Derived numeric audit. High/Mid use battery counters; Low uses current integration. Not independent sensor calibration. |
| `terminal_slots.csv` | One non-Fast slot: semantic verdict, cap/censoring, ledger and raw terminal seconds, timing role, route, missingness | 432 slots; raw timing is reconstructed only where retained request and terminal events exist. |
| `window_replay_cells.csv` | One config × noise trace × profile cell; counts, step indices/delays, expected trajectory flag | 144 old replay cells. Empty exits/recoveries remain missing; steps are not seconds. |
| `router_call_latency.csv` | One matched route-call slot: device group, run/task, app state, host interval `seconds`, source-event hash | 83 matched slots per phone. Same excluded slot on all phones; no new inference. |
| `saved_router_predictions.jsonl` | One synthetic input form with group, text kind, frozen gold, clarification/family fields and text hash; per-method raw generated content, parsed label, final route and guard reason | 284 already consumed inputs. Classes 0/1/2 mean Fast/Slow/Reject. Invalid parses remain nullable. Real main-app guard is called during offline replay. |
| `historical_mixed_SEPARATE.csv` | Existing historical per-mode mean/contrast and interval summaries | High/Low use J/handled request; Mid uses J/machine-successful task. Never pool them with current mAh/request observations. |
| `historical_high_mixed_points.csv` | Historical High pair and block indicators, net J/handled request | Descriptive old Mixed points, retained separately. |
| `analysis_contract.json` | Device group mapping, budget/mixture grid, route timer and model hash | Contract metadata for exported reanalysis, with a reference to the separately reported mobile sessions under mobile_admission/. |

`expected/` contains retained reference tables. A successful verification means the bundled numerical inputs reproduce the checked counts/point metrics/interval tables within numeric tolerance. It does not establish independent semantic adjudication or model/device reproducibility. Mobile numerical verification is a separate command.

`provenance/source_map.json` records anonymous source IDs, destination roles and original source hashes; the private path mapping is withheld. Export operations remove only explicitly listed fields; original source files remain untouched. `SHA256SUMS.json` identifies every distributed file except itself.

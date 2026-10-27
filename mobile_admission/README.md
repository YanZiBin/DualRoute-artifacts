# DualRoute mobile admission observations

Status: LOCAL AUTHOR-REVIEW COPY, NOT PUBLISHED. Prepared on 2026-10-27. Numerical observations are unchanged from the reviewed mobile candidate. Licenses and packaging notices are supplied at the release root; no upload, commit, push or publication has occurred.

This package supports offline numerical reanalysis of the Honor 200 admission experiment. It contains sanitized per-slot timing and branches, independent semantic sidecar labels, numerical health/call/half-open events, and explicit source hashes. It contains no screenshots, accounts, credentials, device serials, private endpoints, machine paths, original prompts, task text or private context.

## Quick verification

Use Python 3.9 or later, with the standard library only. From the extracted package directory run:

```sh
python3 -I analysis/verify_results.py
```

The command reads only files inside this package, checks its file hashes and field whitelist, recomputes all twelve batch records, and compares them with expected results. It neither imports the application nor contacts a model, network, phone or cloud service. Use `--report` with a path outside the sealed package to write an additional local report. The supplied local validation report is outside the ZIP to avoid self-referential hash manifests.

## Observations and denominators

There are 624 planned ledger slots, of which 493 have actual scheduler arrivals. These totals are inventory counts, not pooled performance estimates.

- Original round: nine planned batches, 468 ledger slots, 337 actual arrivals. Batches o-b01 to o-b06 are complete; o-b07 is partial with 25 actual arrivals; o-b08 and o-b09 never started. The remaining 131 slots are placeholders after abort.
- Supplement round: three complete batches, 156 ledger slots and actual arrivals, in the order 180/45/90 seconds. There is one complete supplement batch per interval.
- Each complete batch has 44 execution-task slots and eight R1 refusal-task slots. Busy, runtime rejection and reset failure remain in the full scheduled execution denominator. Correct R1 refusal is reported separately.
- Original and supplement rounds must be reported separately. Nine complete batches across these rounds do not establish three same-condition replicates per interval. The network period, thermal state, execution trajectories and F1 evidence version differ. There is no mixed-round confidence interval, significance test or selected optimum.

The nine complete-batch rows in `data/complete_batch_table.json` are recomputed from the exports. All twelve batch statuses remain in `data/batches.json` and `expected/batch_results.json`.

| Round | Batch | Interval (s) | Busy | Runtime reject | Confirmed execution / 44 | Unresolved execution | Detection / recovery delay (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | o-b01 | 45 | 30 | 1 | 14 | 2 | 94.697 / 130.808 |
| Original | o-b02 | 90 | 28 | 3 | 12 | 3 | 95.059 / 218.631 |
| Original | o-b03 | 180 | 24 | 10 | 7 | 6 | 95.314 / 426.640 |
| Original | o-b04 | 90 | 30 | 4 | 10 | 3 | 95.772 / 247.450 |
| Original | o-b05 | 180 | 23 | 10 | 7 | 7 | 96.179 / 427.300 |
| Original | o-b06 | 45 | 29 | 1 | 14 | 3 | 94.708 / 130.425 |
| Supplement | s-b01 | 180 | 22 | 8 | 15 | 0 | 95.554 / 392.295 |
| Supplement | s-b02 | 45 | 32 | 1 | 15 | 0 | 96.095 / 132.017 |
| Supplement | s-b03 | 90 | 31 | 4 | 10 | 3 | 127.346 / 219.572 |

For a complete batch the completion range is confirmed/44 to (confirmed + unresolved)/44. An upper bound is not an observed success rate. Partial and unstarted batch bounds in the detailed output are planned-slot accounting only, and must not be read as complete-episode performance.

## Outcome interpretation

Raw machine flags, raw ledger semantic labels, independent sidecar verdicts, branch disposition and outcome categories are separate fields. Machine success and half-open success events do not establish independent business completion.

Human-verification barriers are separately categorized as `external_verification_blocked` and are not attributed to model/policy failure. Original o-b07-a006 keeps its historical sidecar `fail`; supplement s-b03-a039 keeps its independent `unjudgeable` label. Both remain without confirmed completion in their original fixed-arrival sequence. No challenge was solved and no author-assisted rerun is bundled. Reset operator-required states in original o-b07 are preserved as uncertain infrastructure blockers rather than presumed to be captcha.

Supplement s-b03-a006 remains a pass: the independently reviewed earlier result screen met the visible-search-results contract before an extra product-detail tap. The extra tap is explicitly flagged. Original o-b05-a040 used a matching existing search-history entry; it is flagged, without claiming fresh text entry. Original o-b06-a012 remains a cap failure despite a last-moment visible result image because native completion did not occur before cancellation.

All six newly observed F1 requests have a matching real foreground package, no observation error and zero cloud calls. Original F1 requests remain unresolved. Observation duration and the full native-terminal-to-worker-release interval are recomputed; no later actual arrivals overlapped these six intervals. This does not establish identical counterfactual trajectories without observation.

## Files

- `data/arrivals.jsonl`: 624 sanitized ledger slots, original independent verdicts and anonymous evidence references.
- `data/events.jsonl`: 5,353 selected events; numerical windows, cloud calls, half-open records, route/native terminal events and count-only device action events.
- `data/batches.json`: twelve batch statuses, identity hashes, timing and thermal metadata.
- `data/protocol_numeric.json`: public numeric protocol, task families and measurement boundaries; no original task or model prompt.
- `data/complete_batch_table.json`: nine complete batches, grouped by original/supplement round.
- `analysis/derived_methods.py`: twelve unchanged pure numerical functions extracted from the existing analysis tool, with private loaders and execution entrypoints excluded.
- `analysis/verify_results.py`: offline verifier and static scan.
- `expected/batch_results.json`: per-batch expected numerical results, backed by the private-source crosscheck.
- `provenance/source_sha256.json`: anonymous source IDs, source SHA-256 and sizes; original private files are withheld.
- `provenance/source_crosscheck.json`: checks completed locally against the retained private sources.
- `docs/DATA_DICTIONARY.md`, `docs/FIELD_WHITELIST.json`, `docs/REPRODUCTION_SCOPE.md`, `SOURCE.md`, `SHA256SUMS.json`: schema, limits and provenance.

## Material not provided

No publicly downloadable cp800 weights or training-data rights have been established by this package. Fresh inference, training, raw screenshot re-adjudication, complete sensor reconstruction, visible first-action time and energy measurement are unsupported. Live collection source is not copied because it embeds device/account/control context; only the safe offline analysis functions are provided. This package cannot directly reproduce a new phone experiment.

The earlier candidate's `application/` snapshot predates the new F1 foreground evidence. Its `main.py` and `phone_agent/experiment_runtime.py` do not contain that later observation. The supplement changed those two files for post-terminal evidence and changed `unified_acquisition.py` for the independent three-batch authorization/order. Do not call the older application snapshot byte-identical to all current phone collection source. `provenance/source_version_comparison.json` identifies all 21 recorded source roles and their original/supplement hashes: 18 unchanged, three changed. The deployed-version AST checks reported an unchanged policy/core after removal of the evidence-only additions; this is a bounded source check, not proof of identical execution trajectories. Continuous network identity and matched network conditions between rounds are unavailable. The observed unavailable-call exposures happened after degradation and do not verify a normal-to-degraded availability override.

Author-controlled code is licensed Apache-2.0 and author-controlled numerical data is licensed CC BY 4.0, as scoped by the root LICENSE, LICENSE-DATA and docs/LICENSE_SCOPE.md. No third-party rights are granted. Public access remains a separate approval and publication step. Static scan results are bounded checks of this candidate, not a review of repository history or withheld private records.

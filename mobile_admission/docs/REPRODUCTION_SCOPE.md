# Reproduction scope

This candidate adds offline numeric evidence for mobile admission experiments. The earlier candidate's statement that all nine mobile batches were uncollected described its preparation time; it does not describe the later observations bundled here. The frozen older candidate remains unchanged locally; this combined release copy has updated packaging and version/license documentation.

## Recomputed here

All twelve batch records are recalculated from sanitized ledger/event exports: actual versus scheduled arrivals; per-task routes; busy/reset/refusal/ordinary/half-open branches; full-denominator and conditional completion ranges; R1 refusal bounds; observed native timing and arrival/cleanup timing; strict native duration > 50,000 ms counts; cap cancellations; half-open claims, eligibility waits and result counts; cloud-start/end pairing and injected-call exposure; P75 windows, transitions, detection and recovery delay; first ordinary cloud request after recovery; six F1 observations and occupancy margins.

Methods are extracted unchanged from the existing offline analysis source. Private-source local checks compare exported results with source-applied scoring and saved final numeric analyses. The original first-batch call-fault correction is taken from the final original-run analysis, not the earlier snapshot that omitted cloud_step availability exposure. Supplement scoring is also compared with the saved final independent review. The public verifier checks only the sanitized package and frozen expected results; it cannot reconstruct withheld private sources from anonymous hashes.

## Experimental units and interpretation

Original batch order: 45/90/180/90/180/45/180/45/90 seconds. The first six completed, the seventh stopped after 25 arrivals, and the last two never started. Supplement order: 180/45/90 seconds, one complete batch at each interval. Original complete batches have two observations per interval; supplement batches have one per interval. Separate rounds do not supply a same-condition n=3 experiment.

The fixed schedule has 52 slots at 30-second intervals and no queue, catch-up or replacement. Full execution-task denominators are 44 per complete batch. R1 contains eight separate semantic-refusal slots. Admission-conditioned results select a branch of the observed trajectory and must not replace full-arrival service results. Placeholder upper bounds in partial/unstarted records describe missing planned slots, not observed task completion.

Health P75 uses linear interpolation at index (n-1)*0.75. The window is capped at ten samples, with at least five required to enter/exit. Enter when P75 > 13,000 ms; exit when P75 < 8,000 ms. Detection is measured from elapsed 360 seconds, recovery from 720 seconds. Every complete batch exhibits two transitions; no isolated-jitter transition is observed. Unavailable calls exposed after degradation cannot prove the availability override's normal-to-degraded operation.

The cap is measured from acceptance, whereas native_e2e_ms is the application's native task duration and excludes reset and later cleanup. Cancellation rows can lack a native task terminal; a missing cloud end is not fabricated. The verifier preserves these incomplete pairs and warning counts. A captured result screen before a cap does not override a frozen cap-failure criterion.

## Semantic evidence and barriers

The sidecar labels are supplied observations made during the prior independent local review. Hashes establish source identity, not correctness of the visual judgments or event-emission completeness. Screenshots, task text, model output and device action arguments are withheld, so an external reader cannot independently re-adjudicate semantic success from this package alone.

Original complete batches retain 15 unresolved F1 requests and nine insufficient W2 detail results; the partial batch adds one of each. The supplement supplies six independent F1 foreground matches, and s-b03 retains two insufficient W2 details and one external human-verification block. The original captcha fail label remains unchanged and has an additional external-block category. Neither historical failure nor new uncertainty is silently changed to success. The two rounds retain separate attribution fields.

The six supplement foreground observations occurred after native task_terminal. Their durations are recomputed from their own start/end values. Full terminal-to-release occupancy is checked against all later actual arrivals and the next scheduled offset. No overlap was observed; no counterfactual equivalence is asserted.

## Missing and unavailable components

- The cp800 Q8 artifact is identified by hash, but no public weight download is provided. No training data or weight redistribution rights are established; exported observations have the separately scoped data license at the release root.
- Original prompts, task text, private context, screenshots and account/device records are not included. Models are not called by verification.
- Safe numerical analysis source is included. Full live collection/deployment/control source is withheld because it depends on private device/reset/endpoint context and operator setup. It cannot be safely presented as a turnkey phone reproduction. The existing application snapshot is outside this separate candidate and is not duplicated. That earlier snapshot predates the supplement's F1 evidence in main.py and experiment_runtime.py; unified_acquisition.py also changed for independent authorization/order. Recorded source hashes show 18 unchanged roles and three changed roles. No claim is made that the old snapshot reproduces the supplement's evidence version.
- A complete continuous network record is absent. Observations at a few times do not prove a stable network; original and supplement network conditions cannot be treated as matched. Warmth, order and evidence-version changes remain confounders.
- No visible first-action time, energy, battery reconstruction, training, fresh policy experiment, captcha-bypass test, author-assisted rerun or mixed-round statistical inference is performed.
- Private manifests were verified locally during export, but anonymous source IDs do not let an outside reader acquire or re-hash those originals. Rule scans cover this candidate only.

This is an unpublished local author-review copy. Code/data license terms are stated at the release root; an upload or public release needs a separate final-package/destination approval. The historical private-branch authorization does not authorize publication.

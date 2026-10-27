# Sources, transformations and SHA-256

Private source material was read only. New output is confined to this separate candidate directory. No original candidate, source code, ledger, event stream or sidecar was edited.

`provenance/source_sha256.json` assigns anonymous English source IDs to original files and gives SHA-256, byte size, role and batch identity. It intentionally includes no original filesystem paths. The IDs cover original final-run analysis, supplement final review, per-batch episode and seal metadata, arrival ledgers, runtime/controller streams, semantic sidecars, local review records and withheld semantic evidence. Private source content is not bundled. These hashes identify the local sources; they are not public download links.

The exporter verified 5,203 sealed manifest entries across twelve batches, and separately checked sidecar-to-ledger hashes and every retained semantic-reference digest. See `provenance/source_crosscheck.json` for exact counts. Duplicate manifest/reference appearances are checks, not counts of independent tasks or unique images. The original final-run analysis is authoritative for the documented first-batch cloud_step availability correction.

## Whitelist transformation

- Replace run/request/event/call IDs with deterministic batch-scoped public IDs; preserve joins and all timing values.
- Keep only listed scalar timing/branch/status fields, selected numerical event lists and public enum values.
- Keep sidecar verdicts verbatim and raw ledger verdicts separately. Add a separate controlled category for the historical external human-verification block; do not edit its original fail label.
- Replace image/event evidence paths with anonymous source IDs and SHA-256. Do not copy images or original files.
- Drop task text, original prompts, context, raw model output, action payloads, logs, error text, device identity, addresses and endpoints. Cloud errors become an error_present boolean. Post-terminal foreground observations are the only app-package strings retained.
- Preserve partial/unstarted slots, nulls, cancellation, warning counts, cleanup failure, uncertain detail evidence and external blocks. Preserve extra product tap and existing search-history flags.

The local builder is outside the redistributable `package/` and ZIP because it accesses private source material supplied as an explicit command argument. It is an extraction aid, not an external reproduction entrypoint. This release places those reviewed contents under mobile_admission/ and adds notices, updated scope text and a regenerated integrity manifest; numerical payloads remain unchanged.

## Included source

`analysis/derived_methods.py` is prominently marked as modified. Twelve pure functions are extracted with unchanged bodies from the existing offline analysis tool. The function-source SHA-256 map is in `provenance/method_extraction.json`. The three constant sets are public task/scheduling definitions. Private loaders, fixed paths, model/device/runtime imports, fixture generators and collection entrypoints are excluded. `analysis/verify_results.py` adds package-only I/O, numerical comparison, P75 recomputation, foreground timing checks, file hash verification and a bounded static scan.

The included functions are numerical analysis, not acquisition or algorithm changes. Live acquisition code has not been represented as runnable public code: private deployment/reset/context dependencies and original prompts remain withheld. No weights or training data are supplied. The release root defines Apache-2.0 for author-controlled code and CC BY 4.0 for author-controlled exported data; third-party rights are excluded. Public distribution still needs separate approval of the final package and destination.

## Integrity layers

`SHA256SUMS.json` hashes every package file other than itself. It is a file-integrity inventory, not a digital signature. The local ZIP receipt outside the ZIP records the archive hash and verifies byte equality for every archived member. Local validation reports are outside the package to avoid recursive report/hash dependencies. File hashes do not independently prove semantic labels or complete event emission.

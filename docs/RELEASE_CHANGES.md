# Changes in this local release copy

This unpublished copy is derived from the two reviewed frozen archives identified in `provenance/input_archives.json`. The original archives, raw observations and private runtime sources remain unchanged.

- Replaced embedded routing/execution research prompts with explicit operator-supplied file loaders in Python and C++. Offline imports do not load a prompt. Missing/empty configuration fails before local model loading; no empty substitute is supplied.
- Retained every FastPathRouter parser/guard method and the extracted numerical functions. Static AST comparisons and actual saved-output verification are reported separately in `validation.json`.
- Distinguished original research core hashes from packaging-adjusted release hashes. Historical expected results remain unchanged; the replay verifies the current release identity through `provenance/release_identity.json`.
- Adapted the existing prompt synchronization test to verify the release's missing-prompt boundary and supplied fixture behavior. In local review v2, added prompt checks before LocalRouteModel.load() enters its loader and before the validation script resolves a device, reads cases or loads weights. Targeted tests assert no downstream load or case I/O on missing configuration; the validation-main test executes the exact function body with explicit stubs for unbundled ML dependencies and does not claim a real weight-loading test. Updated the existing keyboard-session test to supply its explicit fixture. Corrected an inherited AST test's stale assumption about the helper function location; application runtime code was not changed for that AST-test correction.
- Added source notices, code/data licenses, citation metadata and the missing application README. Replaced setup's placeholder URL with the upstream project URL. Source entrypoints and metadata were checked; wheel/install or live execution portability is not claimed.
- Updated stale zero-mobile status, scope and version descriptions. Original and supplemental mobile sessions remain separate. Internal source paths were replaced by anonymous IDs; their private mapping is excluded.
- Updated the mobile scan to allow explicitly listed public Apache license links and to require a Windows drive-letter token boundary, avoiding a URL false positive. The data field whitelist and numerical checks remain active.

29 numerical observation/expected-result files remain byte-identical to the frozen candidates. The analysis contract's mobile metadata is updated separately. Stored bootstrap and historical Mixed intervals remain reference results, not newly recomputed estimates.

`provenance/release_changes.json` records all 128 original input members and changes. Generated manifests/validation are identified separately to avoid circular hashes; final file identities come from the root SHA256SUMS manifest and the archive receipt outside the payload. New release documents have no original input hash. The root manifest covers them.

No new model prediction, training, phone/cloud operation, public repository write or publication is part of preparation. All remaining limitations and actual commands are listed in `validation.json` and the reproduction guide.

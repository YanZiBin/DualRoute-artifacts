# Artifact versions and identities

1. **Compatibility application snapshot.** The original 109-file candidate contains 61 source roles matching the 25 October deployment baseline, plus license/tests/docs/data. This release starts from that snapshot and changes prompt loading, packaging, source notices and documentation. It is not byte-identical to the private runtime.
2. **Original formal mobile acquisition.** Across 19 comparable application roles in the mobile source map, the old application candidate differs in `phone_agent/agent.py` and `phone_agent/experiment_runtime.py`.
3. **Supplemental formal acquisition.** Across the same 19 roles, the old application candidate differs in those two files and `main.py`.
4. **Original versus supplement source map.** A different comparison of 21 roles reports 18 unchanged and three changed roles (`main.py`, `experiment_runtime.py`, and the acquisition orchestrator). This must not be reused as a candidate-to-current comparison.

The formal original round has six complete, one partial (25 arrivals), and two unstarted batches. The supplement has three complete batches at 180/45/90 s. Numerical observations retain both sessions. Model, rules and controlled settings match where recorded, but actual networks, thermal states, time/order, GUI trajectories and evidence versions differ. No same-condition n=3 pooling is justified.

`provenance/release_identity.json` records original research core hashes and packaging-adjusted release hashes separately. `provenance/release_changes.json` records changes relative to the two frozen input candidates. `mobile_admission/provenance/source_version_comparison.json` identifies the formal source roles. None of these hashes publishes withheld source content or certifies a fresh device run.

Prompt loaders and metadata are modified for this release. All FastPathRouter parser/guard methods and the extracted numerical module ASTs are compared with the frozen candidate. Functional evidence comes from rerunning the actual saved-output verifier and existing tests on this release copy, not from AST equality alone. Root validation records actual executed commands and limitations.

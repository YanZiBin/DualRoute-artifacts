# DualRoute research artifacts

**Research artifact release v0.1.0, 27 October 2026.**

This copy accompanies *DualRoute: Separating Execution Boundaries and Cloud Admission for Mobile GUI Agents*, by Zibin Yan and Dan Gui. It provides a prompt-excluded application snapshot, exported observations, and offline numerical verification scripts. The versioned release is [v0.1.0](https://github.com/YanZiBin/DualRoute-artifacts/releases/tag/v0.1.0). Use that release for the exact source snapshot and the original reviewed ZIP. No DOI is assigned.

## What is included

| Input | Observations | Checked reproduction scope |
| --- | ---: | --- |
| Three-phone system benchmark | 504 planned slots | Denominators, exclusions, completion bounds; all four unjudgeable slots and 22 timeouts retained |
| Charge measurements | 36 active-idle pairs | Mean Student t intervals and independent Welch differences; three repeats per device/condition |
| Terminal-time budgets | 432 non-Fast slots | 30/40/50/60 s recounts with missing values, cancellation and recovered timing kept separate |
| Existing sensitivity replay | 144 cells | Reaggregation of existing fixed-input results; no new detector experiment |
| Router-call intervals | 249 | 83 matched slots per phone; host interval, not visible first-action time |
| Consumed synthetic routing set | 142 paired groups, 284 texts | Saved-output replay through the actual parser and full guard; no new model calls |
| Mobile admission observations | 12 batch records, 624 planned slots, 493 actual arrivals | Original six complete/one partial/two unstarted and supplemental three complete, reported separately |

The mobile export contains 5,353 selected numerical events. Nine complete batches across two sessions do not establish three same-condition replicates per interval. Busy requests, runtime refusals, caps, cleanup failure, unresolved screens and external human-verification blocks remain explicit. All unavailable-call exposures occurred after degradation; an availability-triggered Normal-to-Degraded transition is not established.

## Offline verification

Use Python 3.10+ and the pinned NumPy/SciPy versions in `requirements-analysis.txt` for the first command. The second command uses the Python standard library. Both commands read their bundled data, with no phone, model or cloud calls. The first uses temporary files for mixture comparisons and does not change observations.

```bash
python -I -B analysis/verify_results.py
```

```bash
python -I -B mobile_admission/analysis/verify_results.py
```

The first command reports `PASS_EXPORTED_DATA_RECOMPUTATION` and checks 13 reference groups. The second reports `PASS` and checks all 12 mobile batches, including 1,762 P75 windows. The root `validation.json` preserves the checks run on the approved local v2 copy before publication; its unpublished/uploaded fields describe that historical validation stage. Publication changes are limited to the README, release-status and citation documents, plus the regenerated root manifest. Code and observations are unchanged. Public publication and anonymous-download checks are recorded separately in the release notes. `SHA256SUMS.json` covers final package files; the mobile subdirectory also has its own manifest. Do not write reports or bytecode inside the sealed package.

## Application and prompt boundaries

`application/` starts from the reviewed 25 October compatibility snapshot. It is adjusted for this release and is not the full source of either formal mobile acquisition. Research prompts have been removed from Python and C++ sources. Live inference requires the operator's own nonempty UTF-8 prompt files; a missing configuration raises an error rather than supplying an empty or substitute research prompt. No prompt configuration is needed for saved-output replay or CLI help.

Parser/guard methods and numerical functions are retained. `provenance/release_identity.json` distinguishes original research hashes from final release-copy hashes. `docs/ARTIFACT_VERSIONS.md` describes the compatibility snapshot, original formal acquisition and supplement separately. Changes to packaging are not represented as new experiments or validated inference equivalence.

## Limits

The scripts do not train classifiers, repeat cp800 inference, reconstruct original sensors, recreate every figure, independently re-adjudicate withheld screenshots, or reproduce a fresh phone experiment. Stored bootstrap intervals and historical Mixed-study summaries remain reference results; the legacy verifier checks declared point metrics and selected intervals only. No visible-TTFA or new charge experiment is added.

Weights, training corpora, original research prompts, screenshots, accounts, credentials, device serials, private endpoints, original deployment/reset context, unopened holdouts and sealed SP-1 material are excluded. Private-source hashes establish identity, not independent correctness of semantic labels. Complete continuous/matched networks are unavailable across sessions.

## Licenses and source

Author-controlled code is Apache-2.0 (`LICENSE`). Author-controlled exported data is CC BY 4.0 (`LICENSE-DATA`). Upstream Zhipu AI/Open-AutoGLM notices remain; third-party rights and unbundled assets are excluded. See `NOTICE`, `docs/LICENSE_SCOPE.md`, `docs/REPRODUCTION_SCOPE.md`, the data dictionaries and `CITATION.cff`.

# DualRoute application source snapshot

This is a prompt-excluded release copy of the 25 October compatibility application snapshot, derived from Open-AutoGLM under Apache-2.0. It is provided for code inspection and saved-output parser/guard replay, not as the full source of later formal phone acquisition.

Private routing/execution prompts and model weights are not distributed. The release loads operator-supplied nonempty UTF-8 files using `DUALROUTE_ROUTE_PROMPT_FILE`, `DUALROUTE_EXECUTION_PROMPT_ZH_FILE` and `DUALROUTE_EXECUTION_PROMPT_EN_FILE`. Missing configuration fails explicitly when inference/agent construction needs it. CLI help, app listing and offline replay require no prompt/model/device.

From this directory, the declared source entrypoint is:

```bash
python -B main.py --help
```

```bash
python -B main.py --list-apps
```

These commands are inspected locally; a live phone task has additional dependencies and is not verified by release preparation. The default model filename in legacy scripts is historical and must be configured explicitly. No one-command live deployment or new model reproduction is promised.

The inherited `setup.py` now finds this README and uses the upstream project URL. Its package name and console entrypoint remain historical; metadata inspection does not establish wheel/install portability. Use the documented source entrypoint. See `../docs/RUNNING_THE_APPLICATION.md`.

`README.upstream.md` is retained as historical upstream text, including source attribution and citation. Some of its relative assets are outside this bounded snapshot. For those resources, consult the fixed reference tree at https://github.com/zai-org/Open-AutoGLM/tree/86f55382982fb054e8fc98ca80609dff8a2cdc3c . That comparison reference is not asserted to be the research fork point. The supported local instructions are this README and the release docs, not missing historical asset links.

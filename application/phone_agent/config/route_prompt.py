# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# Modified by the DualRoute authors for the prompt-excluded release; numerical/parser/guard scope is documented in docs/REPRODUCTION_SCOPE.md.
"""Load an operator-supplied routing prompt; research prompt text is withheld."""
from phone_agent.config import _read_prompt_file


def get_route_prompt() -> str:
    return _read_prompt_file("DUALROUTE_ROUTE_PROMPT_FILE")

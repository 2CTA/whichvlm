"""Compatibility re-exports for output surfaces."""

from whichvlm.output._console import console
from whichvlm.output.json_output import (
    display_json,
    display_plan_json,
    display_upgrade_json,
)
from whichvlm.output.markdown import display_markdown
from whichvlm.output.plan import display_plan
from whichvlm.output.ranking import display_hardware, display_ranking
from whichvlm.output.upgrade import display_upgrade

__all__ = [
    "console",
    "display_hardware",
    "display_json",
    "display_markdown",
    "display_plan",
    "display_plan_json",
    "display_ranking",
    "display_upgrade",
    "display_upgrade_json",
]

"""Training objectives and protocol helpers for DriftDiff."""

from .objectives import (
    ERM,
    GroupDRO,
    LossBreakdown,
    VREx,
    environment_loss,
    topk_subgroup_preservation_penalty,
)
from .protocol import PreparedEnvironment, prepare_environment

__all__ = [
    "GroupDRO",
    "ERM",
    "VREx",
    "LossBreakdown",
    "PreparedEnvironment",
    "environment_loss",
    "prepare_environment",
    "topk_subgroup_preservation_penalty",
]

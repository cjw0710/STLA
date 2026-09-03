"""Past-only validation stress transformations for temporal popularity state."""

from __future__ import annotations

from dataclasses import replace

import torch

from .metrics import popularity_group_ids, recency_group_ids
from .training import PreparedEnvironment


STRESSES = (
    "recent_hub_amplification",
    "recent_hub_turnover",
    "emerging_influx",
)


def _ordered_nodes(mask: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
    nodes = torch.nonzero(mask, as_tuple=False).flatten()
    if nodes.numel() == 0:
        return nodes
    order = torch.argsort(score[nodes], descending=True, stable=True)
    return nodes[order]


def _transfer_recent_mass(
    recent: torch.Tensor,
    donors: torch.Tensor,
    recipients: torch.Tensor,
    severity: float,
) -> torch.Tensor:
    shifted = recent.clone()
    if donors.numel() == 0 or recipients.numel() == 0 or severity == 0:
        return shifted
    amounts = shifted[donors] * severity
    shifted[donors] -= amounts
    recipient_index = torch.arange(
        donors.numel(), device=recent.device
    ) % recipients.numel()
    shifted.index_add_(0, recipients[recipient_index], amounts)
    return shifted


def perturb_recent_popularity(
    historical: torch.Tensor,
    recent: torch.Tensor,
    stress: str,
    severity: float,
) -> torch.Tensor:
    """Counterfactually perturb only recent past activity, never target labels."""

    if stress not in STRESSES:
        raise ValueError(f"unknown stress: {stress}")
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must be in [0, 1]")
    if historical.ndim != 1 or recent.shape != historical.shape:
        raise ValueError("historical and recent popularity must be equal vectors")
    if bool(torch.any(historical < 0)) or bool(torch.any(recent < 0)):
        raise ValueError("popularity cannot be negative")

    groups = popularity_group_ids(historical)
    head = groups == 0
    donors = _ordered_nodes(head & (recent > 0), recent)
    if stress == "recent_hub_amplification":
        shifted = recent.clone()
        shifted[head] *= 1.0 + 3.0 * severity
        return shifted
    if stress == "recent_hub_turnover":
        recipients = _ordered_nodes(
            ((groups == 1) | (groups == 2)) & (historical > 0),
            historical,
        )
        return _transfer_recent_mass(recent, donors, recipients, severity)

    recipients = torch.nonzero(historical == 0, as_tuple=False).flatten()
    return _transfer_recent_mass(recent, donors, recipients, severity)


def perturb_environment(
    environment: PreparedEnvironment,
    stress: str,
    severity: float,
) -> PreparedEnvironment:
    """Return a validation environment with perturbed past-only recent state.

    Graph tensors, environment context, target cascades, and cumulative history
    are intentionally fixed. This isolates sensitivity of the temporal residual
    and hierarchical safety metadata rather than simulating a full new world.
    """

    recent = perturb_recent_popularity(
        environment.historical_popularity,
        environment.recent_popularity,
        stress,
        severity,
    )
    return replace(
        environment,
        recent_popularity=recent,
        popularity_groups=popularity_group_ids(environment.historical_popularity),
        recency_groups=recency_group_ids(
            environment.historical_popularity,
            recent,
        ),
    )

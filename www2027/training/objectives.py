"""Environment-aware optimization objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..models.temporal_diffusion import DiffusionOutput


@dataclass
class LossBreakdown:
    total: torch.Tensor
    prediction: torch.Tensor
    unweighted_prediction: torch.Tensor
    constraint_penalty: torch.Tensor
    shortcut: torch.Tensor
    independence: torch.Tensor
    mask_balance: torch.Tensor


def topk_subgroup_preservation_penalty(
    output: DiffusionOutput,
    popularity_groups: torch.Tensor,
    recency_groups: torch.Tensor,
    *,
    topk: int = 100,
    margin: float = 0.0,
) -> torch.Tensor:
    """Keep vulnerable candidates from the anchored base model in its top-k.

    Candidate groups are computed from past-only statistics: popularity group
    zero is head and recency group zero is recent-active. The loss activates
    only after a base top-k mid/tail/dormant/emerging candidate falls below the
    student's current top-k boundary.
    """

    if topk < 1 or margin < 0:
        raise ValueError("topk must be positive and margin cannot be negative")
    if popularity_groups.ndim != 1 or recency_groups.shape != popularity_groups.shape:
        raise ValueError("group vectors must be one-dimensional and equal-sized")
    if output.logits.shape != output.temporal_prior_logits.shape:
        raise ValueError("logits and temporal residual must have equal shapes")
    if output.logits.shape[1] != popularity_groups.numel():
        raise ValueError("group vectors must contain one entry per candidate node")

    effective_k = min(topk, output.logits.shape[1])
    anchored_topk = torch.topk(
        output.base_logits.detach(),
        effective_k,
        dim=1,
    ).indices
    protected = (popularity_groups[anchored_topk] != 0) | (
        recency_groups[anchored_topk] != 0
    )
    if not bool(torch.any(protected)):
        return output.logits.new_zeros(())

    student_scores = output.logits.gather(1, anchored_topk)
    student_boundary = torch.topk(
        output.logits.detach(),
        effective_k,
        dim=1,
    ).values[:, -1:]
    violations = torch.relu(student_boundary + margin - student_scores)
    return violations[protected].mean()


def environment_loss(
    output: DiffusionOutput,
    target: torch.Tensor,
    local_popularity: torch.Tensor,
    historical_popularity: torch.Tensor | None = None,
    recent_popularity: torch.Tensor | None = None,
    popularity_groups: torch.Tensor | None = None,
    recency_groups: torch.Tensor | None = None,
    *,
    shortcut_weight: float = 0.1,
    independence_weight: float = 0.05,
    mask_balance_weight: float = 0.01,
    popularity_balance_alpha: float = 0.0,
    dormant_boost: float = 0.0,
    constraint_weight: float = 0.0,
    constraint_margin: float = 0.5,
) -> LossBreakdown:
    """Prediction plus explicit shortcut and separation supervision."""

    if min(popularity_balance_alpha, dormant_boost, constraint_weight, constraint_margin) < 0:
        raise ValueError("balancing and constraint hyperparameters cannot be negative")
    sample_prediction = F.cross_entropy(output.logits, target, reduction="none")
    unweighted_prediction = sample_prediction.mean()
    sample_weight = torch.ones_like(sample_prediction)
    if popularity_balance_alpha or dormant_boost:
        if historical_popularity is None or recent_popularity is None:
            raise ValueError("balanced loss requires historical and recent popularity")
        historical_target = historical_popularity[target].float().clamp_min(0)
        recent_target = recent_popularity[target].float().clamp_min(0)
        if popularity_balance_alpha:
            sample_weight = sample_weight * torch.pow(
                historical_target + 1.0,
                -popularity_balance_alpha,
            )
        if dormant_boost:
            dormant = (historical_target > 0) & (recent_target == 0)
            sample_weight = sample_weight * (1.0 + dormant_boost * dormant.float())
        sample_weight = sample_weight / sample_weight.mean().clamp_min(1e-12)
    prediction = torch.mean(sample_prediction * sample_weight)

    constraint_penalty = sample_prediction.new_zeros(())
    if constraint_weight:
        if popularity_groups is None or recency_groups is None:
            raise ValueError("constraint loss requires precomputed past-only groups")
        target_popularity_group = popularity_groups[target]
        target_recency_group = recency_groups[target]
        group_constraints: list[torch.Tensor] = []
        # Head=0; constrain the aggregate risk of mid/tail/emerging users.
        underserved = target_popularity_group != 0
        if bool(torch.any(underserved)):
            group_constraints.append(
                torch.relu(
                    sample_prediction[underserved].mean()
                    - unweighted_prediction
                    - constraint_margin
                )
            )
        # Recent-active=0; constrain dormant and emerging-user risk separately.
        nonrecent = target_recency_group != 0
        if bool(torch.any(nonrecent)):
            group_constraints.append(
                torch.relu(
                    sample_prediction[nonrecent].mean()
                    - unweighted_prediction
                    - constraint_margin
                )
            )
        if group_constraints:
            constraint_penalty = torch.stack(group_constraints).mean()
            prediction = prediction + constraint_weight * constraint_penalty

    popularity = local_popularity.float().clamp_min(0)
    if bool(popularity.sum() == 0):
        popularity = torch.ones_like(popularity)
    popularity = popularity / popularity.sum()
    shortcut_log_probability = F.log_softmax(output.shortcut_logits, dim=-1)
    shortcut = -(shortcut_log_probability * popularity.unsqueeze(0)).sum(dim=-1).mean()

    cosine = F.cosine_similarity(
        output.stable_representation,
        output.shortcut_representation,
        dim=-1,
    )
    independence = cosine.square().mean()
    mask_balance = (output.edge_mask.mean() - 0.5).square()
    total = (
        prediction
        + shortcut_weight * shortcut
        + independence_weight * independence
        + mask_balance_weight * mask_balance
    )
    return LossBreakdown(
        total=total,
        prediction=prediction,
        unweighted_prediction=unweighted_prediction,
        constraint_penalty=constraint_penalty,
        shortcut=shortcut,
        independence=independence,
        mask_balance=mask_balance,
    )


class GroupDRO(nn.Module):
    """Exponentiated-gradient weighting of temporal environment risks."""

    def __init__(self, num_environments: int, step_size: float = 0.05) -> None:
        super().__init__()
        if num_environments < 1 or step_size <= 0:
            raise ValueError("num_environments and step_size must be positive")
        self.step_size = step_size
        self.register_buffer("log_weights", torch.zeros(num_environments))

    @property
    def weights(self) -> torch.Tensor:
        return torch.softmax(self.log_weights, dim=0)

    def forward(
        self,
        environment_losses: torch.Tensor,
        risk_for_update: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if environment_losses.ndim != 1 or environment_losses.numel() != self.log_weights.numel():
            raise ValueError("environment_losses has the wrong shape")
        if risk_for_update is None:
            risk_for_update = environment_losses
        if risk_for_update.shape != environment_losses.shape:
            raise ValueError("risk_for_update has the wrong shape")
        if self.training:
            with torch.no_grad():
                self.log_weights.add_(self.step_size * risk_for_update.detach())
                self.log_weights.sub_(self.log_weights.mean())
        return torch.sum(self.weights.detach() * environment_losses)


class ERM(nn.Module):
    """Uniform empirical risk across temporal environments."""

    def __init__(self, num_environments: int) -> None:
        super().__init__()
        if num_environments < 1:
            raise ValueError("num_environments must be positive")
        self.register_buffer(
            "uniform_weights",
            torch.full((num_environments,), 1.0 / num_environments),
        )

    @property
    def weights(self) -> torch.Tensor:
        return self.uniform_weights

    def forward(
        self,
        environment_losses: torch.Tensor,
        risk_for_update: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if environment_losses.ndim != 1 or environment_losses.numel() != self.uniform_weights.numel():
            raise ValueError("environment_losses has the wrong shape")
        if risk_for_update is not None and risk_for_update.shape != environment_losses.shape:
            raise ValueError("risk_for_update has the wrong shape")
        return torch.sum(self.uniform_weights * environment_losses)


class VREx(nn.Module):
    """Mean risk plus a smooth variance penalty across environments."""

    def __init__(self, num_environments: int, penalty_weight: float = 1.0) -> None:
        super().__init__()
        if num_environments < 2:
            raise ValueError("VREx requires at least two environments")
        if penalty_weight < 0:
            raise ValueError("penalty_weight cannot be negative")
        self.penalty_weight = penalty_weight
        self.register_buffer(
            "uniform_weights",
            torch.full((num_environments,), 1.0 / num_environments),
        )

    @property
    def weights(self) -> torch.Tensor:
        return self.uniform_weights

    def forward(
        self,
        environment_losses: torch.Tensor,
        risk_for_update: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if environment_losses.ndim != 1 or environment_losses.numel() != self.uniform_weights.numel():
            raise ValueError("environment_losses has the wrong shape")
        risks = environment_losses if risk_for_update is None else risk_for_update
        if risks.shape != environment_losses.shape:
            raise ValueError("risk_for_update has the wrong shape")
        mean_loss = torch.sum(self.uniform_weights * environment_losses)
        risk_variance = torch.var(risks, unbiased=False)
        return mean_loss + self.penalty_weight * risk_variance

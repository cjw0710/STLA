"""Exact next-user ranking metrics without materializing sorted rankings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch


POPULARITY_GROUPS = ("head", "mid", "tail", "emerging")
RECENCY_GROUPS = ("recent_active", "historical_inactive", "emerging")


def protected_union_scores(
    adaptive_logits: torch.Tensor,
    anchor_logits: torch.Tensor,
    popularity_groups: torch.Tensor,
    recency_groups: torch.Tensor,
    *,
    topk: int = 100,
    protected_cutoffs: Iterable[int] | None = None,
) -> torch.Tensor:
    """Fuse adaptive rankings with subgroup-safe anchor prefixes.

    At every requested cutoff, each non-head or non-recent candidate in the
    corresponding anchor prefix is guaranteed to remain in the fused prefix.
    All group labels are derived from past-only node statistics. With no
    ``protected_cutoffs``, the original single-top-k behavior is retained.
    """

    if adaptive_logits.ndim != 2 or anchor_logits.shape != adaptive_logits.shape:
        raise ValueError("adaptive and anchor logits must have equal [B, N] shapes")
    if topk < 1:
        raise ValueError("topk must be positive")
    if popularity_groups.ndim != 1 or recency_groups.shape != popularity_groups.shape:
        raise ValueError("group vectors must be one-dimensional and equal-sized")
    if popularity_groups.numel() != adaptive_logits.shape[1]:
        raise ValueError("group vectors must contain one entry per candidate")

    if protected_cutoffs is None:
        requested_cutoffs = (topk,)
    else:
        requested_cutoffs = tuple(sorted(set(protected_cutoffs)))
        if not requested_cutoffs:
            raise ValueError("protected_cutoffs must not be empty")
        if any(cutoff < 1 or cutoff > topk for cutoff in requested_cutoffs):
            raise ValueError("protected cutoffs must be positive and no larger than topk")

    effective_k = min(topk, adaptive_logits.shape[1])
    effective_cutoffs = tuple(
        sorted({min(cutoff, effective_k) for cutoff in requested_cutoffs})
    )
    anchor_topk = torch.topk(
        anchor_logits.detach(), effective_k, dim=1
    ).indices
    protected = (popularity_groups[anchor_topk] != 0) | (
        recency_groups[anchor_topk] != 0
    )
    if not bool(torch.any(protected)):
        return adaptive_logits

    adaptive_topk = torch.topk(
        adaptive_logits.detach(), effective_k, dim=1
    ).indices
    # Start with the adaptive list and fill its prefixes from smallest to
    # largest. A protected item already present later in the list is swapped
    # forward, preserving uniqueness. An absent item replaces the lowest-ranked
    # non-required candidate in that prefix. Because required sets are nested,
    # later passes cannot invalidate a guarantee established at a smaller K.
    # Transfer the small B x K index blocks once. Calling ``tolist`` on every
    # GPU row separately introduces hundreds of device synchronizations and can
    # make the deterministic fusion slower than the model forward itself.
    selected_rows = adaptive_topk.detach().cpu().tolist()
    anchor_rows = anchor_topk.detach().cpu().tolist()
    protected_rows = protected.detach().cpu().tolist()
    for selected_row, anchor_row, protected_row in zip(
        selected_rows, anchor_rows, protected_rows
    ):
        for cutoff in effective_cutoffs:
            required = [
                node
                for node, is_protected in zip(
                    anchor_row[:cutoff], protected_row[:cutoff]
                )
                if is_protected
            ]
            required_set = set(required)
            prefix_set = set(selected_row[:cutoff])
            missing = [node for node in required if node not in prefix_set]
            if not missing:
                continue
            replace_positions = [
                position
                for position in range(cutoff - 1, -1, -1)
                if selected_row[position] not in required_set
            ][: len(missing)]
            if len(replace_positions) != len(missing):
                raise RuntimeError("protected union could not allocate its safety slots")
            for node, position in zip(missing, replace_positions):
                try:
                    later_position = selected_row.index(node, cutoff)
                except ValueError:
                    selected_row[position] = node
                else:
                    selected_row[position], selected_row[later_position] = (
                        selected_row[later_position],
                        selected_row[position],
                    )
    selected = torch.tensor(
        selected_rows,
        device=adaptive_topk.device,
        dtype=adaptive_topk.dtype,
    )

    # Make the constructed list the exact top-k while retaining its order. The
    # metric layer consumes ranks only; values outside the selected list remain
    # unchanged and strictly below the assigned selected scores.
    fused = adaptive_logits.clone()
    maximum = torch.max(adaptive_logits.detach(), dim=1, keepdim=True).values
    rank_score = torch.arange(
        effective_k,
        0,
        -1,
        device=adaptive_logits.device,
        dtype=adaptive_logits.dtype,
    ).reshape(1, -1)
    rank_score = maximum + 1.0 + rank_score / effective_k
    fused.scatter_(1, selected, rank_score)
    return fused


@dataclass
class RankingAccumulator:
    cutoffs: tuple[int, ...] = (10, 50, 100)
    count: int = 0
    hit_sums: dict[int, float] = field(default_factory=dict)
    map_sums: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.cutoffs or any(cutoff < 1 for cutoff in self.cutoffs):
            raise ValueError("cutoffs must be positive")
        self.hit_sums = {cutoff: 0.0 for cutoff in self.cutoffs}
        self.map_sums = {cutoff: 0.0 for cutoff in self.cutoffs}

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        if logits.ndim != 2 or target.ndim != 1 or logits.shape[0] != target.shape[0]:
            raise ValueError("logits and target batch dimensions do not match")
        target_score = logits.gather(1, target.unsqueeze(1))
        ranks = 1 + torch.sum(logits > target_score, dim=1)
        ranks = ranks.detach().cpu()
        self.count += int(ranks.numel())
        for cutoff in self.cutoffs:
            hits = ranks <= cutoff
            self.hit_sums[cutoff] += float(hits.sum())
            self.map_sums[cutoff] += float(
                torch.where(hits, 1.0 / ranks.float(), torch.zeros_like(ranks.float())).sum()
            )

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            raise ValueError("no ranking examples were accumulated")
        metrics: dict[str, float] = {}
        for cutoff in self.cutoffs:
            metrics[f"hit@{cutoff}"] = self.hit_sums[cutoff] / self.count
            metrics[f"map@{cutoff}"] = self.map_sums[cutoff] / self.count
        return metrics

    def merge(self, other: "RankingAccumulator") -> None:
        if self.cutoffs != other.cutoffs:
            raise ValueError("cannot merge accumulators with different cutoffs")
        self.count += other.count
        for cutoff in self.cutoffs:
            self.hit_sums[cutoff] += other.hit_sums[cutoff]
            self.map_sums[cutoff] += other.map_sums[cutoff]


def popularity_group_ids(
    historical_popularity: torch.Tensor,
    *,
    head_fraction: float = 0.2,
    mid_fraction: float = 0.3,
) -> torch.Tensor:
    """Assign past-active users to head/mid/tail and zeros to emerging."""

    if head_fraction <= 0 or mid_fraction <= 0 or head_fraction + mid_fraction >= 1:
        raise ValueError("head and mid fractions must be positive and sum to less than one")
    counts = historical_popularity.flatten()
    if bool(torch.any(counts < 0)):
        raise ValueError("historical popularity cannot be negative")
    groups = torch.full_like(counts, 3, dtype=torch.long)
    active = torch.nonzero(counts > 0, as_tuple=False).flatten()
    if active.numel() == 0:
        return groups
    order = torch.argsort(counts[active], descending=True, stable=True)
    ranked = active[order]
    head_count = max(1, int(round(head_fraction * active.numel())))
    mid_count = max(1, int(round(mid_fraction * active.numel())))
    mid_end = min(ranked.numel(), head_count + mid_count)
    groups[ranked] = 2
    groups[ranked[:head_count]] = 0
    groups[ranked[head_count:mid_end]] = 1
    return groups


def recency_group_ids(
    historical_popularity: torch.Tensor,
    recent_popularity: torch.Tensor,
) -> torch.Tensor:
    """Separate recent, dormant historical, and unseen/emerging users."""

    historical = historical_popularity.flatten()
    recent = recent_popularity.flatten()
    if historical.shape != recent.shape:
        raise ValueError("historical and recent popularity shapes differ")
    if bool(torch.any(historical < 0)) or bool(torch.any(recent < 0)):
        raise ValueError("popularity cannot be negative")
    groups = torch.full_like(historical, 2, dtype=torch.long)
    groups[historical > 0] = 1
    groups[recent > 0] = 0
    return groups


def aggregate_environment_metrics(
    metrics: Iterable[dict[str, float]],
) -> dict[str, float]:
    """Report mean and worst temporal-environment performance."""

    metric_list = list(metrics)
    if not metric_list:
        raise ValueError("at least one environment metric dictionary is required")
    keys = metric_list[0].keys()
    result: dict[str, float] = {}
    for key in keys:
        values = [environment[key] for environment in metric_list]
        result[key] = sum(values) / len(values)
        result[f"worst_{key}"] = min(values)
    return result

"""Distribution and active-set drift metrics for temporal environments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


def _as_counts(values: np.ndarray | Sequence[float]) -> np.ndarray:
    counts = np.asarray(values, dtype=np.float64)
    if counts.ndim != 1:
        raise ValueError("population counts must be one-dimensional")
    if np.any(counts < 0):
        raise ValueError("population counts cannot be negative")
    return counts


def _distribution(values: np.ndarray | Sequence[float]) -> np.ndarray:
    counts = _as_counts(values)
    total = counts.sum()
    if total == 0:
        return np.full_like(counts, 1.0 / max(len(counts), 1))
    return counts / total


def jensen_shannon_divergence(
    first: np.ndarray | Sequence[float],
    second: np.ndarray | Sequence[float],
    *,
    base: float = 2.0,
) -> float:
    """Return bounded Jensen-Shannon divergence between two count vectors."""

    p = _distribution(first)
    q = _distribution(second)
    if p.shape != q.shape:
        raise ValueError("population vectors must have the same shape")
    midpoint = 0.5 * (p + q)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        active = left > 0
        logarithm = np.log(left[active] / right[active]) / np.log(base)
        return float(np.sum(left[active] * logarithm))

    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


def _top_active_set(values: np.ndarray | Sequence[float], fraction: float) -> set[int]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    counts = _as_counts(values)
    active = np.flatnonzero(counts > 0)
    if active.size == 0:
        return set()
    k = max(1, int(np.ceil(fraction * active.size)))
    ranking = np.lexsort((active, -counts[active]))
    return set(int(node) for node in active[ranking[:k]])


def top_fraction_jaccard(
    first: np.ndarray | Sequence[float],
    second: np.ndarray | Sequence[float],
    *,
    fraction: float = 0.2,
) -> float:
    """Jaccard overlap of the most active users in two environments."""

    first_set = _top_active_set(first, fraction)
    second_set = _top_active_set(second, fraction)
    union = first_set | second_set
    return 1.0 if not union else len(first_set & second_set) / len(union)


def active_user_churn(
    first: np.ndarray | Sequence[float],
    second: np.ndarray | Sequence[float],
) -> float:
    """One minus the Jaccard overlap of active-user sets."""

    first_counts = _as_counts(first)
    second_counts = _as_counts(second)
    if first_counts.shape != second_counts.shape:
        raise ValueError("population vectors must have the same shape")
    first_set = set(np.flatnonzero(first_counts > 0).tolist())
    second_set = set(np.flatnonzero(second_counts > 0).tolist())
    union = first_set | second_set
    return 0.0 if not union else 1.0 - len(first_set & second_set) / len(union)


@dataclass(frozen=True)
class DriftReport:
    """Consecutive-window drift observations and their means."""

    js_divergence: tuple[float, ...]
    top_hub_jaccard: tuple[float, ...]
    active_user_churn: tuple[float, ...]

    @property
    def mean_js_divergence(self) -> float:
        return float(np.mean(self.js_divergence)) if self.js_divergence else 0.0

    @property
    def mean_top_hub_jaccard(self) -> float:
        return float(np.mean(self.top_hub_jaccard)) if self.top_hub_jaccard else 1.0

    @property
    def mean_active_user_churn(self) -> float:
        return float(np.mean(self.active_user_churn)) if self.active_user_churn else 0.0

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            mean_js_divergence=self.mean_js_divergence,
            mean_top_hub_jaccard=self.mean_top_hub_jaccard,
            mean_active_user_churn=self.mean_active_user_churn,
        )
        return payload


def compute_drift_report(
    populations: Sequence[np.ndarray | Sequence[float]],
    *,
    hub_fraction: float = 0.2,
) -> DriftReport:
    """Compute drift metrics for every consecutive pair of environments."""

    if len(populations) < 2:
        raise ValueError("at least two environments are required")
    return DriftReport(
        js_divergence=tuple(
            jensen_shannon_divergence(first, second)
            for first, second in zip(populations, populations[1:])
        ),
        top_hub_jaccard=tuple(
            top_fraction_jaccard(first, second, fraction=hub_fraction)
            for first, second in zip(populations, populations[1:])
        ),
        active_user_churn=tuple(
            active_user_churn(first, second)
            for first, second in zip(populations, populations[1:])
        ),
    )

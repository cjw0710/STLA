# Post-freeze validation sensitivity report

## Status

This report is a diagnostic supplement to the frozen one-shot experiment. It
uses the 20 checkpoints locked before test materialization and changes neither
model selection nor any reported test result. All stress targets are validation
cascades; `test_materialized=false` and `selection_changes_permitted=false` are
stored in every result file and in the aggregate summary.

The stress implementation, evaluator, and reproducible aggregate are:

- `www2027/stress.py`
- `www2027/evaluate_validation_stress.py`
- `www2027/summarize_validation_stress.py`
- `www2027/artifacts/postfreeze_validation_stress/*.json`
- `www2027/artifacts/postfreeze_validation_stress_summary.json`

## Natural drift before the held-out test

The audit divides the chronological train+validation stream into six equal-count
windows and reports the mean across five adjacent transitions. No test cascade
is used to form a window.

| Dataset | Popularity JSD | Top-hub Jaccard | Active-user churn |
| --- | ---: | ---: | ---: |
| Christianity | 0.279 | 0.482 | 0.613 |
| Android | 0.283 | 0.316 | 0.477 |
| Douban | 0.310 | 0.391 | 0.658 |
| Twitter | 0.340 | 0.329 | 0.571 |

Twitter has the largest mean popularity JSD and the largest single adjacent
transition (0.424). Douban has the largest active-user churn. These descriptors
establish natural non-stationarity but do not, by themselves, identify the
cause of the frozen Twitter test failure.

## Stress protocol

For each dataset and each frozen seed in `{21, 42, 84, 126, 168}`, the evaluator
applies three perturbations at severity 0.5 and 1.0:

1. **Recent-hub amplification:** multiply the recent popularity of historical
   head users by `1 + 3 * severity`.
2. **Recent-hub turnover:** transfer a severity fraction of recent mass from
   head/recent-active donors to historical mid/tail recipients.
3. **Emerging influx:** transfer head recent mass to historically unseen users.

Only past-only recent popularity and the recency/safety groups recomputed from
it are perturbed. The rolling graph, 28-dimensional environment context,
cumulative history, validation targets, anchor parameters, and residual
parameters stay fixed. Consequently, this is an input-sensitivity analysis,
not a learned response to drift and not a simulation of a complete future
world.

Each table entry is mean +/- sample standard deviation of MAP@100 differences
over the five frozen seeds. `Delta input` compares the stressed fused ranking
with the same checkpoint under unperturbed validation input; `Delta anchor`
compares the stressed fused ranking with the frozen anchor. Parentheses give
the number of positive seeds.

## Severity 0.5

| Dataset | Stress | Delta input | Delta anchor |
| --- | --- | ---: | ---: |
| Android | amplification | -0.000284 +/- 0.000182 (0/5) | +0.001182 +/- 0.000731 (5/5) |
| Android | turnover | -0.000306 +/- 0.000309 (2/5) | +0.001160 +/- 0.000626 (5/5) |
| Android | influx | -0.000231 +/- 0.000338 (2/5) | +0.001235 +/- 0.000575 (5/5) |
| Christianity | amplification | -0.000190 +/- 0.000951 (3/5) | +0.000420 +/- 0.001093 (3/5) |
| Christianity | turnover | +0.000113 +/- 0.000584 (3/5) | +0.000722 +/- 0.001431 (4/5) |
| Christianity | influx | +0.000031 +/- 0.000659 (3/5) | +0.000641 +/- 0.001399 (3/5) |
| Douban | amplification | -0.000185 +/- 0.000340 (2/5) | +0.001468 +/- 0.000734 (5/5) |
| Douban | turnover | -0.000204 +/- 0.000078 (0/5) | +0.001448 +/- 0.000932 (5/5) |
| Douban | influx | -0.000167 +/- 0.000111 (0/5) | +0.001486 +/- 0.000935 (5/5) |
| Twitter | amplification | +0.000471 +/- 0.000791 (4/5) | +0.010556 +/- 0.002403 (5/5) |
| Twitter | turnover | -0.013014 +/- 0.001967 (0/5) | -0.002929 +/- 0.001014 (0/5) |
| Twitter | influx | -0.014861 +/- 0.002478 (0/5) | -0.004776 +/- 0.000680 (0/5) |

## Severity 1.0

| Dataset | Stress | Delta input | Delta anchor |
| --- | --- | ---: | ---: |
| Android | amplification | -0.000324 +/- 0.000317 (1/5) | +0.001142 +/- 0.000529 (5/5) |
| Android | turnover | -0.000829 +/- 0.000506 (0/5) | +0.000637 +/- 0.000332 (5/5) |
| Android | influx | -0.000823 +/- 0.000501 (0/5) | +0.000643 +/- 0.000336 (5/5) |
| Christianity | amplification | -0.000291 +/- 0.001079 (2/5) | +0.000318 +/- 0.001070 (3/5) |
| Christianity | turnover | -0.001091 +/- 0.001999 (2/5) | -0.000482 +/- 0.000827 (2/5) |
| Christianity | influx | -0.001066 +/- 0.001962 (2/5) | -0.000457 +/- 0.000842 (2/5) |
| Douban | amplification | -0.000366 +/- 0.000448 (1/5) | +0.001287 +/- 0.000584 (5/5) |
| Douban | turnover | -0.001670 +/- 0.000938 (0/5) | -0.000017 +/- 0.000243 (3/5) |
| Douban | influx | -0.001659 +/- 0.000938 (0/5) | -0.000006 +/- 0.000233 (3/5) |
| Twitter | amplification | -0.000165 +/- 0.001252 (3/5) | +0.009921 +/- 0.001947 (5/5) |
| Twitter | turnover | -0.011013 +/- 0.001936 (0/5) | -0.000927 +/- 0.001141 (1/5) |
| Twitter | influx | -0.011032 +/- 0.001927 (0/5) | -0.000946 +/- 0.001125 (1/5) |

## Interpretation

Scale change and identity change are not equivalent. Amplifying existing hubs
has a small effect on the fused ranking and leaves it above the anchor on every
Android, Douban, and Twitter seed. Moving recent mass to different users is much
more damaging. Android retains a positive anchor margin, Douban falls to parity,
and Christianity becomes mixed at full severity. Twitter collapses under both
turnover and influx: MAP@100 falls by about 0.011 relative to its unperturbed
validation behavior and below the anchor on four of five seeds.

This supports a precise failure hypothesis: the temporal residual is more
sensitive to *who receives recent mass* than to a rescaling of established hub
activity. It is consistent with, but does not prove, identity relocation as the
mechanism behind the frozen Twitter failure. Testing that mechanism requires a
new untouched temporal benchmark, not post-test Twitter tuning.

The deterministic safety property survives every perturbation. Across 120
dataset-seed-stress-severity conditions, the evaluator audited 8,042, 17,078,
and 24,424 protected anchor hits at K=10, 50, and 100, respectively, with zero
violations at every cutoff. This confirms that protected coverage is robust to
the tested input shocks; it does not imply robustness of MAP for unprotected
head/recent-active users.


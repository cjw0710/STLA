# Temporal popularity residual: validation-only screening

Date: 2026-08-29

## What changed

The predictor now has an optional temporal popularity residual (TPR) head.  For
each target environment it constructs five node features using only preceding
cascades: standardized cumulative popularity, standardized recent popularity,
their difference, a historical-inactive indicator, and an emerging-user
indicator.  A sample-conditioned gate maps the cascade representation and
past-only environment context to five coefficients, then adds the resulting
node score to the base ranking logits.

The final gate layer is initialized to zero.  TPR therefore starts from exactly
the same predictions as dynamic ERM and learns only a residual.  Its cost is
`O(BN)` per batch, matching the full next-user logits already required by the
base model; it does not create an `N x N` graph.

All experiments in this note use `--skip-test`.  No test environments were
materialized.

## Three-seed validation results

The paired runs use seeds 21, 42, and 84, 10 epochs, 50 joint-environment steps
per epoch, full validation evaluation, and validation-selected checkpoints.

| Dataset | ERM MAP@100 | TPR MAP@100 | Paired delta | ERM worst MAP@100 | TPR worst MAP@100 | Paired delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Christianity | 0.08062 | 0.08184 | +0.00122 | 0.07257 | 0.07642 | +0.00385 |
| Android | 0.01612 | 0.01779 | +0.00166 | 0.01395 | 0.01586 | +0.00192 |

TPR raises mean MAP in 2/3 seeds on each dataset.  It raises worst-environment
MAP in 2/3 seeds on each dataset.  Christianity historical-inactive MAP and
Hit@100 improve in all three seeds, but Android reproduces that direction in
only two of three seeds.  Mid-popularity quality does not improve: it declines
or remains zero in these runs.

## Larger-dataset seed-21 extension

| Dataset | ERM MAP@100 | TPR MAP@100 | Delta | ERM worst MAP@100 | TPR worst MAP@100 | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Douban | 0.04171 | 0.04627 | +0.00456 | 0.04029 | 0.04449 | +0.00420 |
| Twitter | 0.00496 | 0.01710 | +0.01213 | 0.00488 | 0.01310 | +0.00822 |

These gains establish that the implementation scales beyond 12,000 nodes and
that the temporal residual can substantially improve validation ranking.  They
are still single-seed screening evidence.

The stratified result also changes the interpretation.  On Douban and Twitter,
the gain is dominated by head and recent-active targets.  Historical-inactive
Hit@100 falls from 0.00957 to 0 on Douban and from 0.01182 to 0 on Twitter.
Twitter mid Hit@100 improves, but Twitter tail Hit@100 falls from 0.00566 to 0;
Douban mid Hit@100 also falls to 0.  TPR is therefore a strong temporal
popularity adapter, not a general tail or reactivation solution.

## Decision

Keep TPR as a promising structural component and strong ablation, but do not
yet promote it to the WWW main method.  It passes the mean and worst-environment
validation checks on four datasets, while failing the subgroup-preservation
criterion on the two larger datasets.  Do not unlock any new test result at
this stage.

The next method iteration should retain the adaptive residual while explicitly
separating current-popularity adaptation from coverage preservation.  The
validation gate should require:

1. positive paired mean and worst MAP on at least three datasets;
2. multi-seed confirmation on Douban and Twitter;
3. no collapse of historical-inactive, mid, or tail Hit@100 relative to ERM;
4. test evaluation only after the architecture and rule are frozen.

Artifacts are stored under `artifacts/temporal_prior_validation_3seed/`,
`artifacts/temporal_prior_validation_android_3seed/`, and
`artifacts/temporal_prior_validation_large_s21/`.

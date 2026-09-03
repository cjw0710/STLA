# DriftDiff convergence and stratified follow-up

Date: 2026-08-29

## Why this follow-up was required

The initial three-seed screening used five epochs. Both static and dynamic
learning curves were still improving, so the apparent dynamic-mask gain could
have been an optimization-stage effect. A seed-21 Christianity pair was rerun
with 15 epochs, 100 four-environment updates per epoch, full validation, and
validation-only early stopping.

## Converged comparison

| Method | Selected epoch | Test MAP@100 | Worst MAP@100 |
|---|---:|---:|---:|
| static ERM | 14 | 0.0954 | 0.0928 |
| dynamic ERM | 10 | 0.0953 | 0.0871 |
| static balanced | 15 | 0.0822 | 0.0688 |
| dynamic balanced | 8 | 0.0877 | 0.0826 |

The dynamic advantage observed at epoch five did not survive convergence.
Dynamic ERM tied mean MAP but reduced worst-period MAP by 0.0057. The balanced
loss used historical-popularity exponent 0.25 and a 1.5 multiplier for dormant
users. It was tested as a predeclared exploratory correction and reduced total
performance substantially.

## Past-only test strata

The groups below are defined using information available before each test
environment. They do not use future target frequencies.

Dynamic ERM minus static ERM:

| Group | Count | ΔMAP@100 | ΔHits@100 |
|---|---:|---:|---:|
| head | 678 | +0.00125 | -0.01032 |
| mid | 137 | -0.00053 | -0.02190 |
| tail | 128 | +0.00001 | 0.00000 |
| recent active | 537 | +0.00255 | -0.00559 |
| historical inactive | 406 | -0.00147 | -0.01724 |
| emerging | 1 | 0.00000 | 0.00000 |

This is not the desired debiasing behavior. The dynamic model improves the
reciprocal rank of recent/head users but reduces top-100 coverage for mid and
dormant users.

Dynamic balanced minus dynamic ERM:

| Group | ΔMAP@100 | ΔHits@100 |
|---|---:|---:|
| head | -0.01253 | -0.05752 |
| mid | +0.00142 | +0.02190 |
| tail | -0.00003 | 0.00000 |
| recent active | -0.01567 | -0.05773 |
| historical inactive | +0.00026 | -0.01232 |

Simple inverse-popularity weighting trades away too much head/recent accuracy
and still does not improve dormant-user hit coverage. It should remain a
negative ablation.

## Decision

The current DriftDiff formulation is a no-go for a WWW main-method claim. The
five-epoch positive signal was real at that optimization stage but not robust
to convergence. GroupDRO, V-REx, and direct popularity reweighting also fail.

The reusable contributions remain valuable:

- strict chronological and past-only data protocol;
- sparse environment-conditioned low-rank implementation;
- validation-only selection and delayed test materialization;
- drift, worst-period, popularity, recency, and emerging-user metrics;
- auditable negative ablations.

The next method should explicitly separate adaptation from recent-popularity
amplification. Candidate directions must be selected on validation strata only.
A promising pivot is a constrained objective that improves dormant/mid coverage
subject to a bound on average validation MAP loss, rather than unconstrained
sample reweighting. No further test-guided tuning should be performed on the
current Christianity split.

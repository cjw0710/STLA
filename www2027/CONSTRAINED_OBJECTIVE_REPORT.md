# Constrained-objective gate: negative result

Date: 2026-08-29

This experiment tested whether a group-risk constraint could recover
mid-popularity and dormant-user coverage without sacrificing mean ranking
quality.  It is a development result, not a paper-ready benchmark.

## Protocol

- Dataset: Christianity, seed 21.
- Strict chronological 70/10/20 split with timestamp ties preserved.
- Dynamic mask, ERM, 10 epochs, 50 joint-environment steps per epoch.
- Constraint margin: 0.5; weights screened on validation only: 0.05, 0.10,
  and 0.20.
- Eligibility was fixed before test: validation MAP@100 loss at most 1%
  relative to same-budget dynamic ERM, with improvements in both mid and
  historical-inactive Hit@100.
- `VALIDATION_SELECTION.md` records the pre-test selection decision.

Only weight 0.05 passed the validation gate.  The paired ERM checkpoint and
the locked 0.05 checkpoint were then evaluated once on the complete test split.

## Locked test result

| Metric | Dynamic ERM | Constraint 0.05 | Delta |
| --- | ---: | ---: | ---: |
| MAP@100 | 0.082290 | 0.082025 | -0.000265 |
| Hit@100 | 0.520040 | 0.516407 | -0.003633 |
| Worst MAP@100 | 0.077232 | 0.076221 | -0.001011 |
| Worst Hit@100 | 0.493188 | 0.482289 | -0.010899 |
| Head MAP@100 | 0.114919 | 0.114643 | -0.000276 |
| Head Hit@100 | 0.716814 | 0.710914 | -0.005900 |
| Mid MAP@100 | 0.000790 | 0.000854 | +0.000064 |
| Mid Hit@100 | 0.043796 | 0.043796 | 0.000000 |
| Dormant MAP@100 | 0.007870 | 0.007783 | -0.000087 |
| Dormant Hit@100 | 0.194581 | 0.187192 | -0.007389 |

Tail and emerging Hit@100 remain zero.  The only favorable test movement is a
very small mid MAP increase; the intended dormant coverage improvement does not
replicate, and both mean and worst-environment quality decline.

## Decision

Reject the group-risk constraint as the main WWW contribution.  Keep it as a
negative ablation showing that loss reweighting alone does not solve temporal
coverage failure.  Do not tune more constraint weights against this test split.
The next prototype should change the predictor structurally so that it can use
past-only node-level temporal popularity state, then return to validation-only
screening.

The immutable result files are under `artifacts/constraint_locked_test/`.

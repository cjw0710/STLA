# Locked validation selection before test evaluation

Date: 2026-08-29

This note was written before materializing the test split for the constrained
objective.  It fixes the selection rule and prevents choosing a constraint
weight from test outcomes.

## Selection rule

Use the same seed-21, 10-epoch, 50-step-per-epoch budget for every candidate.
The unconstrained dynamic ERM checkpoint is the paired baseline.  A constrained
candidate is eligible only when its validation MAP@100 loss relative to that
baseline is no more than 1% (relative).  Among eligible candidates, prefer a
candidate that improves both mid-popularity Hit@100 and historical-inactive
Hit@100.  Test is evaluated only for the paired baseline and the one locked
candidate.

## Validation-only observations

| Method | MAP@100 | Relative change | Worst MAP@100 | Mid Hit@100 | Dormant Hit@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dynamic ERM | 0.083544 | baseline | 0.078598 | 0.027397 | 0.182266 |
| Constraint 0.05 | 0.082785 | -0.91% | 0.076871 | 0.041096 | 0.187192 |
| Constraint 0.10 | 0.082423 | -1.34% | 0.076708 | 0.041096 | 0.172414 |
| Constraint 0.20 | 0.080144 | -4.07% | 0.075574 | 0.041096 | 0.147783 |

The 0.05 candidate is the only eligible constrained setting.  Relative to ERM,
it raises mid Hit@100 by 0.013699 and dormant Hit@100 by 0.004926 while keeping
recent-active Hit@100 unchanged.  The trade-off is explicit: worst-environment
MAP@100 falls by 0.001727 and dormant MAP@100 falls by 0.000526.  Tail and
emerging Hits are zero for every candidate, so this validation split provides
no evidence of improvement there.

## Locked decision

Lock `constraint_weight=0.05`, `constraint_margin=0.5`.  Evaluate exactly the
same-budget dynamic ERM checkpoint and this checkpoint on test once.  The test
result is confirmatory evidence for this candidate, not another model-selection
round.  A failure on test rejects this direction rather than triggering a new
constraint-weight search on test.

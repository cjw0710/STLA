# Anchored dual-path temporal adaptation

Date: 2026-08-30

Status: historical validation report. The frozen test is complete; see
`FINAL_ONE_SHOT_TEST_REPORT.md`.

## Motivation

The unconstrained temporal popularity residual (TPR) improves mean and
worst-environment validation ranking, but on Douban and Twitter its gain is
dominated by head/recent-active targets and can collapse dormant or tail
Hit@100. Loss reweighting and a group-risk constraint did not solve this.

## Method

The anchored dual-path model has three stages.

1. Train and select the dynamic ERM anchor on validation.
2. Initialize a temporal residual model from that checkpoint, freeze the entire
   anchor, and train only the 1,477-parameter residual gate. A Top-100 hinge
   penalty activates when an anchor Mid/Tail/Dormant/Emerging candidate is
   displaced.
3. At ranking time, form one hierarchical protected union. Process
   `K=10,50,100` in order, moving any protected anchor candidate missing from
   the corresponding adaptive prefix into that prefix. Protected groups are
   computed only from historical and recent activity preceding the target
   environment.

The model now returns `base_logits` directly. Recovering the anchor by
subtracting a large residual caused floating-point cancellation near rank 100;
the direct path avoids that error. Anchor, adaptive, and fused metrics are
accumulated in the same forward pass to avoid GPU sparse-aggregation jitter.

For each cutoff `K` in `{10,50,100}`, let `P_K` be protected candidates in the anchor Top-K and
`A_K` the adaptive list. The fused list contains all of `P_K` and fills its
remaining slots using the highest-ranked candidates from `A_K`. Therefore any
protected target hit by the anchor at K remains a fused hit at K.

## Three-seed validation results

All values below are paired against the anchor logits from the same forward.
Seeds are 21, 42, and 84. Training uses 50 joint-environment steps per epoch,
full validation evaluation, and no test materialization. The residual learning
rate is `1e-3` except on Twitter, where a validation-only stability check selects
`1e-4`.

| Dataset | Anchor MAP@100 | Fused MAP@100 | Mean delta | Positive seeds | Mean delta worst MAP@100 | Positive seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Christianity | 0.08062 | 0.08153 | +0.00091 | 1/3 | +0.00081 | 1/3 |
| Android | 0.01612 | 0.01654 | +0.00042 | 2/3 | +0.00139 | 2/3 |
| Douban | 0.04060 | 0.04251 | +0.00190 | 3/3 | +0.00220 | 3/3 |
| Twitter | 0.00538 | 0.01410 | +0.00872 | 3/3 | +0.00729 | 3/3 |

The coverage audit contains 36/207/396 protected anchor hits at K=10/50/100.
There are zero fused-list violations at every cutoff. Across every dataset,
seed, cutoff, and observed protected stratum, fused Hit@K is never lower than
its paired anchor value. Full multi-cutoff evidence is in
`HIERARCHICAL_PROTECTED_UNION_REPORT.md`.

The safety mechanism trades away some unrestricted gain. Compared with the
free TPR, its mean MAP improvement is smaller, but the large-dataset subgroup
collapse is removed. Christianity remains seed-sensitive: only seed 42 improves
mean and worst MAP, although the three-seed average is positive. Christianity
overall Hit@100 also falls by 0.00422 on average because the guarantee applies
to protected groups rather than head/recent-active targets.

## Decision

Promote anchored dual-path TPR to the current main-method candidate. It now
passes the validation requirements that the previous variants failed:

- positive mean and worst MAP deltas on all four datasets;
- worst-MAP improvements on more than three datasets;
- 3/3 positive large-dataset seeds;
- exact non-collapse of protected Hit@100;
- sparse-graph scalability beyond 12,000 users.

The Twitter `1e-4` residual run selects epochs 5-6 for all seeds, compared with
epochs 1-2 for two of the `1e-3` runs. Mean fused MAP rises from 0.01340 to
0.01441 and mean worst MAP rises from 0.00940 to 0.01207. The lower learning
rate is therefore retained as the validation-selected stable setting.

These validation gates were subsequently completed with five seeds and a
hash-locked pre-test manifest. The one-shot test is reported separately in
`FINAL_ONE_SHOT_TEST_REPORT.md`; no post-test tuning is allowed.

Canonical artifacts:

- `artifacts/anchored_protected_union_validation_christian_v4/`
- `artifacts/anchored_protected_union_validation_android/`
- `artifacts/anchored_protected_union_validation_large_s21/`
- `artifacts/anchored_protected_union_validation_large_extra_seeds/`
- `artifacts/anchored_protected_union_validation_twitter_lr1e4/`

Earlier Christianity `v1`-`v3` directories are debugging artifacts from the
floating-point and list-capacity fixes and must not be used as results.

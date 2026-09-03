# Hierarchical protected-union validation report

Date: 2026-08-29

Status: historical three-seed validation report. The five-seed frozen test is
complete; see `FINAL_ONE_SHOT_TEST_REPORT.md`.

## Change

The final dual-path ranking now enforces subgroup preservation in one nested
ranking at `K=10`, `K=50`, and `K=100`. Starting from the adaptive Top-100, the
construction processes cutoffs from smallest to largest. At each cutoff, a
protected anchor item missing from that prefix is swapped forward from a later
position or replaces the lowest-ranked non-required item. The required sets are
nested, so a later pass cannot invalidate an earlier guarantee.

For each `K` in `{10, 50, 100}`, every non-head or non-recent target hit by the
anchor at `K` remains a hit in the fused ranking at `K`. Popularity and recency
groups use only statistics preceding the evaluated environment.

## Three-seed validation audit

Seeds are 21, 42, and 84. All comparisons use anchor, adaptive, and fused
scores from the same forward pass. Existing validation-selected checkpoints
are reused without retraining, and no test data are materialized.

| Dataset | Delta MAP@10 | Delta worst MAP@10 | Delta MAP@50 | Delta worst MAP@50 | Delta MAP@100 | Delta worst MAP@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Christianity | +0.00117 | +0.00136 | +0.00098 | +0.00081 | +0.00091 | +0.00081 |
| Android | +0.00029 | +0.00118 | +0.00024 | +0.00128 | +0.00042 | +0.00139 |
| Douban | +0.00166 | +0.00232 | +0.00179 | +0.00212 | +0.00190 | +0.00220 |
| Twitter | +0.00768 | +0.00609 | +0.00870 | +0.00724 | +0.00872 | +0.00729 |

The exact coverage audit is:

| Dataset | Protected hits at 10 | Violations | Protected hits at 50 | Violations | Protected hits at 100 | Violations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Christianity | 12 | 0 | 67 | 0 | 117 | 0 |
| Android | 10 | 0 | 76 | 0 | 135 | 0 |
| Douban | 0 | 0 | 9 | 0 | 37 | 0 |
| Twitter | 14 | 0 | 55 | 0 | 107 | 0 |
| Total | 36 | 0 | 207 | 0 | 396 | 0 |

Across every dataset, seed, cutoff, and observed protected stratum, the minimum
paired Hit@K delta is zero. The guarantee does not cover head/recent-active
targets, so unrestricted overall Hit@K may still decrease even when mean MAP
improves.

Compared with the earlier Top-100-only fusion, hierarchical protection changes
mean MAP@100 by `+0.00004`, `-0.00006`, `-0.00003`, and `-0.00032` on
Christianity, Android, Douban, and Twitter respectively. These small costs are
accepted because the method now makes one consistent, auditable claim at all
three reported cutoffs.

## Decision

Use hierarchical protected union as the final inference rule. Do not report the
Top-100-only variant as the main method. The remaining pre-test work is:

1. add the pre-registered seeds 126 and 168;
2. report five-seed mean, standard deviation, and exact paired tests;
3. freeze the configuration and test command before unlocking the one-shot
   test evaluation.

Canonical machine-readable results are in
`artifacts/hierarchical_protected_union_validation/` and
`artifacts/hierarchical_protected_union_validation_summary.json`.

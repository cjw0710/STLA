# Frozen one-shot hierarchical test report

Date: 2026-08-30

Status: complete. No post-test model or hyperparameter selection is permitted.

## Protocol

The exact 20 validation-selected checkpoints, their epochs and SHA-256 values,
the residual learning rates, and the K=10/50/100 inference rule were frozen in
`artifacts/pretest_selection_manifest.json` before test materialization. The
test evaluator verified that manifest, used three chronological test
environments, evaluated the complete split, and accumulated anchor, adaptive,
and fused rankings in the same forward pass.

The first attempted invocation stopped before reading any dataset because old
checkpoints do not embed a seed field. The compatibility check was corrected
to enforce the frozen path and SHA-256 and compare the embedded seed only when
present. The corrected evaluator hash was frozen before the successful run.

## Five-seed test results

Values are mean ± sample standard deviation. `p` is the exact one-sided paired
sign-flip test on fused-minus-anchor deltas. With five pairs, 0.03125 is the
smallest possible value.

| Dataset | Anchor MAP@100 | Fused MAP@100 | Delta | Positive seeds | p | Delta worst MAP@100 | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Christianity | 0.07623 ± 0.00617 | 0.07848 ± 0.00570 | +0.00225 | 5/5 | 0.03125 | +0.00400 | 0.03125 |
| Android | 0.01242 ± 0.00110 | 0.01565 ± 0.00121 | +0.00322 | 5/5 | 0.03125 | +0.00144 | 0.03125 |
| Douban | 0.05877 ± 0.00340 | 0.06872 ± 0.00200 | +0.00995 | 5/5 | 0.03125 | +0.00362 | 0.03125 |
| Twitter | 0.00608 ± 0.00128 | 0.00565 ± 0.00047 | -0.00043 | 2/5 | 0.71875 | -0.00204 | 1.00000 |

The multi-cutoff MAP deltas are:

| Dataset | Delta MAP@10 | Delta MAP@50 | Delta MAP@100 |
| --- | ---: | ---: | ---: |
| Christianity | +0.00248 | +0.00226 | +0.00225 |
| Android | +0.00277 | +0.00313 | +0.00322 |
| Douban | +0.00967 | +0.00975 | +0.00995 |
| Twitter | -0.00042 | -0.00039 | -0.00043 |

Christianity, Android, and Douban improve MAP and worst-environment MAP at all
three cutoffs in all five seeds. Their paired tests attain `p=0.03125` for
every reported MAP and worst-MAP cutoff. Twitter does not reproduce its strong
validation gain: the residual path is already below the anchor, and protected
fusion trades additional unrestricted MAP for subgroup safety.

## Exact safety audit

| Dataset | Protected hits at 10 | Violations | Protected hits at 50 | Violations | Protected hits at 100 | Violations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Christianity | 26 | 0 | 183 | 0 | 359 | 0 |
| Android | 8 | 0 | 96 | 0 | 211 | 0 |
| Douban | 1 | 0 | 65 | 0 | 162 | 0 |
| Twitter | 16 | 0 | 74 | 0 | 142 | 0 |
| Total | 51 | 0 | 418 | 0 | 874 | 0 |

Across every dataset, seed, cutoff, and observed protected stratum, the minimum
fused-minus-anchor Hit@K delta is zero. The construction therefore satisfies
its exact preservation claim on test even where overall accuracy fails.

On Twitter at K=100, mean head Hit falls by 0.01519 and recent-active Hit by
0.01100, while mid and tail Hit increase by 0.00364 and 0.00237. This is not a
contradiction: the formal guarantee protects non-head or non-recent anchor
hits, not head/recent-active targets or reciprocal rank within the prefix.

## Paper decision

The result supports a WWW paper claim of **subgroup-safe temporal adaptation**,
not universal temporal generalization:

- statistically significant mean and worst-period gains on three of four
  datasets and all three cutoffs;
- an exact, leakage-safe preservation guarantee with 0/874 test violations;
- a transparent Twitter failure showing that validation-period popularity
  adaptation can reverse under later drift.

No configuration may now be changed using these test results. Any attempt to
repair Twitter must be evaluated on new datasets or a newly designed nested
validation/stress-test protocol, with the current test retained only as the
already-consumed final result. In the paper, report Twitter as a negative
result and avoid cherry-picking a per-dataset anchor fallback after seeing
test.

Canonical artifacts:

- `artifacts/final_one_shot_hierarchical_test/` — 20 immutable per-run JSONs;
- `artifacts/final_one_shot_test_summary.json` — machine-readable statistics;
- `artifacts/pretest_selection_manifest.json` — frozen selection provenance.

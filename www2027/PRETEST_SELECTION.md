# Frozen pre-test selection

Freeze date: 2026-08-30

Status: configuration and checkpoint hashes frozen before test materialization.

## Final configuration

- Seeds: 21, 42, 84, 126, 168.
- Residual learning rate: `1e-3` on Christianity and Douban; `1e-4` on
  Android and Twitter.
- Checkpoint rule: maximum mean validation MAP@100 over trained epochs. There
  is no per-seed fallback to the anchor.
- Inference rule: one hierarchical protected union at K=10/50/100.
- Test protocol: chronological 70/10/20 split with timestamp ties preserved,
  three test environments, complete evaluation, and same-forward
  anchor/adaptive/fused accumulation.

The lower Android learning rate was selected after the original `1e-3`
configuration became negative over five seeds. It gives 5/5 positive MAP
deltas at every cutoff. A matching lower-rate check on Christianity did not
improve the primary five-seed mean MAP@100, so Christianity retains `1e-3`.
No further hyperparameter search is permitted after this freeze.

## Five-seed validation evidence

Values are mean ± sample standard deviation. `p` is the exact one-sided paired
sign-flip test on the mean fused-minus-anchor delta; with five pairs, the
smallest attainable value is 0.03125.

| Dataset | Anchor MAP@100 | Fused MAP@100 | Delta | Positive seeds | p | Delta worst MAP@100 | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Christianity | 0.07945 ± 0.00460 | 0.08006 ± 0.00330 | +0.00061 | 2/5 | 0.31250 | +0.00019 | 0.43750 |
| Android | 0.01614 ± 0.00187 | 0.01761 ± 0.00127 | +0.00147 | 5/5 | 0.03125 | +0.00146 | 0.06250 |
| Douban | 0.04177 ± 0.00193 | 0.04342 ± 0.00156 | +0.00165 | 5/5 | 0.03125 | +0.00206 | 0.03125 |
| Twitter | 0.00500 ± 0.00066 | 0.01508 ± 0.00249 | +0.01009 | 5/5 | 0.03125 | +0.00871 | 0.03125 |

MAP@10 and MAP@50 also improve on average for every dataset. Android, Douban,
and Twitter are significant at all three cutoffs. Christianity remains a
positive but non-significant heterogeneous result and must not be described as
statistically significant.

Across the five seeds, the exact guarantee covers 66, 331, and 647 protected
anchor hits at K=10, 50, and 100 respectively, with zero violations. The
minimum protected-stratum paired Hit@K delta is zero at every cutoff.

## Frozen artifacts

The exact checkpoint paths, selected epochs, SHA-256 values, evaluator hash,
and validation-summary hash are recorded in
`artifacts/pretest_selection_manifest.json`. The one-shot test evaluator refuses
checkpoints or cutoffs that differ from that manifest and reuses an existing
matching output instead of touching test again.

After test execution, results are descriptive confirmation only. They cannot
be used to select a different learning rate, epoch, seed, or reranking rule.

Before any test data were loaded, the first invocation exposed that legacy
checkpoints do not embed a `seed` field. The evaluator was corrected to compare
that field only when present; the frozen path and SHA-256 remain mandatory. No
result directory existed and no experiment configuration changed. The manifest
records the corrected evaluator hash.

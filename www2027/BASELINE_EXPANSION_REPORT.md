# Baseline Expansion Report

## Scope and protocol

This expansion is **post-confirmation and validation-only**. It does not reopen,
materialize, or evaluate the sealed MemeTracker test partition. All learned
development baselines use the same timestamp-tie-preserving 70/10/20 split,
past-only graph/hypergraph construction, real-user candidate space, observed-
prefix masking, validation-only checkpoint selection, and five seeds
`21, 42, 84, 126, 168`.

DyHGCN, MS-HGAT, and DisenIDP use the model code in the pinned BuzzBloom
community integration. This is not represented as three author-official
repositories. Data, masking, selection, and metric semantics are supplied by
the local temporal adapter, while learned baselines use a matched Adam
protocol. The three nonparametric rankings are deterministic and use only
information preceding each validation environment and prediction prefix.

## Expanded comparison

The paper now reports ten method rows on the four development datasets:

1. cumulative popularity;
2. recent-window popularity;
3. standardized popularity momentum;
4. DeDiff;
5. Dynamic DeDiff;
6. MS-HGAT;
7. DyHGCN;
8. DyHGCN + STLA;
9. DisenIDP;
10. DisenIDP + STLA.

### Deterministic past-only rankings

| Method | Christianity | Android | Douban | Twitter |
|---|---:|---:|---:|---:|
| Cumulative popularity | 0.06472 | 0.01685 | 0.04202 | 0.00388 |
| Recent popularity | **0.07360** | **0.01867** | **0.04596** | **0.01235** |
| Popularity momentum | 0.01818 | 0.00161 | 0.03757 | 0.00728 |

Recent-window popularity is the strongest heuristic on every dataset and
improves cumulative popularity by 0.00182--0.00888 MAP@100. Momentum by itself
is not a competitive predictor.

### MS-HGAT five-seed validation results

| Dataset | Mean MAP@100 | Worst-period MAP@100 |
|---|---:|---:|
| Christianity | 0.09275 +/- 0.00144 | 0.08704 +/- 0.00491 |
| Android | 0.02772 +/- 0.00076 | 0.02121 +/- 0.00046 |
| Douban | 0.04637 +/- 0.00088 | 0.04518 +/- 0.00110 |
| Twitter | 0.02751 +/- 0.00684 | 0.02080 +/- 0.00571 |

Sixteen missing MS-HGAT runs were added to the four existing seed-21 runs, so
the aggregate contains 20 complete model-dataset-seed runs. MS-HGAT is
competitive on Christianity and Android, but DyHGCN remains substantially
stronger on Douban and Twitter.

### Dynamic DeDiff boundary

Dynamic DeDiff changes five-seed validation MAP@100 from
0.08301 +/- 0.01058 to 0.08563 +/- 0.01162 on Christianity and from
0.02784 +/- 0.00156 to 0.02835 +/- 0.00107 on Android. The exact one-sided
paired p-values are 0.0625 and 0.125, so neither comparison is claimed as
significant. Its released dense operator is not evaluated on the larger
datasets.

## Reproduction

```powershell
D:\conda\envs\cgt_gpu128\python.exe -m www2027.evaluate_popularity_baselines
D:\conda\envs\cgt_gpu128\python.exe -m www2027.run_postfreeze_temporal_baselines --models MSHGAT --datasets christian android douban twitter --seeds 21 42 84 126 168
D:\conda\envs\cgt_gpu128\python.exe -m unittest discover -s www2027\tests -v
```

The complete suite passes 69/69 tests after adding the deterministic-baseline
causality test.

## Recorded artifacts

- `artifacts/popularity_baseline_validation_summary.json`
- `artifacts/postfreeze_mshgat_fiveseed_summary.json`
- `artifacts/postfreeze_dynamic_dediff_summary.json`
- `evaluate_popularity_baselines.py`
- `tests/test_popularity_baselines.py`

The paper table rounds these recorded values to five decimal places. No value
in this report is pooled with or used to revise the frozen MemeTracker claim.

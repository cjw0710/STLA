# Frozen inference-chain ablation and efficiency report

## Scope and protocol

This report uses all 20 checkpoints frozen before the one-shot test. The
ablation is evaluated only on validation and performs one shared model forward
per batch before constructing four rankings:

1. `anchor`: frozen stable score only;
2. `adaptive`: anchor plus the learned temporal residual;
3. `top100_union`: adaptive ranking with protected anchor hits guaranteed only
   at K=100;
4. `hierarchical_union`: the final rule with guarantees at K=10, 50, and 100.

Every artifact stores `test_materialized=false` and
`selection_changes_permitted=false`. The results are descriptive and do not
change the frozen method.

## Exact independent-protocol counts

These counts come from the independent 70/10/20 chronological loader with
equal start timestamps kept together, prefix length capped at 50, and target
environments split 4/2/3 for train/validation/test.

| Dataset | Users | Cascades T/V/Te | Prefix examples T/V/Te | Past directed edges, first-last |
| --- | ---: | ---: | ---: | ---: |
| Christianity | 1,652 | 412/59/118 | 9,246/560/944 | 1,652-44,620 |
| Android | 2,928 | 475/67/136 | 13,350/1,201/2,152 | 2,928-102,648 |
| Douban | 12,233 | 2,432/348/695 | 36,149/6,249/11,295 | 12,233-240,633 |
| Twitter | 12,628 | 2,404/344/687 | 46,487/5,273/11,361 | 12,628-316,800 |

The edge range is the smallest to largest rolling sparse graph supplied before
a target environment. The first graph contains one self-loop per vocabulary
node; later counts include cumulative directed interaction edges. Full
per-environment counts are in `artifacts/protocol_counts.json`.

## Five-seed validation ablation

Mean +/- sample standard deviation MAP@100:

| Dataset | Anchor | Adaptive | Top-100 union | Hierarchical union |
| --- | ---: | ---: | ---: | ---: |
| Christianity | 0.079452 +/- 0.004602 | 0.080055 +/- 0.003292 | 0.080046 +/- 0.003310 | 0.080061 +/- 0.003301 |
| Android | 0.016144 +/- 0.001867 | 0.017715 +/- 0.001276 | 0.017626 +/- 0.001272 | 0.017610 +/- 0.001275 |
| Douban | 0.041772 +/- 0.001933 | 0.043474 +/- 0.001548 | 0.043448 +/- 0.001543 | 0.043425 +/- 0.001559 |
| Twitter | 0.004996 +/- 0.000662 | 0.015387 +/- 0.002371 | 0.015287 +/- 0.002414 | 0.015082 +/- 0.002492 |

The residual is positive against the anchor in all five seeds on Android,
Douban, and Twitter (`p=0.03125`); Christianity is mixed (`2/5`, `p=0.3125`).
The strong Twitter validation result is retained as historical selection
evidence even though it reverses on the immutable test.

### Why hierarchical fusion is necessary

Aggregated over four datasets and five seeds:

| Rule | K | Protected anchor hits | Violations | Minimum protected-stratum Hit delta |
| --- | ---: | ---: | ---: | ---: |
| Top-100 only | 10 | 66 | 38 | -0.007260 |
| Top-100 only | 50 | 331 | 112 | -0.010692 |
| Top-100 only | 100 | 647 | 0 | 0.000000 |
| Hierarchical | 10 | 66 | 0 | 0.000000 |
| Hierarchical | 50 | 331 | 0 | 0.000000 |
| Hierarchical | 100 | 647 | 0 | 0.000000 |

Top-100-only fusion is not sufficient for reporting multiple cutoffs: it loses
38 of 66 protected K=10 hits and 112 of 331 protected K=50 hits. The
hierarchical rule eliminates every violation. Relative to the unrestricted
adaptive path, its mean MAP@100 changes by +0.000006, -0.000105, -0.000049, and
-0.000305 on Christianity, Android, Douban, and Twitter, respectively. Thus the
exact multi-prefix property has a small validation MAP cost compared with the
failure of the simpler rule.

## Efficiency

The benchmark uses seed 21, validation data only, batch size 64, data loading
excluded, five warm-up batches, and up to 20 batches repeated five times. Values
are median wall-clock milliseconds on the locally reported NVIDIA GeForce RTX
5090 D. Christianity contains only ten validation batches, giving 50 rather
than 100 measurements.

| Dataset | Anchor params | Residual params (% total) | Anchor forward | Adaptive forward | Fusion | Adaptive + fusion | Peak allocated | Incremental peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Christianity | 95,757 | 1,477 (1.519%) | 6.070 | 6.504 | 3.488 | 10.248 | 21.92 MiB | 11.01 MiB |
| Android | 158,281 | 1,477 (0.925%) | 8.697 | 10.386 | 5.421 | 15.667 | 38.28 MiB | 24.59 MiB |
| Douban | 614,226 | 1,477 (0.240%) | 9.880 | 10.968 | 5.553 | 16.817 | 69.13 MiB | 47.74 MiB |
| Twitter | 633,581 | 1,477 (0.233%) | 11.879 | 12.480 | 9.210 | 22.483 | 97.66 MiB | 72.67 MiB |

The residual adds only 1,477 parameters and at most 1.7 ms to the median
forward. Hierarchical fusion remains the larger incremental cost because it
constructs an exact ordered Top-100 list, but the final batch latency is below
23 ms and absolute peak allocated CUDA memory is below 98 MiB on all four
datasets. `Peak allocated` is the maximum `torch.cuda.max_memory_allocated`
over the adaptive-plus-fusion pipeline. `Incremental peak` subtracts the
resident model, validation graphs, and current batch allocation measured at
stage start. Latency and memory use separate loops so CUDA memory-stat calls do
not perturb the timing samples. Data loading is excluded from both.

During profiling, the original fusion implementation called `tolist()` once
per GPU row, causing hundreds of device synchronizations and 29-46 ms of fusion
latency in a short diagnostic. The final implementation transfers each B x K
index block once and returns it once. A full Christianity validation rerun is
bit-for-bit identical in all path metrics and guarantees before and after the
optimization; all unit tests also pass.

## Remaining ablation gap

This experiment isolates the final inference chain without retraining. The
separate five-seed no-preservation-loss and static/no-mask retraining studies
are complete in `PRESERVATION_ABLATION_REPORT.md` and
`MASK_ABLATION_REPORT.md`. Alternative sequence backbones and corrected task
baselines remain required for a complete WWW submission.

# DriftDiff screening report

Date: 2026-08-29

## Scope

This report records engineering and screening evidence only. It is not a final
WWW result table. The experiments use three paired seeds and a bounded training
budget; no significance claim is made.

The comparison changes only the graph mask:

- `static_erm`: one globally learned sparse low-rank mask;
- `dynamic_erm`: the same low-rank factors conditioned on cumulative and
  most-recent past-environment statistics.

Both methods use the same chronological 70/10/20 split, four training
environments, model, optimizer, batches, validation-only checkpoint selection,
and one final test evaluation. Test graphs and datasets are materialized only
after checkpoint selection.

## Configuration

- seeds: 21, 42, 84;
- 5 epochs, 100 four-environment updates per epoch;
- batch size 64, maximum prefix length 50;
- hidden dimension 32, mask rank 8, context dimension 8;
- complete validation and test evaluation (no batch truncation).

## Three-seed results

Values are mean ± sample standard deviation.

| Dataset | Method | MAP@100 | Worst MAP@100 | Hits@100 | Worst Hits@100 |
|---|---|---:|---:|---:|---:|
| Christianity | static ERM | 0.0774 ± 0.0049 | 0.0720 ± 0.0033 | 0.4994 ± 0.0071 | 0.4705 ± 0.0096 |
| Christianity | dynamic ERM | 0.0792 ± 0.0056 | 0.0737 ± 0.0051 | 0.5142 ± 0.0110 | 0.4932 ± 0.0082 |
| Android | static ERM | 0.0124 ± 0.0019 | 0.0099 ± 0.0017 | 0.1337 ± 0.0073 | 0.1107 ± 0.0086 |
| Android | dynamic ERM | 0.0132 ± 0.0003 | 0.0106 ± 0.0006 | 0.1337 ± 0.0077 | 0.1090 ± 0.0093 |

Paired mean changes (`dynamic_erm - static_erm`):

| Dataset | ΔMAP@100 | ΔWorst MAP@100 | ΔHits@100 | ΔWorst Hits@100 |
|---|---:|---:|---:|---:|
| Christianity | +0.00184 | +0.00176 | +0.01489 | +0.02271 |
| Android | +0.00076 | +0.00072 | -0.00006 | -0.00166 |

Christianity MAP improved in all three paired seeds; worst MAP improved in two
of three. Android MAP and worst MAP improved in two of three seeds. Android hit
coverage did not improve, so the Android result cannot be described as a
uniform ranking gain.

## Mechanism findings

The first prototype failed for two identifiable reasons:

1. Low-rank node factors were initialized too close to zero. The learned mask
   had an across-edge standard deviation of only about 0.00017 and was
   effectively constant.
2. Unscaled tied dot-product logits produced prediction cross-entropy of 19–26,
   far above the random-classification scale of roughly `log(N)`.

After using recent plus cumulative past features, an effective low-rank
initialization, and `1/sqrt(d)` logit scaling, mask edge variation rose to about
0.053 and the model learned normally.

GroupDRO and V-REx were tested separately. On the long Christianity seed-21
run, static ERM achieved 0.0803/0.0740 mean/worst MAP@100, dynamic ERM achieved
0.0823/0.0772, dynamic GroupDRO achieved 0.0790/0.0734, and dynamic V-REx
achieved 0.0774/0.0709. GroupDRO over-weighted the earliest cold-start
environment. Neither robust objective is retained as the current main method.

## Decision

The dynamic mask has a repeatable positive MAP signal on two datasets, so the
direction is worth continuing. It does not yet meet the WWW go/no-go threshold:

- only three rather than five seeds;
- only two datasets;
- Android Hits do not improve;
- learning curves are still rising at epoch five;
- no strong external baselines, stress tests, or significance tests yet.

Next experiments should prioritize convergence and churn/head-tail analysis on
these two datasets before spending compute on Douban and Twitter. The main
candidate is `dynamic_erm`; GroupDRO and V-REx should remain negative ablations.

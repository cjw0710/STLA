# Post-freeze structural mask ablation

## Scope and protocol

This report compares the anchor stage with a dynamic mask, a static mask, and
no mask. The dynamic anchors are the 20 checkpoints frozen before the one-shot
test. The static- and no-mask anchors were retrained after test consumption as
a descriptive validation-only ablation; they cannot be used to change the
selected method, hyperparameters, or frozen test claims.

All retraining uses the locked anchor protocol: chronological 70/10/20 split
with timestamp ties preserved, four training and two validation environments,
five seeds (21, 42, 84, 126, 168), batch size 64, maximum prefix length 50,
50 joint-environment steps per epoch, a 10-epoch cap, patience 3 after a
minimum of 5 epochs, and full validation (`max_eval_batches=0`). The objective
is ERM with no temporal prior. The 40 result files explicitly record
`test_materialized=false` and `selection_changes_permitted=false`.

## MAP@100

| Dataset | Dynamic | Static | No mask | Dynamic - static | Pos. | p | Dynamic - none | Pos. | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Christianity | 0.07945 ± 0.00460 | 0.07828 ± 0.00561 | 0.07574 ± 0.00603 | +0.00118 | 3/5 | 0.15625 | +0.00372 | 5/5 | 0.03125 |
| Android | 0.01614 ± 0.00187 | 0.01558 ± 0.00138 | 0.01530 ± 0.00166 | +0.00056 | 3/5 | 0.12500 | +0.00085 | 5/5 | 0.03125 |
| Douban | 0.04177 ± 0.00193 | 0.04156 ± 0.00203 | 0.04144 ± 0.00172 | +0.00021 | 2/5 | 0.31250 | +0.00034 | 2/5 | 0.46875 |
| Twitter | 0.00500 ± 0.00066 | 0.00490 ± 0.00046 | 0.00477 ± 0.00074 | +0.00010 | 4/5 | 0.31250 | +0.00022 | 3/5 | 0.25000 |

The exact one-sided paired sign-flip test has a minimum attainable p-value of
0.03125 with five seeds. Dynamic masking beats no masking in all five seeds on
Christianity and Android, but not on Douban or Twitter. No dataset provides
significant evidence that the dynamic mask is better than the static mask.

## Worst-period MAP@100

| Dataset | Dynamic | Static | No mask | Dynamic - static | Pos./p | Dynamic - none | Pos./p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Christianity | 0.07331 | 0.07098 | 0.06932 | +0.00233 | 3/5, 0.12500 | +0.00399 | 4/5, 0.06250 |
| Android | 0.01453 | 0.01339 | 0.01288 | +0.00113 | 4/5, 0.06250 | +0.00165 | 5/5, 0.03125 |
| Douban | 0.04070 | 0.04034 | 0.04019 | +0.00036 | 2/5, 0.37500 | +0.00051 | 1/5, 0.50000 |
| Twitter | 0.00440 | 0.00444 | 0.00432 | -0.00004 | 3/5, 0.62500 | +0.00008 | 3/5, 0.34375 |

Android is the cleanest worst-period case: dynamic masking improves over no
mask in all five seeds. Christianity is directionally positive but does not
reach the minimum p-value. Douban is neutral, and Twitter slightly favors the
static mask in mean worst-period MAP.

## Cutoff and protected-stratum audit

The aggregate direction is consistent at K=10, 50, and 100. Dynamic-minus-no-
mask MAP is +0.00253/+0.00356/+0.00372 on Christianity and
+0.00055/+0.00088/+0.00085 on Android. The corresponding changes are much
smaller on Douban (+0.00018/+0.00030/+0.00034) and Twitter
(+0.00031/+0.00021/+0.00022).

The structural ablation does not establish uniform subgroup improvement. At
K=100, dynamic masking raises historical-inactive Hit by 0.01576 over no mask
on Christianity and 0.00182 on Android, but lowers it by 0.00146 on Twitter.
It also lowers Twitter mid- and tail-popularity Hit by 0.00177 and 0.00031.
These are anchor-stage descriptive effects and are distinct from the exact
hierarchical protected-union guarantee, which operates after residual scoring.

## Interpretation

The defensible claim is narrow:

1. a sparse mask is useful on Christianity and Android relative to no mask;
2. the benefit is not stable on Douban or Twitter;
3. dynamic conditioning is not significantly better than a static mask on any
   individual dataset under five paired seeds; and
4. the paper should not present dynamic masking as a universal source of the
   final gains.

For a WWW submission, the strongest evidence remains the modular combination
of a frozen anchor, lightweight temporal residual, and exact hierarchical
prefix preservation, together with the explicit Twitter failure analysis. The
dynamic mask can remain part of the architecture, but its contribution must be
described as dataset-dependent unless new untouched benchmarks provide stronger
evidence.

## Reproduction

```powershell
D:\conda\envs\cgt_gpu128\python.exe -m www2027.run_postfreeze_mask_ablation

$dynamic = Get-ChildItem www2027\artifacts\postfreeze_validation_inference_ablation\*.json
$mask = Get-ChildItem www2027\artifacts\postfreeze_mask_ablation_training\*.json
D:\conda\envs\cgt_gpu128\python.exe -m www2027.summarize_mask_ablation `
  --dynamic-json $dynamic.FullName `
  --mask-json $mask.FullName `
  --output-json www2027\artifacts\postfreeze_mask_ablation_summary.json
```

Machine-readable aggregate:
`www2027/artifacts/postfreeze_mask_ablation_summary.json`.

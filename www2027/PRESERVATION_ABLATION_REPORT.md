# Preservation-loss ablation report

## Question and protocol

Does the soft Top-100 preservation loss contribute beyond the deterministic
hierarchical inference guarantee?

To answer this without touching the consumed test, we retrained the residual
gate from each of the same 20 frozen ERM bases with `preservation_weight=0`.
Everything else is paired with the final configuration: seeds
`21/42/84/126/168`, 10-epoch cap, three-epoch patience after a five-epoch
minimum, 50 joint-environment steps per epoch, full validation, batch size 64,
and dataset-specific residual learning rates. The backbone remains frozen and
only 1,477 residual parameters are trained. Every selected checkpoint is then
evaluated with the same hierarchical union at K=10/50/100.

The run is strictly validation-only. No ablated checkpoint is evaluated on
test, and the results cannot change the frozen main method.

## Five-seed result

Mean +/- sample standard deviation MAP@100 after hierarchical fusion:

| Dataset | No preservation | Main | Main - no preservation | Positive seeds | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Christianity | 0.080377 +/- 0.004513 | 0.080061 +/- 0.003301 | -0.000316 +/- 0.001775 | 2/5 | 0.62500 |
| Android | 0.017510 +/- 0.001347 | 0.017610 +/- 0.001275 | +0.000100 +/- 0.000310 | 3/5 | 0.25000 |
| Douban | 0.042662 +/- 0.001930 | 0.043425 +/- 0.001559 | +0.000763 +/- 0.000828 | 4/5 | 0.06250 |
| Twitter | 0.011292 +/- 0.002651 | 0.015082 +/- 0.002492 | +0.003790 +/- 0.002296 | 5/5 | 0.03125 |

The corresponding mean worst-MAP@100 differences are -0.001600, +0.000429,
+0.000803, and +0.003605. The soft loss is therefore not uniformly beneficial:
Christianity is mixed and Android is nearly unchanged. It is strongly useful
on Twitter and suggestive on Douban.

## What the loss does and does not do

Both configurations have zero hierarchical guarantee violations at every
cutoff. The loss is not the source of the formal property.

The mean MAP@100 change introduced by hierarchical fusion relative to the free
adaptive path is:

| Dataset | Main | No preservation |
| --- | ---: | ---: |
| Christianity | +0.000007 | -0.000040 |
| Android | -0.000104 | -0.000109 |
| Douban | -0.000049 | -0.000076 |
| Twitter | -0.000306 | -0.000279 |

The correction costs are similar. The soft loss does not simply make the final
fusion operation cheaper. Instead, it changes residual optimization and the
validation-selected solution. Mean selected epoch changes from 4.8 to 2.8 on
Christianity, 3.2 to 4.8 on Android, 2.6 to 1.2 on Douban, and 7.2 to 9.0 on
Twitter when the loss is removed.

The defensible interpretation is narrow: the soft loss is a training
regularizer that materially improves the adaptive solution on Twitter and
possibly Douban, while the deterministic hierarchical union independently
provides exact coverage safety. It does not repair the later Twitter test
regime change; the frozen test result remains negative.

## Reproducible artifacts

- runner: `www2027/run_postfreeze_no_preservation.py`;
- training results: `artifacts/postfreeze_no_preservation_training/`;
- validation reranking: `artifacts/postfreeze_no_preservation_hierarchical/`;
- paired summary: `artifacts/postfreeze_preservation_ablation_summary.json`;
- aggregation: `www2027/summarize_preservation_ablation.py`.


# Strong-Backbone Temporal Adapter Report

## Status

The validation-only incremental-value and second-backbone gates pass. A
3,517-parameter backbone-agnostic temporal logit adapter was attached to frozen
corrected DyHGCN and DisenIDP checkpoints on all four datasets and replicated
with seeds 21, 42, 84, 126, and 168. Across the eight backbone-dataset cells,
all mean MAP@100 deltas are non-negative and seven cells improve in all five
seeds. No test dataset, tensors, forward pass, or selection access was used.

This is still post-freeze validation evidence and cannot revise the consumed
one-shot result. Both DyHGCN and DisenIDP are complete over five seeds on all
four datasets (40 adapter runs in total).

## Adapter

The adapter consumes only:

- frozen real-user anchor logits;
- the current observed prefix and timestamps;
- cumulative and recent node popularity from preceding temporal environments;
- a 28-dimensional past-only environment summary.

It does not use or require a backbone hidden state. An environment encoder and
an eight-feature causal prefix descriptor produce five coefficients. Those
coefficients score five node features: standardized historical popularity,
standardized recent popularity, their difference, dormancy, and emergence.
The resulting residual is added to the frozen logits before applying the same
hierarchical protected union at K=10/50/100.

The final gate layer is initialized to zero, so epoch 0 is exactly the frozen
anchor. Validation selection is allowed to fall back to epoch 0, preventing a
negative adapter from being forced into the result. The implementation is in
`models/temporal_logit_adapter.py` and `train_strong_logit_adapter.py`.

## Seed-21 DyHGCN validation results

| Dataset | Selected epoch | Anchor MAP@100 | Fused MAP@100 | Delta | Anchor worst | Fused worst | Delta worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Christianity | 7 | 0.08582 | 0.08986 | +0.00404 | 0.08075 | 0.08759 | +0.00684 |
| Android | 0 | 0.02859 | 0.02859 | 0.00000 | 0.01971 | 0.01971 | 0.00000 |
| Douban | 1 | 0.05649 | 0.05763 | +0.00114 | 0.05445 | 0.05680 | +0.00235 |
| Twitter | 5 | 0.19237 | 0.19850 | +0.00613 | 0.18708 | 0.19322 | +0.00614 |

The unrestricted adaptive and protected-fused Christianity MAP@100 values are
0.08974 and 0.08986, respectively; protection does not erase the adapter gain.
The corresponding values on the other datasets are retained in the result
JSON files.

## Five-seed DyHGCN result

| Dataset | Anchor MAP@100 | Adapted MAP@100 | Delta | Positive seeds | p | Delta worst | p worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Christianity | 0.08818 ± 0.00260 | 0.09120 ± 0.00169 | +0.00302 | 5/5 | 0.03125 | +0.00352 | 0.06250 |
| Android | 0.02620 ± 0.00185 | 0.02683 ± 0.00132 | +0.00063 | 2/5 | 0.25000 | +0.00032 | 0.25000 |
| Douban | 0.05836 ± 0.00125 | 0.05992 ± 0.00141 | +0.00156 | 5/5 | 0.03125 | +0.00182 | 0.03125 |
| Twitter | 0.18354 ± 0.00502 | 0.18817 ± 0.00621 | +0.00463 | 5/5 | 0.03125 | +0.00686 | 0.03125 |

Android has two positive seeds and three exact epoch-zero fallbacks, with no
negative final seed. Christianity, Douban, and Twitter improve mean MAP in all
five seeds. Worst MAP is significant on Douban and Twitter; Christianity has
four positive and one negative worst-period seed.

Across these 20 runs, the guarantee preserves 2,664/6,115/8,279 protected hits
at K=10/50/100 with zero violations.

## Seed-21 DisenIDP validation results

| Dataset | Selected epoch | Anchor MAP@100 | Fused MAP@100 | Delta | Delta worst |
| --- | ---: | ---: | ---: | ---: | ---: |
| Christianity | 5 | 0.09647 | 0.10049 | +0.00402 | +0.00540 |
| Android | 8 | 0.02829 | 0.02929 | +0.00099 | +0.00129 |
| Douban | 1 | 0.04601 | 0.05608 | +0.01008 | +0.01069 |
| Twitter | 2 | 0.12541 | 0.13027 | +0.00486 | +0.00306 |

This is an architecturally distinct replication: all four datasets improve in
mean and worst MAP@100. It confirms that the adapter is not tied to DyHGCN's
hidden representation or graph operator.

## Five-seed DisenIDP result

| Dataset | Anchor MAP@100 | Adapted MAP@100 | Delta | Positive seeds | p | Delta worst | Positive worst | p worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Christianity | 0.09485 ± 0.00139 | 0.09813 ± 0.00160 | +0.00328 | 5/5 | 0.03125 | +0.00413 | 5/5 | 0.03125 |
| Android | 0.02772 ± 0.00046 | 0.02838 ± 0.00070 | +0.00065 | 5/5 | 0.03125 | +0.00005 | 3/5 | 0.46875 |
| Douban | 0.04380 ± 0.00143 | 0.05460 ± 0.00097 | +0.01080 | 5/5 | 0.03125 | +0.01063 | 5/5 | 0.03125 |
| Twitter | 0.12916 ± 0.00306 | 0.13442 ± 0.00372 | +0.00526 | 5/5 | 0.03125 | +0.00397 | 5/5 | 0.03125 |

All five paired seeds improve mean MAP@100 on every dataset. Worst-period
MAP@100 also improves in all five seeds on Christianity, Douban, and Twitter.
Android is the boundary case: its worst-period mean delta is only +0.00005,
with three positive and two negative seeds (p=0.46875), despite its 5/5 mean
MAP improvement. Across these 20 DisenIDP runs, the guarantee preserves
3,486/7,454/9,652 protected hits at K=10/50/100 with zero violations.

## Exact protected-hit audit

| Cutoff | DyHGCN protected hits | DisenIDP protected hits | Total | Violations |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 2,664 | 3,486 | 6,150 | 0 |
| 50 | 6,115 | 7,454 | 13,569 | 0 |
| 100 | 8,279 | 9,652 | 17,931 | 0 |

The guarantee remains conditional: it preserves protected targets already hit
by either anchor. It does not guarantee retrieval of a target the anchor misses or
preserve all unprotected head/recent-active hits.

## Decision

The backbone migration gate and its architecturally distinct replication are
complete. The validation evidence supports the adapter's average-ranking claim
on two strong architectures. It also bounds the robustness claim: worst-period
improvement is consistent in six of eight architecture-dataset cells, neutral
to mixed on Android, and must not be described as universal.

MS-HGAT is optional rather than a submission blocker. The next protocol step is
to define a new untouched test benchmark before any confirmatory evaluation; do
not reopen the already consumed DeDiff test split.

That step is now complete on the independently sealed MemeTracker benchmark.
On its one-shot test, DyHGCN improves from 0.09868 ± 0.00236 to
0.10631 ± 0.00242 MAP@100 (+0.00764), and DisenIDP improves from
0.08846 ± 0.00296 to 0.09781 ± 0.00353 (+0.00935). Both mean and
worst-period results improve in 5/5 seeds with p=0.03125, while preserving
1,790/6,055/10,554 protected hits at K=10/50/100 with zero violations. See
`MEMETRACKER_CONFIRMATION_REPORT.md` for the timestamp parser, split hashes,
selection freeze, and one-shot audit.

Artifacts:

- `artifacts/postfreeze_strong_adapter/dyhgcn_*.json`
- `artifacts/postfreeze_strong_adapter/disenidp_*.json`
- `artifacts/postfreeze_strong_adapter_disenidp_summary.json`
- `artifacts/postfreeze_strong_adapter_dyhgcn_summary.json`
- `checkpoints/postfreeze_strong_adapter/dyhgcn_*_seed21.pt`
- `models/temporal_logit_adapter.py`
- `train_strong_logit_adapter.py`
- `run_postfreeze_strong_adapter.py`

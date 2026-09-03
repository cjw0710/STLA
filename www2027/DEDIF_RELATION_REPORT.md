# Direct Relationship to CIKM DeDiff

## Scope

This post-freeze, validation-only experiment makes the relationship to the
CIKM paper executable rather than rhetorical. The original `model.py`,
`module.py`, `dataLoader.py`, and `graph.py` are not edited. The corrected
runner imports the original `DeDiff` class and `loss_function`, replaces the
test-selected legacy runner with a tie-preserving 70/10/20 chronological
protocol, constructs every graph from preceding cascades only, and selects the
checkpoint using mean validation MAP@100. The test partition remains a tuple
of immutable records and is never converted to a Dataset, DataLoader, tensor,
or model input.

The direct extension freezes that corrected DeDiff checkpoint and attaches the
same zero-initialized temporal logit adapter and hierarchical protected-union
rule used for the strong-backbone experiment. DeDiff's original user ids are
supported with `input_id_offset=0`; the BuzzBloom baselines retain their
existing offset of two. Epoch zero therefore reproduces DeDiff's logits and
ranking exactly and provides a validation-selected no-regression fallback.

## Seed-21 validation result

All values below use real-user targets only. Unlike the legacy evaluator, EOS
is not counted as a next-user prediction.

| Dataset | Corrected DeDiff MAP@100 | Worst MAP@100 | DyHGCN MAP@100 | DyHGCN worst | Selected adapter epoch | Final delta | Protected hits K=10/50/100 | Violations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Christianity | 0.093242 | 0.090048 | 0.085821 | 0.080749 | 0 | 0.000000 | 7 / 33 / 48 | 0 / 0 / 0 |
| Android | 0.027461 | 0.020805 | 0.028591 | 0.019705 | 0 | 0.000000 | 10 / 40 / 62 | 0 / 0 / 0 |

Corrected DeDiff is a strong anchor on Christianity and is slightly below
DyHGCN in Android mean MAP, although its worst-period MAP is slightly higher.
The 3,517-parameter fixed-feature adapter does not improve either DeDiff
checkpoint and safely selects the exact epoch-zero fallback. A more expressive
rank-eight candidate residual (17,253 parameters on Christianity) was also
screened; it lowers training cross entropy but not validation MAP, so it too
falls back to the unchanged anchor. This negative result is retained rather
than hidden.

## What can and cannot be claimed

The current WWW method has two distinct relationships to DeDiff:

1. **Direct lineage:** it can wrap a corrected, frozen DeDiff model without
   changing its source or sacrificing any protected anchor hit.
2. **Problem extension:** it replaces DeDiff's static popularity setting with
   past-only temporal environments, validation-only selection, worst-period
   reporting, and an exact multi-cutoff preservation rule.

The output-only adapter does **not** improve DeDiff itself. A subsequent direct
internal variant does: the environment-conditioned rank-eight correction of
DeDiff's debiasing projection improves seed-21 validation MAP@100 by 0.001241
on Christianity and 0.000223 on Android. Christianity worst-period MAP falls
by 0.000338, so the evidence does not support a universal robustness claim.
See `DYNAMIC_DEDIFF_REPORT.md`. The paper can now describe a direct dynamic
DeDiff extension, while keeping the backbone-agnostic DyHGCN result as separate
generality evidence.

## Original implementation scalability boundary

The unmodified model stores a dense `user_num x user_num` `Debasing` parameter
and multiplies dense graph matrices by it. Its memory is quadratic and the
dense graph product is cubic in the number of users. The single `Debasing`
matrix alone occupies approximately 10.4 MiB on Christianity, 32.7 MiB on
Android, 570.9 MiB on Douban, and 608.4 MiB on Twitter in float32; forward and
backward require several additional dense matrices. For this reason the direct
corrected run is presently limited to Christianity and Android. This is an
architectural limitation of the preserved CIKM implementation, not a missing-
data or protocol failure.

## Decision

Keep corrected DeDiff as the direct-lineage baseline and retain the failed
output-adapter result as an ablation. The environment-conditioned internal
low-rank correction has now passed the seed-21 validation development gate on
both tractable datasets. It should advance to multi-seed validation, while the
safe adapter on strong external backbones remains the generality track.

Artifacts:

- `artifacts/postfreeze_temporal_dediff/dediff_christian_seed21.json`
- `artifacts/postfreeze_temporal_dediff/dediff_android_seed21.json`
- `artifacts/postfreeze_dediff_adapter/dediff_christian_seed21.json`
- `artifacts/postfreeze_dediff_adapter/dediff_android_seed21.json`
- `artifacts/postfreeze_dediff_adapter/dediff_christian_rank8_seed21.json`

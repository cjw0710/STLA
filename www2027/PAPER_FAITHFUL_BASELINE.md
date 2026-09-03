# DeDiff paper-faithful baseline boundary

## Status

The repository now distinguishes three materially different DeDiff variants:

1. **Released legacy DeDiff** (`model.py`, `module.py`, `main.py`) uses a dense
   `N x N` debiasing operator, evaluates the test split every epoch, and counts
   EOS in the legacy metric path.
2. **Protocol-corrected legacy DeDiff** (`train_temporal_dediff.py`) preserves
   the released model while replacing test-selected training with past-only
   chronological environments, validation-only selection, and real-user-only
   evaluation. Its results measure the released architecture under a safe
   protocol; they are not a reproduction of the PDF's low-rank edge mask.
3. **Paper-formula DeDiff candidate** (`models/paper_graph_disentangler.py` and
   `models/paper_faithful_dediff.py`) implements the sparse low-rank graph
   equations, score-level social-temporal attention, causal dot-product head,
   bias-distribution objective, and composite loss. Ambiguous choices are
   frozen in `config/paper_faithful_v1.json`; no result is labeled a reproduced
   paper result until validation selection and exact-data issues are resolved.

The WWW dynamic DeDiff and strong-backbone adapter remain separate extension
tracks. No existing artifact or checkpoint was overwritten by this milestone.

## Implemented equation mapping

| PDF component | Implementation status |
| --- | --- |
| Eq. (2), shared `sigmoid(P Q^T)` mask | Implemented exactly on observed interaction/social edges by `PaperLowRankEdgeMask`; only `2NK` factor parameters are stored. |
| Eq. (2), complementary causal/bias views | Implemented; causal and bias edge weights sum exactly to the input edge weight. |
| Eqs. (3)-(4), four GCN views | Implemented with sparse normalized propagation, self-loops, sigmoid activation, independent view encoders, and sum fusion of `H^(0)...H^(L)`. |
| Eqs. (5)-(7), interaction/social proxy BPR | Implemented as a graph-stage loss. The social proxy follows the printed degree-weighted definition rather than silently degree-normalizing it. |
| Eqs. (8)-(10), causal/bias disagreement | Implemented with concatenated graph-view embeddings. The printed unhinged objective is the default; a named hinged correction is available only for ablation. |
| Eqs. (11)-(14), STAN | Implemented with causal future masking and score-level social-distance/time-interval bias. The manifest follows the printed per-head `d x d` projections and uses `sqrt(head_dimension)` scaling because printed `d/B` is non-integral. |
| Eq. (15), causal prediction | Implemented as the printed dot product between the final causal context and causal candidate embeddings, with seen users excluded. |
| Eq. (16), bias objective | Implemented with an explicit PDF-direction `prediction_to_target` KL; the released-code reverse direction is available only as a named ablation. |
| Eq. (17), composite loss | Implemented with explicit `alpha`, `lambda_disagreement`, and `lambda_inter_view` rather than implicit unit weights hidden in source. |

## Verification

Thirteen paper-faithful tests establish that:

- sparse edge scores equal values gathered from an explicit dense
  `sigmoid(P Q^T)` reference;
- causal and bias weights are complementary;
- interaction and social graphs reuse the same learned mask;
- all four graph views and both disentanglement losses are differentiable;
- empty observed graphs remain valid through the stated self-loop path;
- future cascade tokens cannot alter earlier STAN states;
- padded queries produce zero states and padding/EOS cannot become a target;
- both causal and bias STAN branches and the graph mask receive gradients;
- legacy one-based ids and prefix social distances are remapped exactly.

The complete `www2027/tests` suite passes: **61/61 tests**.

Run the focused tests:

```powershell
D:\conda\envs\cgt_gpu128\python.exe -m unittest www2027.tests.test_paper_graph_disentangler -v
```

Run the complete regression suite:

```powershell
D:\conda\envs\cgt_gpu128\python.exe -m unittest discover -s www2027\tests -v
```

## Frozen executable resolutions

1. **Rank `K` is absent from the PDF implementation details.** The manifest
   freezes validation candidates `{8, 16, 32}` and engineering default 16.
2. **Attention dimensions are inconsistent.** The PDF specifies `d=64`,
   `B=10`, and per-head `d'=d/B`, which is non-integral. The released code uses
   a different attention parameterization and defaults to six heads. The
   formula candidate follows the printed full `d x d` head projections, so its
   head dimension is 64 and its scale is `sqrt(64)`.
3. **The PDF describes score-level joint social-temporal injection, whereas the
   source runs separate temporal and social encoders and fuses their outputs.**
   The formula candidate designates the score-level PDF definition as
   authoritative; the released fused-output path remains a separate baseline.
4. **The next-user head differs.** The formula candidate uses the Eq. (15) dot
   product. The released learned linear classifier is not reused.
5. **Loss coefficients are not reproducible from source.** The PDF reports
   validation tuning for `alpha`, `lambda_1`, and `lambda_2`; the released loss
   adds all three auxiliary objectives with implicit unit weights. The runner
   exposes all weights and restricts their future selection to validation.
6. **The popularity target is underspecified.** The training-only source,
   smoothing, normalization, and whether it is cascade- or graph-frequency
   based need to be fixed in the protocol. The manifest uses smoothed real-user
   activation frequency from training cascades only.
7. **The printed disagreement objective is unbounded without a hinge or norm
   constraint.** The exact printed form remains the default; the corrected
   hinged form is available only through `--hinged-disagreement` and must be
   reported as an ablation.

These choices make the model executable and auditable; they do not prove which
interpretation generated the PDF tables. Author confirmation or the original
camera-ready experiment configuration is still preferable.

## Bundled-data audit

The attached code bundle is not a direct match for every dataset statistic in
Table 1 of the PDF. Counts below are read from the three bundled cascade JSON
files; the user column is the maximum real user id represented by the bundled
model vocabulary.

| Dataset | PDF users | Bundle max user id | PDF cascades | Bundle train/valid/test | Bundle total |
| --- | ---: | ---: | ---: | --- | ---: |
| Twitter | 12,627 | 12,627 | 3,442 | 2,748 / 343 / 344 | 3,435 |
| Douban | 12,232 | 12,232 | 3,475 | 2,780 / 347 / 348 | 3,475 |
| Android | 9,958 | 2,927 | 679 | 542 / 68 / 68 | 678 |
| Christianity | 2,897 | 1,651 | 589 | 471 / 59 / 59 | 589 |

The Android and Christianity vocabulary differences are too large to treat as
rounding or a PAD/EOS convention. Twitter is also missing seven cascades and
Android one cascade relative to the table. A full reproduction must either
recover the exact processed data used for the PDF or explicitly report that it
uses the released bundle rather than the paper dataset.

There is also a released graph-channel naming reversal. `build_two_graphs`
stacks the loaded social/friendship graph first and the cascade co-occurrence
graph second, while `get_info` names channel zero `A_interaction` and channel
one `A_social`. The protocol-corrected legacy runner preserves this behavior
for checkpoint compatibility. The paper-faithful runner must instead route the
two semantic graphs by meaning and record that incompatibility.

## Next gate

`train_paper_faithful.py` now reads only `cascade_train.json` and
`cascade_valid.json`, builds both graphs from training-period information, and
selects exclusively by validation MAP@100. It never opens
`cascade_test.json`. A one-step Christianity seed-21 engineering smoke passed
on CUDA with 555,796 parameters, 41,338 interaction edges, 97,443 social edges,
and finite loss 20.12297. Its one-batch validation MAP is zero, which is not a
scientific result and is recorded only as an end-to-end execution check.

The next scientific gate is a validation-only Christianity sweep over rank and
loss weights, followed by multiple seeds. The already consumed historical test
results must remain outside all further model selection.

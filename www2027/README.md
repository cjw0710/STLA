# WWW 2027 prototype

This directory is intentionally independent of the published CIKM DeDiff
implementation. It establishes the leakage-safe foundation for a new paper on
robust information diffusion under temporal popularity drift.

The clean anonymous manuscript is now in `www2027/paper/main.tex`, with the
compiled nine-page submission PDF at `output/pdf/STLA_WWW2027_submission.pdf` (eight main
pages and one full references page, with no appendix). The paper uses
STLA (Subgroup-Safe Temporal Logit Adaptation) as the method name and treats
the independently sealed MemeTracker one-shot test as its primary result.

Implemented in this milestone:

- strict chronological train/validation/test partitions without splitting
  equal start timestamps;
- contiguous temporal training environments;
- interaction graphs and popularity statistics built from preceding cascades
  only;
- reproducible drift metrics (Jensen-Shannon divergence, top-hub overlap, and
  active-user churn);
- an environment encoder based on past-only population/topology summaries;
- an environment-conditioned low-rank mask evaluated only on sparse edges.
- a next-user predictor with a dedicated padding id (real user zero is kept);
- selectable ERM, GroupDRO, and V-REx objectives, validation-only checkpoint
  selection, and one final test pass with mean and worst-environment metrics.

Run the tests from `D:\DeDiff`:

```powershell
D:\conda\envs\cgt_gpu128\python.exe -m unittest discover -s www2027\tests -v
```

Audit all bundled datasets with five temporal windows:

```powershell
D:\conda\envs\cgt_gpu128\python.exe -m www2027.audit_drift --windows 5 --json-out www2027\artifacts\drift_audit.json
```

A fast end-to-end smoke
run (one optimizer step, one evaluation batch per environment) is:

```powershell
D:\conda\envs\cgt_gpu128\python.exe -m www2027.train_temporal --dataset christian --epochs 1 --steps-per-epoch 1 --max-eval-batches 1 --dimension 16 --rank 4 --context-dim 8 --environment-hidden-dim 16 --max-prefix-length 30 --batch-size 8 --result-json www2027\artifacts\christian_smoke.json
```

For an experiment, remove both smoke limits. Checkpoints are selected by mean
validation MAP@100. Test data are evaluated only once, after the best validation
checkpoint has been restored. Both mean and worst-environment metrics are saved.

Run the paired screening pilot (static-mask ERM versus dynamic-mask ERM):

```powershell
D:\conda\envs\cgt_gpu128\python.exe -m www2027.run_pilot
```

The runner uses distinct checkpoints per method and seed, resumes completed
runs, and writes aggregate paired deltas to `artifacts/pilot/summary.json`.
Its bounded-step defaults are for screening only, not paper reporting.

The three-seed screening evidence and negative robust-objective ablations are
documented in `PILOT_REPORT.md`. ERM was the strongest screening setting;
GroupDRO and V-REx remain available as ablations.

`CONVERGENCE_REPORT.md` supersedes the early screening interpretation: after
longer validation-selected training, the dynamic gain disappears and stratified
evaluation reveals reduced mid/dormant coverage. The current formulation is a
no-go for a main-method claim; popularity-balanced training is retained only as
a negative ablation.

The validation-only constrained-objective gate is recorded in
`VALIDATION_SELECTION.md`. Once a candidate has been locked, use
`evaluate_checkpoint.py` to evaluate the existing checkpoint without retraining.
The command is idempotent: a matching result JSON is reused rather than touching
the test split again.

`CONSTRAINED_OBJECTIVE_REPORT.md` records the resulting negative test: the
constraint did not reproduce its validation coverage gain and is rejected as a
main-method direction.

`TEMPORAL_PRIOR_REPORT.md` documents the next structural pivot. The temporal
popularity residual improves mean and worst validation MAP on four datasets,
but large-dataset gains currently come with dormant/tail collapse, so no new
test result has been unlocked.

`DUAL_PATH_REPORT.md` documents the subgroup-safe successor. It freezes the ERM
anchor and trains only a small temporal residual gate.
`HIERARCHICAL_PROTECTED_UNION_REPORT.md` records the final inference rule: one
nested ranking exactly preserves protected anchor hits at K=10/50/100.
Three-seed validation is positive on all four datasets; audits cover
36/207/396 protected hits with zero violations. This is the current main-method
candidate; its historical validation stage remained locked pending five seeds
and a frozen pre-test rule.

`PRETEST_SELECTION.md` completes those two gates. Five-seed validation is
significant for MAP on Android, Douban, and Twitter; Christianity remains
positive but non-significant. The exact checkpoint hashes and final
configuration are frozen in `artifacts/pretest_selection_manifest.json` before
any hierarchical one-shot test evaluation.

`FINAL_ONE_SHOT_TEST_REPORT.md` records the completed immutable test pass.
Christianity, Android, and Douban improve mean and worst MAP at K=10/50/100 in
all five seeds with exact paired `p=0.03125`. Twitter is a negative test result.
The hierarchical guarantee covers 51/418/874 protected test hits at
K=10/50/100 with zero violations. No post-test tuning is allowed.

`POSTFREEZE_STRESS_REPORT.md` adds a validation-only input-sensitivity audit
using all 20 frozen checkpoints. It covers hub amplification, hub turnover, and
emerging-user influx at two severities (120 conditions total). The audit
preserves 8,042/17,078/24,424 protected anchor hits at K=10/50/100 with zero
violations and identifies turnover/influx, rather than hub rescaling, as the
main Twitter failure hypothesis. It is diagnostic evidence and is not used for
model selection.

Rebuild the stress aggregate after evaluating checkpoints:

```powershell
$resultFiles = Get-ChildItem www2027\artifacts\postfreeze_validation_stress\*.json
D:\conda\envs\cgt_gpu128\python.exe -m www2027.summarize_validation_stress --result-json $resultFiles.FullName --output-json www2027\artifacts\postfreeze_validation_stress_summary.json
```

`INFERENCE_ABLATION_AND_EFFICIENCY_REPORT.md` consolidates the exact protocol
counts, a five-seed validation-only comparison of anchor/adaptive/Top-100-only/
hierarchical rankings, and a local latency/parameter benchmark. Top-100-only
fusion loses 38 protected K=10 hits and 112 protected K=50 hits; hierarchical
fusion reduces both counts to zero. The residual gate adds 1,477 parameters.
The final pipeline stays below 23 ms median latency and 98 MiB peak allocated
CUDA memory for batch size 64 on all four datasets.

`PRESERVATION_ABLATION_REPORT.md` reports a full five-seed, four-dataset
validation-only retraining ablation. The soft preservation loss is mixed on
Christianity, nearly neutral on Android, suggestive on Douban, and improves
Twitter MAP@100 by 0.00379 in all five seeds. Both variants retain zero formal
violations after hierarchical fusion, separating optimization benefit from the
deterministic safety guarantee.

`MASK_ABLATION_REPORT.md` reports the completed 40-run structural retraining
ablation under the same locked anchor protocol. Dynamic masking beats no mask
in all five seeds on Christianity and Android (`p=0.03125`), but is neutral on
Douban and Twitter. It is not significantly better than a static mask on any
individual dataset, so the paper treats its contribution as dataset-dependent.

`TEMPORAL_BASELINE_REPORT.md` records the validation-only readiness gate
for DyHGCN, MS-HGAT, and DisenIDP. The original seed-21 gate showed that every
corrected strong baseline beat the existing seed-21 DriftDiff path in mean and worst MAP@100;
the strongest gaps are very large on Twitter. The submission path was therefore
migrated to frozen DyHGCN and DisenIDP anchors. The third-party source remains
pinned and unchanged.

`BASELINE_EXPANSION_REPORT.md` records the broader paper comparison. It adds
three deterministic past-only rankings and completes MS-HGAT over five seeds
on Christianity, Android, Douban, and Twitter (20 runs total). Together with
DeDiff, Dynamic DeDiff, DyHGCN, DisenIDP, and the two STLA variants, the paper
now reports ten method rows under clearly separated validation-only evidence.
The sealed MemeTracker result is not reopened or pooled with these additions.

Run or resume the temporal baseline matrix with:

```powershell
D:\conda\envs\cgt_gpu128\python.exe -m www2027.run_postfreeze_temporal_baselines --seeds 21
$baseline = Get-ChildItem www2027\artifacts\postfreeze_temporal_baselines\*.json
D:\conda\envs\cgt_gpu128\python.exe -m www2027.summarize_temporal_baselines --result-json $baseline.FullName --output-json www2027\artifacts\postfreeze_temporal_baseline_summary_seed21.json
```

`STRONG_BACKBONE_ADAPTER_REPORT.md` records the completed follow-up. A
3,517-parameter, zero-initialized temporal logit adapter is trained on top of
frozen DyHGCN and DisenIDP anchors and selected with an exact epoch-0 fallback.
Both architectures are complete on all four datasets with seeds 21, 42, 84,
126, and 168 (40 adapter runs).

DyHGCN is now complete over five seeds on all four datasets. Mean MAP@100
improves by +0.00302/+0.00063/+0.00156/+0.00463 on Christianity/Android/
Douban/Twitter. Christianity, Douban, and Twitter are positive in 5/5 seeds
with p=0.03125; Android has two positive and three exact fallback seeds
(p=0.25). Worst MAP is significant on Douban and Twitter. Across 20 runs,
2,664/6,115/8,279 protected hits are retained with zero violations.

DisenIDP mean MAP@100 improves by +0.00328/+0.00065/+0.01080/+0.00526 on the
same datasets. Every cell is positive in 5/5 seeds with exact one-sided
`p=0.03125`. Worst-period MAP is positive in 5/5 seeds on Christianity, Douban,
and Twitter; Android is mixed (3/5, `p=0.46875`). Across its 20 runs,
3,486/7,454/9,652 protected hits are retained with zero violations. These are
post-freeze validation-only results: the consumed legacy test split remains
closed and is not reused for model selection or a revised test claim.

`MEMETRACKER_CONFIRMATION_REPORT.md` records the independent confirmatory
benchmark. The released compact timestamps are deterministically restored by
inserting the omitted decimal point after six digits; all 10,130 retained
cascades are then nondecreasing and split chronologically into 7,091/1,013/2,026
train/validation/test cascades. The selection loader never opens the sealed
test JSON. After all ten anchor and adapter checkpoint hashes plus the evaluator
hash were frozen, the test was evaluated once. DyHGCN improves MAP@100 from
0.09868 ± 0.00236 to 0.10631 ± 0.00242 (+0.00764), while DisenIDP improves
from 0.08846 ± 0.00296 to 0.09781 ± 0.00353 (+0.00935). Both mean and
worst-period metrics improve in 5/5 seeds (`p=0.03125`), with
1,790/6,055/10,554 protected hits retained at K=10/50/100 and zero violations.
All 63 unit tests passed immediately before the selection manifest was frozen.

`MEMETRACKER_COMPONENT_ABLATION_REPORT.md` records the post-confirmation,
validation-only component study. Thirty new runs cover two anchors, three
ablations, five paired seeds, and never materialize or reuse the sealed test.
Keeping only historical candidate popularity trails full STLA by 0.00273 on
DyHGCN and 0.00244 on DisenIDP, with full higher in 5/5 seeds for each
(`p=0.03125`). Environment and prefix context have smaller,
architecture-dependent effects, so the paper explicitly avoids claiming that
every component is uniformly necessary. The current regression suite contains
69 passing tests.

Rebuild or resume this validation-only study with:

```powershell
D:\conda\envs\cgt_gpu128\python.exe -m www2027.run_memetracker_adapter_ablations
D:\conda\envs\cgt_gpu128\python.exe -m www2027.summarize_memetracker_adapter_ablations
```

`DEDIF_RELATION_REPORT.md` makes the relationship to the CIKM model direct.
The unchanged original `DeDiff` class and loss are retrained with the corrected
temporal protocol on Christianity and Android, then frozen beneath the same
logit adapter. Corrected DeDiff reaches validation MAP@100 0.09324 and 0.02746;
both adapter runs select the exact epoch-zero fallback, preserving 7/33/48 and
10/40/62 protected hits with zero violations. A rank-eight candidate residual
also falls back on Christianity. This establishes safe output compatibility but
not an output-adapter improvement.

`DYNAMIC_DEDIFF_REPORT.md` records the successful internal follow-up. It
rewrites `(A @ D) @ X` as `A @ (D @ X)`, uses sparse graph propagation, and
adds a past-conditioned rank-eight correction inside DeDiff's causal/bias
decomposition. Across five seeds, mean validation MAP@100 improves by +0.00262
on Christianity (4 positive, 1 fallback, p=0.0625) and +0.00051 on Android
(3 positive, 2 fallback, p=0.125). Mean worst-period deltas are +0.00327 and
+0.00069. All 72/320/535 protected hits at K=10/50/100 are retained with zero
violations. Optional rank-256 SVD factors remove 69.0%/82.5% of the static
`D` floats; Android is nearly lossless while Christianity has a visible
robustness cost. The sparse associative path cuts incremental peak allocation
but is not faster on these small graphs. Forty-eight unit tests pass.

`PAPER_FAITHFUL_BASELINE.md` separates the released dense `A @ D` model from
the equations actually printed in the CIKM PDF. The independent formula path
implements the shared sparse `sigmoid(P Q^T)` graph mask, four GCN views,
score-level social-temporal attention, causal dot-product prediction, explicit
bias KL direction, and weighted composite loss. All ambiguity resolutions are
frozen in `config/paper_faithful_v1.json`; they do not overwrite legacy code or
claim that the PDF tables have already been reproduced. The report also audits
the bundled datasets and released social/interaction channel reversal. A
train/valid-only Christianity smoke completes successfully without opening the
test file. The full regression suite now contains 61 passing tests.

Run the bounded formula-faithful engineering smoke with:

```powershell
D:\conda\envs\cgt_gpu128\python.exe -m www2027.train_paper_faithful --dataset christian --epochs 1 --minimum-epochs 1 --patience 1 --steps-per-epoch 1 --max-eval-batches 1 --batch-size 8 --max-prefix-length 30 --checkpoint www2027\checkpoints\paper_faithful\christian_smoke_s21.pt --result-json www2027\artifacts\paper_faithful\christian_smoke_s21.json
```

Run the corrected direct chain with:

```powershell
D:\conda\envs\cgt_gpu128\python.exe -m www2027.train_temporal_dediff --dataset christian --checkpoint www2027\checkpoints\postfreeze_temporal_dediff\dediff_christian_seed21.pt --result-json www2027\artifacts\postfreeze_temporal_dediff\dediff_christian_seed21.json
D:\conda\envs\cgt_gpu128\python.exe -m www2027.train_dediff_logit_adapter --anchor-result www2027\artifacts\postfreeze_temporal_dediff\dediff_christian_seed21.json --checkpoint www2027\checkpoints\postfreeze_dediff_adapter\dediff_christian_seed21.pt --result-json www2027\artifacts\postfreeze_dediff_adapter\dediff_christian_seed21.json
D:\conda\envs\cgt_gpu128\python.exe -m www2027.train_dynamic_dediff_adapter --anchor-result www2027\artifacts\postfreeze_temporal_dediff\dediff_christian_seed21.json --temporal-rank 8 --checkpoint www2027\checkpoints\postfreeze_dynamic_dediff\dediff_christian_rank8_seed21.pt --result-json www2027\artifacts\postfreeze_dynamic_dediff\dediff_christian_rank8_seed21.json
```

Run or resume this gate with:

```powershell
D:\conda\envs\cgt_gpu128\python.exe -m www2027.run_postfreeze_strong_adapter
```

Rebuild this structural ablation and its aggregate with:

```powershell
D:\conda\envs\cgt_gpu128\python.exe -m www2027.run_postfreeze_mask_ablation
$dynamic = Get-ChildItem www2027\artifacts\postfreeze_validation_inference_ablation\*.json
$mask = Get-ChildItem www2027\artifacts\postfreeze_mask_ablation_training\*.json
D:\conda\envs\cgt_gpu128\python.exe -m www2027.summarize_mask_ablation --dynamic-json $dynamic.FullName --mask-json $mask.FullName --output-json www2027\artifacts\postfreeze_mask_ablation_summary.json
```

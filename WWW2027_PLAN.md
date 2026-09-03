# WWW 2027 Research Plan: DriftDiff

## 1. Positioning

Working title:

> DriftDiff: Environment-Adaptive Debiasing for Robust Information Diffusion on Evolving Social Webs

Target track: **Social Networks and Social Media**. Secondary option: **Graph Algorithms and Modeling for the Web**.

This work must be treated as a new conference paper rather than an extended CIKM paper. The CIKM DeDiff paper is prior work and should be cited in the third person. The new paper studies a different problem: next-user prediction when popularity and exposure patterns change across Web environments.

## 2. Empirical Motivation

We sorted the chronological train+validation stream by cascade start time, split
it into six equal-count temporal windows, and measured five consecutive-window
transitions without using the held-out test.

| Dataset | Mean popularity JSD | Mean Top-20% hub Jaccard | Mean active-user churn |
|---|---:|---:|---:|
| Android | 0.283 | 0.316 | 0.477 |
| Christianity | 0.279 | 0.482 | 0.613 |
| Douban | 0.310 | 0.391 | 0.658 |
| Twitter | 0.340 | 0.329 | 0.571 |

Lower hub Jaccard and higher churn indicate substantial temporal changes in the users dominating diffusion. The reproducible audit defines the top 20% separately among active users in each window; exact per-transition values are saved in `www2027/artifacts/drift_audit_train_valid_6windows.json`.

As a diagnostic only, the existing checkpoints were evaluated over four chronological slices of the test set:

| Dataset | H@100 range | MAP@100 range | H@10 range | MAP@10 range |
|---|---:|---:|---:|---:|
| Android | 0.2559-0.3587 | 0.0530-0.1189 | 0.0775-0.1659 | 0.0471-0.1133 |
| Christianity | 0.5755-0.6615 | 0.1915-0.2233 | 0.2925-0.3608 | 0.1807-0.2136 |

These checkpoint results are not final paper results because the original code selected checkpoints on the test set. They only demonstrate that performance is strongly environment-dependent.

## 3. Problem Definition

Let temporal Web environments be indexed by `e`. Each environment contains social and interaction graphs, cascades, and a popularity vector:

`G_e = (V_e, E^S_e, E^I_e)`, `D_e = {C_e}`, and `pi_e`.

The goal is to predict the next activated user while:

1. maximizing average ranking accuracy;
2. maximizing worst-environment accuracy;
3. reducing sensitivity to changes in `pi_e`;
4. avoiding access to future edges, cascades, or popularity statistics.

The paper should use the terms **stable diffusion signal** and **popularity shortcut** unless explicit causal identification assumptions and interventions are introduced. This avoids overclaiming causality from observational data.

## 4. Proposed Method

### 4.1 Past-only environment encoder

For each environment, compute a context representation from statistics available before prediction:

`z_e = EnvEncoder(degree histogram, activity histogram, cascade statistics, hub turnover)`.

Do not assign an unconstrained learned ID embedding to each environment, because unseen test environments would have no valid embedding.

### 4.2 Environment-conditioned sparse low-rank mask

For an observed edge `(i,j)`:

`m_ij^e = sigmoid((p_i * s_e)^T (q_j * t_e))`,

where `s_e = W_s z_e` and `t_e = W_t z_e`.

The stable and shortcut graph weights are:

`A_stable^e(i,j) = A^e(i,j) * m_ij^e`,

`A_shortcut^e(i,j) = A^e(i,j) * (1 - m_ij^e)`.

The mask must be evaluated only for sparse observed edges. Complexity should be `O((N + E_env)K + |E_graph|K)`, not a dense `N x N` transformation.

### 4.3 Frozen-anchor temporal popularity residual

The selected ERM anchor is frozen. For candidate `v`, build five past-only
features from cumulative and recent activity:

`phi_e(v) = [z(hist_v), z(recent_v), z(recent_v)-z(hist_v), dormant_v, emerging_v]`.

A small sample-conditioned gate predicts five coefficients and adds
`r(v|c,e) = beta(c,e)^T phi_e(v)` to the anchor score. Only this 1,477-parameter
gate is trained. The final gate layer is zero-initialized, so optimization starts
exactly at the anchor.

### 4.4 Hierarchical protected union

At inference, process K=10, 50, and 100 in order and insert every non-head or
non-recent candidate retrieved by the anchor into the corresponding adaptive
prefix. Because the protected sets are nested, the final ranking simultaneously
preserves all protected anchor hits at all three cutoffs. This deterministic
property, not the training regularizer, supplies the formal guarantee.

GroupDRO, V-REx, balanced losses, and unconstrained residuals were negative
validation ablations and are not part of the final method.

## 5. Temporal Evaluation Protocol

1. Sort cascades by their start timestamp.
2. Use the earliest 70% for training environments, the next 10% for validation, and the final 20% for test.
3. Split training into 4-5 chronological environments.
4. Build every graph cumulatively from past-only cascades.
5. Tune all hyperparameters on validation environments only.
6. Report five seeds with mean, standard deviation, and paired significance tests.

### Metrics

- Hits@10/50/100 and MAP@10/50/100;
- worst-environment Hits/MAP;
- relative performance drop from best to worst period;
- head/mid/tail user Hits/MAP;
- Spearman correlation between recommendation frequency and historical popularity;
- average popularity of recommended users versus ground-truth users;
- performance on newly active and low-frequency users;
- latency, peak memory, and scaling with users/edges.

### Stress tests

- hub amplification: strengthen the same popular users;
- hub turnover: replace training-period hubs with different test-period hubs;
- emerging users: introduce users with limited historical observations;
- abrupt versus gradual popularity drift;
- different environment window sizes.

## 6. Baselines and Ablations

Baseline families:

1. sequence and diffusion prediction models used by DeDiff;
2. static DeDiff;
3. dynamic graph diffusion models;
4. representative graph OOD/domain-generalization methods adapted to next-user ranking;
5. simple temporal retraining and popularity-reweighting baselines.

Required ablations:

- static mask instead of environment-conditioned mask;
- global instead of local popularity target;
- no environment-risk variance;
- no stable-shortcut separation;
- learned environment ID versus past-only environment encoder;
- dense versus sparse implementation only as an efficiency comparison, not as the main model.

## 7. Novelty Boundary from CIKM DeDiff

| Dimension | CIKM DeDiff | WWW DriftDiff |
|---|---|---|
| Research question | static popularity bias | temporal popularity drift and environment generalization |
| Decomposition | one global mask | past-conditioned dynamic sparse mask |
| Temporal adaptation | none | frozen anchor plus past-only five-feature residual |
| Safety | soft representation separation | exact protected-hit preservation at K=10/50/100 |
| Evaluation | in-distribution and static hub amplification | natural drift, hub turnover, emerging influx, worst-period and subgroup metrics |
| Artifact | current dense implementation | reproducible sparse edge implementation |

## 8. Code Structure

Preserve the CIKM implementation and add new code under `www2027/`:

```text
www2027/
  data/
    temporal_split.py
    rolling_graph.py
    sequence_dataset.py
  models/
    environment_encoder.py
    dynamic_low_rank_mask.py
    sparse_propagation.py
    temporal_diffusion.py
  training/
    objectives.py
    protocol.py
  metrics/
    drift.py
    ranking.py
  train_temporal.py
  audit_drift.py
```

Implementation status on Aug 31: the independent prototype includes
chronological splitting, recent-plus-cumulative past-only graph context, sparse
environment-conditioned masks, a GRU next-user predictor, negative robust-loss
ablations, delayed test materialization, and subgroup metrics. Thirty-three unit
tests pass. The final anchored dual-path method freezes the ERM predictor,
trains a 1,477-parameter temporal residual, and forms one hierarchical protected
union at K=10/50/100. Its configuration and 20 checkpoint hashes were frozen
before a one-shot five-seed test. Christianity, Android, and Douban improve MAP
and worst MAP at all cutoffs in all five seeds (`p=0.03125`); Twitter is a
negative result, with MAP@100 delta -0.00043 and worst-MAP delta -0.00204. The
exact test guarantee covers 51/418/874 protected anchor hits with zero
violations and no protected-stratum Hit@K decline. A post-freeze validation-only
audit over all 20 checkpoints and 120 stress conditions preserves
8,042/17,078/24,424 protected hits with zero violations; hub turnover and
emerging influx, rather than hub amplification, expose the strongest Twitter
sensitivity. See `www2027/FINAL_ONE_SHOT_TEST_REPORT.md` and
`www2027/POSTFREEZE_STRESS_REPORT.md`. The test has been consumed and no
post-test tuning is permitted.

The post-loader protocol audit now reports exact users, 70/10/20 cascade
counts, prefix-example counts, and rolling sparse-edge ranges. A five-seed
validation-only inference ablation shows that Top-100-only fusion loses 38/66
protected hits at K=10 and 112/331 at K=50, whereas the hierarchical rule has
zero violations at every cutoff. The residual adds 1,477 parameters; the final
local benchmark reports median adaptive-plus-fusion latency below 23 ms and
peak allocated CUDA memory below 98 MiB per batch on all four datasets. See
`www2027/INFERENCE_ABLATION_AND_EFFICIENCY_REPORT.md`.

The five-seed no-preservation-loss retraining is also complete. The soft loss
has mixed effect on Christianity, is neutral on Android, is suggestive on
Douban, and improves Twitter validation MAP@100 by 0.00379 in all five seeds.
Both variants have zero post-fusion violations, so the paper separates its
optimization role from the deterministic guarantee. See
`www2027/PRESERVATION_ABLATION_REPORT.md`.

The five-seed structural anchor ablation is complete for dynamic, static, and
no-mask variants (40 new validation-only retraining runs). Dynamic masking
beats no mask in all five seeds on Christianity and Android, but not on Douban
or Twitter; it is not significantly better than a static mask on any dataset.
The WWW claim is therefore explicitly dataset-dependent. See
`www2027/MASK_ABLATION_REPORT.md`.

A corrected seed-21, validation-only readiness gate is now complete for
DyHGCN, MS-HGAT, and DisenIDP on all four datasets. All 12 settings beat the
current seed-21 DriftDiff path in mean and worst MAP@100; the largest gap is
DyHGCN on Twitter (0.19237 versus 0.01371). This changes the project status:
the exact hierarchical safety layer remains viable, but the current backbone
is a no-go for a WWW main-method claim. Five-seed baseline expansion is paused
until the residual and protected union are attached to a strong frozen anchor.
See `www2027/TEMPORAL_BASELINE_REPORT.md`.

The first strong-backbone migration gate now passes. A 3,517-parameter
backbone-agnostic temporal logit adapter improves frozen DyHGCN validation
MAP@100 on Christianity, Douban, and Twitter by +0.00404, +0.00114, and
+0.00613; Android selects the exact epoch-0 anchor fallback. Worst MAP follows
the same three-positive/one-flat pattern. Hierarchical fusion retains
565/1,253/1,652 protected hits at K=10/50/100 with zero violations. This is
single-seed, validation-only evidence; it authorizes multi-seed replication,
not a new test pass. See `www2027/STRONG_BACKBONE_ADAPTER_REPORT.md`.

The second-architecture gate also passes on DisenIDP. Seed-21 mean MAP@100
improves on Christianity/Android/Douban/Twitter by
+0.00402/+0.00099/+0.01008/+0.00486, and worst MAP improves on all four.
Together, DyHGCN and DisenIDP preserve 1,271/2,725/3,593 protected hits at
K=10/50/100 with zero violations. MS-HGAT is now optional rather than a blocker.
DisenIDP Christianity is also complete over five seeds: mean MAP@100 delta is
+0.00328 and worst-MAP delta is +0.00413, both positive in 5/5 seeds with exact
one-sided p=0.03125. The remaining DisenIDP dataset cells are still single-seed.

The complete DyHGCN five-seed matrix is now available. Mean MAP@100 deltas on
Christianity/Android/Douban/Twitter are +0.00302/+0.00063/+0.00156/+0.00463.
The first, third, and fourth datasets are positive in 5/5 seeds with p=0.03125;
Android has two positive and three exact fallback seeds with no negative seed.
Across all 20 runs, 2,664/6,115/8,279 protected hits are preserved with zero
violations.

The direct-lineage gate is also complete on the two datasets where the
unchanged dense CIKM implementation is tractable. Corrected DeDiff obtains
seed-21 validation MAP@100 0.09324 on Christianity and 0.02746 on Android. The
same logit adapter, as well as a rank-eight candidate variant on Christianity,
selects the exact epoch-zero fallback; all protected hits are preserved with
zero violations. This proves safe compatibility but not accuracy improvement
from output calibration. The subsequent internal dynamic DeDiff gate passes:
the rank-eight past-conditioned correction improves five-seed mean MAP@100 by
+0.00262 on Christianity (4 positive, 1 fallback, p=0.0625) and +0.00051 on
Android (3 positive, 2 fallback, p=0.125). Mean worst-MAP deltas are +0.00327
and +0.00069. Rank-256 static compression removes 69.0%/82.5% of `D` storage
and is nearly lossless only on Android. See
`www2027/DEDIF_RELATION_REPORT.md` and
`www2027/DYNAMIC_DEDIFF_REPORT.md`.

## 9. Schedule

- Aug 29-Sep 6: repair and reproduce the static sparse baseline;
- Sep 7-Sep 14: temporal split, rolling graphs, and drift benchmark;
- Sep 15-Sep 22: environment encoder and dynamic low-rank mask;
- Sep 23-Sep 29: stability objective and main baseline runs;
- Sep 30-Oct 6: all datasets, five seeds, stress tests, and ablations;
- Oct 7-Oct 10: complete the eight-page main paper and register the abstract;
- Oct 11-Oct 17: final experiments, leakage audit, proofreading, and submission.

## 10. Go/No-Go Criteria

The original method passed the three-dataset accuracy and sparse-scaling gates,
but the strong-baseline readiness gate invalidates its backbone. The project is
now a **conditional go for a backbone-agnostic safe-adaptation layer**, not yet
a submission-ready three-positive/one-negative paper. The remaining submission
gates are:

1. the corrected DyHGCN five-seed matrix is complete;
2. expand remaining DisenIDP cells only if the substantially higher compute
   cost is justified; Christianity five-seed significance is already complete;
3. confirmation of the turnover/influx hypothesis on new untouched data;
4. final causal-wording, leakage, anonymity, and reproducibility audits.

If new-data accuracy does not support the mechanism, retain the exact safety
result and position turnover/influx strictly as a limitation rather than adding
post-hoc losses to the consumed benchmark.

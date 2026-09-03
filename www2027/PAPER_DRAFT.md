# DriftDiff: Subgroup-Safe Information Diffusion Prediction under Temporal Popularity Drift

> Working WWW manuscript. All reported test values come from the frozen
> one-shot protocol. Bracketed TODOs mark experiments or citations that do not
> yet exist and must not be replaced with invented results.

## Abstract

Information diffusion predictors are commonly trained on historical cascades
whose user activity and graph structure are treated as stationary. In deployed
social systems, however, popular users, active communities, and exposure paths
change over time. A model that adapts to recent popularity can improve future
ranking, but can also suppress low-frequency or temporarily inactive users.
We study next-user diffusion prediction under natural temporal popularity
drift using chronological environments, rolling past-only graphs, and
worst-period and user-stratified evaluation. We propose DriftDiff, a dual-path
predictor that combines an environment-conditioned sparse low-rank graph
encoder with a lightweight temporal popularity residual. The stable anchor is
selected first and then frozen; only a 1,477-parameter residual gate is tuned.
At inference, a hierarchical protected-union operator preserves every
non-head or non-recent candidate retrieved by the anchor at K=10, 50, and 100
while otherwise retaining the adaptive ranking. This yields an exact
prefix-preservation guarantee using only past activity. In a hash-locked
five-seed one-shot evaluation, DriftDiff significantly improves mean and
worst-period MAP at all three cutoffs on Christianity, Android, and Douban
(one-sided exact paired p=0.03125), with MAP@100 gains of 0.00225, 0.00322,
and 0.00995. Across all datasets, 874 protected anchor hits at K=100 are
preserved with zero violations. Twitter is a negative result: a strong
validation gain reverses on the final period, exposing a limit of popularity
adaptation under regime change. A five-seed validation-only sensitivity audit
localizes this weakness to hub-identity turnover and emerging-user influx,
while simple hub amplification has little effect. These findings support
subgroup-safe temporal adaptation rather than universal temporal
generalization.

## 1 Introduction

Predicting the next participant in an information cascade supports rumor
monitoring, campaign analysis, and online diffusion modeling. Modern methods
combine cascade sequences with social or interaction graphs, but usually learn
from one historical distribution. This assumption is fragile. The users who
dominate one time window may become inactive later, new users may emerge, and
hub identities may turn over even when the overall activity distribution looks
similar.

The CIKM DeDiff study addresses popularity shortcuts through a global low-rank
graph decomposition and representation-level separation. Its robustness test
artificially oversamples edges incident to high-degree training users while
leaving validation and test fixed. That experiment establishes sensitivity to
amplified static popularity, but does not answer three deployment questions:

1. How much do popularity and active-user identities change naturally over
   chronological periods?
2. Can a predictor adapt using only information available before a target
   period?
3. Can adaptation improve accuracy without removing low-frequency or dormant
   candidates already recovered by a stable model?

We answer these questions with a leakage-safe temporal protocol and DriftDiff.
The core design deliberately separates *adaptation* from *safety*. A frozen
anchor extracts a stable diffusion signal from a rolling graph. A small
residual uses cumulative and recent past activity to adjust candidate scores.
Because unconstrained residuals can erase subgroup coverage, a deterministic
hierarchical fusion layer makes the final safety property exact rather than
expecting a soft loss to enforce it.

Our contributions are:

- We formulate natural temporal popularity drift for next-user diffusion
  prediction using chronological train/validation/test environments, rolling
  past-only graphs, and mean, worst-period, and user-stratified metrics.
- We introduce a sparse environment-conditioned low-rank graph encoder and a
  frozen-anchor temporal residual whose trainable adaptation head contains only
  1,477 parameters in the reported configuration.
- We develop a hierarchical protected-union ranking that simultaneously
  preserves protected anchor hits at K=10, 50, and 100, and prove that later
  prefix updates cannot invalidate earlier guarantees.
- We report a pre-registered five-seed one-shot evaluation. Three datasets show
  significant mean and worst-period improvements; all 874 protected anchor
  hits at K=100 are preserved; Twitter exposes a genuine failure under later
  drift.

We use the terms **stable diffusion signal** and **popularity shortcut**. The
observational data and losses do not identify causal effects, so the WWW paper
must not label learned edges or representations as causal.

## 2 Related Work

### 2.1 Information diffusion prediction

[TODO: condense sequence, graph, hypergraph, continuous-time, retrieval, and
large-scale diffusion models. Reuse factual citations from DeDiff only after
checking their final bibliographic metadata. Add WWW 2026 GRID and LSID as
recent task-specific baselines.]

### 2.2 Popularity bias and graph debiasing

DeDiff separates diffusion-relevant and popularity-related signals using a
global low-rank edge mask, two graph branches, asymmetric supervision, and
representation disagreement. Other graph debiasing and recommendation methods
study popularity, confounding, or invariance in different prediction settings.
Our problem differs in that both candidate activity and graph evidence evolve
chronologically, and no future-period popularity may be used as an input.

### 2.3 Temporal distribution shift and safe ranking

[TODO: add primary literature on temporal graph distribution shift, online or
continual graph learning, and constrained/safe ranking. Distinguish our exact
prefix guarantee from exposure parity and calibration objectives.]

## 3 Problem Formulation

Let cascades be ordered by start time. Environment e contains target cascades
`D_e`, while its input graph `G_<e`, cumulative activity `p_<e`, and recent
activity `p_recent(e)` are built only from earlier cascades. Given a prefix
`C_k`, the model ranks unactivated users as candidates for the next activation.

We evaluate:

- Hits@K and MAP@K for K in {10, 50, 100};
- the mean and minimum metric across chronological target environments;
- head, mid, tail, emerging, recent-active, and historical-inactive strata;
- exact preservation violations for protected anchor hits.

Head and recent-active labels are computed independently for each target
environment from preceding activity. A candidate is protected if it is
non-head or non-recent. These labels are audit and safety metadata, not future
inputs.

## 4 Method

### 4.1 Past-only environment encoder

For cumulative and most-recent history, we summarize activity and graph degree
with active-user fraction, log-count moments and quantiles, Gini coefficient,
normalized entropy, top-1% and top-5% mass, and degree moments. Concatenating
the two 14-dimensional summaries gives `x_e in R^28`. A small MLP produces an
environment context:

`z_e = EnvEncoder(x_e)`.

There is no learned environment-ID embedding, so unseen future environments do
not require a new parameter.

### 4.2 Environment-conditioned sparse low-rank mask

For an observed edge `(i,j)` with weight `a_ij`, node factors `p_i,q_j in R^r`
are modulated by the past context:

`m_ij^e = sigmoid(((p_i * s_l(z_e))^T (q_j * s_r(z_e))) / sqrt(r) + b(z_e))`.

The stable and shortcut edge weights are

`a_stable,ij^e = a_ij m_ij^e`,

`a_shortcut,ij^e = a_ij (1 - m_ij^e)`.

The mask is evaluated only on supplied sparse edges. Its storage is `O(Nr)`
and edge scoring is `O(|E_e|r)`; no dense `N x N` mask is materialized.

The stable graph embeddings and elapsed-time features are encoded by a GRU to
obtain cascade representation `h_c`. The anchor score for candidate v is

`s_0(v | c,e) = h_c^T u_v / sqrt(d) + b_v`.

### 4.3 Frozen-anchor temporal popularity residual

For each candidate, we construct five past-only features:

`phi_e(v) = [z(hist_v), z(recent_v), z(recent_v)-z(hist_v), dormant_v, emerging_v]`.

A sample-conditioned gate maps `[h_c; z_e]` to five coefficients:

`beta(c,e) = MLP([h_c; z_e])`,

`r(v | c,e) = beta(c,e)^T phi_e(v)`.

The adaptive score is `s_1 = s_0 + r`. The final layer of the gate is
zero-initialized, so adaptation starts exactly from the anchor. The anchor is
selected on validation and frozen; only the residual gate is trained. A
Top-100 hinge penalizes protected anchor candidates that fall below the
adaptive Top-100 boundary. This soft regularizer aids training but is not the
source of the formal guarantee.

### 4.4 Hierarchical protected-union ranking

Let `A_K` be the anchor Top-K and let `P_K` be protected candidates in `A_K`.
Starting from the adaptive Top-100 list L, process K=10, 50, and 100 in order.
For every item in `P_K` missing from the first K positions, either swap it
forward from a later position or replace the lowest-ranked item in the prefix
that is not required by `P_K`.

**Proposition 1 (simultaneous prefix preservation).** For every reported
cutoff K, `P_K` is a subset of the final fused Top-K.

**Proof sketch.** The construction explicitly inserts every missing member of
`P_K` while never replacing an item required by `P_K`. For increasing cutoffs,
the required sets are nested: `P_10 subseteq P_50 subseteq P_100`. Therefore a
later update cannot remove a member required at an earlier cutoff. Swapping an
item already in the Top-100 preserves uniqueness; replacing an absent item
preserves list capacity. Induction over the ordered cutoffs gives the claim.

The guarantee is conditional: it preserves protected hits already recovered by
the anchor. It does not guarantee reciprocal rank, head/recent-active accuracy,
or recall of protected targets missed by the anchor.

## 5 Experimental Protocol

### 5.1 Data and chronological environments

We use Christianity, Android, Douban, and Twitter. Cascades are sorted by start
time and split 70/10/20 with equal timestamps kept together. Training is split
into four environments, validation into two, and test into three. Test graphs
and tensors are not materialized before all selections are frozen.

| Dataset | Users | Cascades T/V/Te | Prefix examples T/V/Te | Rolling directed edges |
| --- | ---: | ---: | ---: | ---: |
| Christianity | 1,652 | 412/59/118 | 9,246/560/944 | 1,652-44,620 |
| Android | 2,928 | 475/67/136 | 13,350/1,201/2,152 | 2,928-102,648 |
| Douban | 12,233 | 2,432/348/695 | 36,149/6,249/11,295 | 12,233-240,633 |
| Twitter | 12,628 | 2,404/344/687 | 46,487/5,273/11,361 | 12,628-316,800 |

Prefix length is capped at 50. Edge ranges run from the earliest training
snapshot to the last past-only test snapshot and include one self-loop per
vocabulary node.

### 5.2 Selection and statistical protocol

Five seeds are fixed to 21, 42, 84, 126, and 168. Checkpoints maximize mean
validation MAP@100. Residual learning rate is `1e-3` on Christianity/Douban and
`1e-4` on Android/Twitter. The 20 checkpoint paths, epochs, hashes, evaluator
hash, and reranking cutoffs are frozen before one-shot test materialization.
We report mean, sample standard deviation, positive-seed count, and the exact
one-sided paired sign-flip p-value. With five pairs, the minimum p-value is
0.03125.

### 5.3 Baselines and ablations

Required comparison families:

1. original task baselines under the same temporal protocol: DyHGCN, MS-HGAT,
   DisenIDP, RotDiff, MINDS, GODEN, CARE, GRID, and LSID where runnable;
2. the CIKM DeDiff architecture retrained under the corrected temporal split;
3. simple temporal baselines: last-window retraining, historical popularity,
   recent popularity, and score interpolation;
4. DriftDiff ablations: static mask, no mask, no residual, free residual,
   no preservation loss, Top-100-only fusion, and no protected union;
5. negative robust-objective ablations: GroupDRO, V-REx, balanced loss, and the
   subgroup-risk constraint already screened on validation.

[TODO: the five-seed anchor/residual/fusion ablation, structural mask
retraining ablation, and negative objective screens are complete. A corrected
seed-21 readiness gate for DyHGCN, MS-HGAT, and DisenIDP is complete, but it
invalidates the current backbone as a competitive main method. Do not promote
the single-seed values to the main result table. First migrate the adapter and
protected union to a strong frozen backbone, then run five seeds.]

## 6 Results

### 6.1 Frozen one-shot main result

| Dataset | Anchor MAP@100 | DriftDiff MAP@100 | Delta | Positive seeds | p | Delta worst MAP@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Christianity | 0.07623 ± 0.00617 | 0.07848 ± 0.00570 | +0.00225 | 5/5 | 0.03125 | +0.00400 |
| Android | 0.01242 ± 0.00110 | 0.01565 ± 0.00121 | +0.00322 | 5/5 | 0.03125 | +0.00144 |
| Douban | 0.05877 ± 0.00340 | 0.06872 ± 0.00200 | +0.00995 | 5/5 | 0.03125 | +0.00362 |
| Twitter | 0.00608 ± 0.00128 | 0.00565 ± 0.00047 | -0.00043 | 2/5 | 0.71875 | -0.00204 |

The first three datasets improve mean and worst-period MAP at K=10, 50, and
100 in all five seeds. Twitter reverses its validation gain. This falsifies a
universal temporal-generalization claim and motivates explicit failure
analysis.

### 6.2 Safety guarantee

The final test contains 51, 418, and 874 protected anchor hits at K=10, 50, and
100. There are zero violations at every cutoff. The minimum protected-stratum
Hit@K delta across datasets and seeds is zero.

Overall Hit can still fall because head/recent-active users are not protected.
On Twitter at K=100, head and recent-active Hit decline by 0.01519 and 0.01100,
while mid and tail Hit improve by 0.00364 and 0.00237.

### 6.3 Inference-chain ablation and efficiency

All four paths share one forward pass and are evaluated on validation only.

| Dataset | Anchor | Adaptive | Top-100 union | Hierarchical union |
| --- | ---: | ---: | ---: | ---: |
| Christianity | 0.07945 | 0.08006 | 0.08005 | 0.08006 |
| Android | 0.01614 | 0.01772 | 0.01763 | 0.01761 |
| Douban | 0.04177 | 0.04347 | 0.04345 | 0.04343 |
| Twitter | 0.00500 | 0.01539 | 0.01529 | 0.01508 |

Top-100-only fusion preserves all 647 protected validation hits at K=100 but
loses 38/66 at K=10 and 112/331 at K=50. Hierarchical fusion has zero
violations at all three cutoffs. Its MAP@100 change from the unrestricted
adaptive path ranges from +0.00001 to -0.00031 across datasets.

The residual gate adds 1,477 parameters, or 0.23%-1.52% over the anchor. On an
RTX 5090 D with batch size 64, median anchor/adaptive/fused batch latency is
6.07/6.50/10.25 ms on Christianity, 8.70/10.39/15.67 ms on Android,
9.88/10.97/16.82 ms on Douban, and 11.88/12.48/22.48 ms on Twitter. Absolute
peak allocated CUDA memory for the final pipeline is 21.92/38.28/69.13/97.66
MiB, respectively. Data loading is excluded. Timing and memory are measured in
separate loops. A batched-index fusion implementation is exactly
metric-equivalent to the original and removes per-row GPU synchronization.

### 6.4 Structural mask ablation

We retrain static-mask and no-mask anchors with the same locked five-seed
protocol and compare them with the frozen dynamic anchors. Because this is a
post-test validation-only analysis, it is descriptive and cannot change the
selected method.

| Dataset | Dynamic | Static | No mask | Dynamic - static (pos./p) | Dynamic - none (pos./p) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Christianity | 0.07945 | 0.07828 | 0.07574 | +0.00118 (3/5, 0.15625) | +0.00372 (5/5, 0.03125) |
| Android | 0.01614 | 0.01558 | 0.01530 | +0.00056 (3/5, 0.12500) | +0.00085 (5/5, 0.03125) |
| Douban | 0.04177 | 0.04156 | 0.04144 | +0.00021 (2/5, 0.31250) | +0.00034 (2/5, 0.46875) |
| Twitter | 0.00500 | 0.00490 | 0.00477 | +0.00010 (4/5, 0.31250) | +0.00022 (3/5, 0.25000) |

Dynamic masking beats no masking in all five seeds on Christianity and
Android, including a 5/5 worst-period improvement on Android, but its effect is
not stable on Douban or Twitter. No dataset shows significant evidence that
dynamic conditioning improves over a static mask. Protected-stratum effects
are also heterogeneous: historical-inactive Hit@100 improves over no mask by
0.01576 on Christianity and 0.00182 on Android but declines by 0.00146 on
Twitter. We therefore treat the mask as a dataset-dependent component rather
than a universal source of the final gains.

### 6.5 Preservation-loss ablation

We retrain the 1,477-parameter residual from all 20 frozen bases with the soft
Top-100 preservation weight set from one to zero, then apply the same
hierarchical inference rule. This ablation is validation-only.

| Dataset | No preservation | Main | Delta | Positive seeds | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Christianity | 0.08038 | 0.08006 | -0.00032 | 2/5 | 0.62500 |
| Android | 0.01751 | 0.01761 | +0.00010 | 3/5 | 0.25000 |
| Douban | 0.04266 | 0.04343 | +0.00076 | 4/5 | 0.06250 |
| Twitter | 0.01129 | 0.01508 | +0.00379 | 5/5 | 0.03125 |

The loss is not uniformly beneficial, but it materially improves the Twitter
adaptive solution and is suggestive on Douban. Both settings have zero
hierarchical guarantee violations, confirming that the soft term is an
optimization regularizer rather than the source of formal safety. The mean
fusion correction relative to the free adaptive path changes by less than
0.00031 in every setting; the loss mainly changes which residual solution is
selected. It does not fix the frozen Twitter test failure.

### 6.6 Why Twitter fails

The adaptive path itself underperforms the anchor on the frozen Twitter test,
showing that the failure is not caused solely by protected fusion. Fusion
further exchanges unrestricted reciprocal rank for exact protected coverage.
The evidence is consistent with a change between validation and final temporal
regimes. To narrow the mechanism without reusing test for selection, we first
audit only the chronological train+validation stream. Six equal-count windows
give the following mean adjacent-window descriptors:

| Dataset | Popularity JSD | Top-hub Jaccard | Active-user churn |
| --- | ---: | ---: | ---: |
| Christianity | 0.279 | 0.482 | 0.613 |
| Android | 0.283 | 0.316 | 0.477 |
| Douban | 0.310 | 0.391 | 0.658 |
| Twitter | 0.340 | 0.329 | 0.571 |

Twitter has the highest mean popularity JSD and the largest single transition
(0.424), although Douban has higher active-user churn. Descriptors alone
therefore do not identify the failure.

We next perturb only past recent-popularity input on validation, leaving the
rolling graph, environment context, cumulative history, targets, and all 20
frozen checkpoints unchanged. At full severity, the five-seed MAP@100 effects
are:

| Dataset | Stress | Delta from unperturbed | Delta from anchor | Positive vs. anchor |
| --- | --- | ---: | ---: | ---: |
| Christianity | hub amplification | -0.00029 | +0.00032 | 3/5 |
| Christianity | hub turnover | -0.00109 | -0.00048 | 2/5 |
| Christianity | emerging influx | -0.00107 | -0.00046 | 2/5 |
| Android | hub amplification | -0.00032 | +0.00114 | 5/5 |
| Android | hub turnover | -0.00083 | +0.00064 | 5/5 |
| Android | emerging influx | -0.00082 | +0.00064 | 5/5 |
| Douban | hub amplification | -0.00037 | +0.00129 | 5/5 |
| Douban | hub turnover | -0.00167 | -0.00002 | 3/5 |
| Douban | emerging influx | -0.00166 | -0.00001 | 3/5 |
| Twitter | hub amplification | -0.00017 | +0.00992 | 5/5 |
| Twitter | hub turnover | -0.01101 | -0.00093 | 1/5 |
| Twitter | emerging influx | -0.01103 | -0.00095 | 1/5 |

Hub amplification barely changes Twitter validation behavior, whereas turnover
and influx erase its residual advantage. Across 120 stress conditions, all
8,042/17,078/24,424 protected anchor hits at K=10/50/100 are preserved. This
supports sensitivity to hub-identity relocation as a failure hypothesis, not a
causal explanation: the perturbations are controlled input sensitivity tests,
not complete future-world simulations. Confirming the mechanism requires a new
untouched temporal benchmark.

### 6.7 Strong-baseline readiness gate

After the main test was frozen and consumed, we ran a validation-only,
single-seed readiness gate for three task baselines using a pinned community
integration. This gate cannot change the frozen result and is not a statistical
comparison. It exists to determine whether a corrected five-seed baseline
matrix is worth running before the main architecture is revised.

| Dataset | DriftDiff | DyHGCN | MS-HGAT | DisenIDP |
| --- | ---: | ---: | ---: | ---: |
| Christianity | 0.08328 | 0.08582 | 0.09479 | **0.09647** |
| Android | 0.01895 | **0.02859** | 0.02825 | 0.02829 |
| Douban | 0.04361 | **0.05649** | 0.04565 | 0.04601 |
| Twitter | 0.01371 | **0.19237** | 0.02893 | 0.12541 |

All entries are seed-21 validation MAP@100 with validation-only checkpoint
selection. The test partition was not materialized. Every baseline also beats
DriftDiff in worst MAP@100. The result is a no-go for claiming that the current
backbone is competitive. The protected-union theorem survives because it is a
ranking-layer property, but the empirical method must be rebuilt around a
strong anchor before submission.

### 6.8 Backbone-agnostic adapter gate

We next freeze the corrected DyHGCN anchor and train a 3,517-parameter temporal
logit adapter that consumes no backbone hidden state. Its final layer is zero
initialized, making epoch 0 exactly equal to the anchor and allowing validation
selection to fall back safely.

| Dataset | DyHGCN | Adapted + protected | Delta | Delta worst |
| --- | ---: | ---: | ---: | ---: |
| Christianity | 0.08582 | 0.08986 | +0.00404 | +0.00684 |
| Android | 0.02859 | 0.02859 | 0.00000 | 0.00000 |
| Douban | 0.05649 | 0.05763 | +0.00114 | +0.00235 |
| Twitter | 0.19237 | 0.19850 | +0.00613 | +0.00614 |

These are post-freeze, seed-21 validation-only results and are not significance
claims. They nevertheless pass the incremental-value gate: three datasets
improve and the fourth selects the exact anchor fallback. Across the four
datasets, all 565/1,253/1,652 protected anchor hits at K=10/50/100 are retained
with zero violations.

We replicate the same unchanged adapter on DisenIDP. MAP@100 improves by
+0.00402, +0.00099, +0.01008, and +0.00486 on Christianity, Android, Douban,
and Twitter; worst MAP also improves on all four. Across DyHGCN and DisenIDP,
1,271/2,725/3,593 protected hits are retained with zero violations. This passes
the second-architecture gate without using a backbone hidden state. Five-seed
strong-backbone replication remains required before this table can enter a
submission.

The full DyHGCN five-seed matrix is now complete. Mean MAP@100 deltas are
+0.00302, +0.00063, +0.00156, and +0.00463 on Christianity, Android, Douban,
and Twitter. Christianity, Douban, and Twitter improve in 5/5 seeds with exact
p=0.03125; Android has two positive and three exact fallback seeds (p=0.25),
with no negative final seed. Worst MAP improves significantly on Douban and
Twitter, while Christianity has four positive and one negative worst-period
seed. The 20-run guarantee retains 2,664/6,115/8,279 protected hits at
K=10/50/100 with zero violations.

For DisenIDP on Christianity, five-seed replication is complete. Anchor and
adapted MAP@100 are 0.09485 ± 0.00139 and 0.09813 ± 0.00160, a +0.00328 gain in
all five seeds (exact one-sided p=0.03125). Worst MAP improves by +0.00413 in
all five seeds (p=0.03125). The safety audit retains 40/223/371 protected hits
at K=10/50/100 with zero violations. Other strong-backbone dataset cells remain
single-seed validation evidence.

### 6.9 Direct CIKM DeDiff lineage gate

We additionally retrain the unchanged CIKM `DeDiff` class and its original
loss under the corrected temporal protocol, excluding EOS from real-user
metrics. At seed 21, DeDiff reaches validation MAP@100 0.09324 (worst 0.09005)
on Christianity and 0.02746 (worst 0.02081) on Android. The same 3,517-parameter
logit adapter selects the exact epoch-zero fallback on both datasets, with zero
change in mean or worst MAP. Hierarchical fusion preserves 7/33/48 and
10/40/62 protected anchor hits at K=10/50/100, respectively, with zero
violations. A rank-eight learned candidate residual also fails to improve
Christianity and falls back exactly.

This result establishes direct architectural lineage and no-regression
compatibility, but not an accuracy improvement from output calibration. It
motivates moving the temporal correction inside DeDiff's decomposition.

### 6.10 Environment-conditioned internal DeDiff

Because the CIKM configuration uses one graph-convolution layer, its causal
path `(A D) X` can be reassociated as `A (D X)` without materializing `A D`.
We add the environment-conditioned correction
`U diag(g(s_e)) V^T X` to `D X`, where `s_e` contains past-only cumulative and
recent graph statistics. The original checkpoint is frozen, the final gate is
zero initialized, and only 27,696/48,112 parameters are trained on
Christianity/Android.

| Dataset | DeDiff | Dynamic DeDiff | Delta | Positive seeds | p | Delta worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Christianity | 0.08301 ± 0.01058 | 0.08563 ± 0.01162 | +0.00262 | 4/5 | 0.06250 | +0.00327 |
| Android | 0.02784 ± 0.00156 | 0.02835 ± 0.00107 | +0.00051 | 3/5 | 0.12500 | +0.00069 |

These five-seed validation-only results are the first direct average
improvement over the corrected DeDiff anchor. The remaining seeds select the
exact anchor fallback, yielding no negative mean-MAP seed, but neither exact
paired test reaches 0.05. Christianity worst MAP has one negative seed, so the
result is not evidence of uniform robustness. A rank-256 SVD replacement for
the static `D` removes 69.0% and 82.5% of its stored floats. It is nearly
lossless on Android but reduces Christianity worst-period MAP. The sparse
associative path reduces incremental peak allocation by 45%-56% but is 3%-10%
slower end to end on these small graphs because sparse-kernel overhead
dominates. We therefore claim lower memory and removal of the cubic dense
intermediate, not a latency improvement.

## 7 Discussion and Limitations

DriftDiff provides a narrow but auditable safety property. It does not enforce
demographic fairness, causal influence recovery, or accuracy parity. Protection
depends on the anchor retrieving the target and on group labels computed from
past activity. The method can sacrifice unprotected head/recent accuracy, as
observed on Twitter.

The corrected strong-baseline readiness gate exposes a more fundamental
limitation: the current sequential backbone is weaker than all three evaluated
task baselines in every seed-21 dataset setting. Consequently, the present
accuracy tables are not submission-ready even though the post-processing
guarantee is exact. A WWW version must demonstrate incremental value on a
competitive frozen backbone.

The one-shot test has been consumed. No Twitter-specific checkpoint, learning
rate, fallback, or reranking change may be selected using this result. Future
repairs require new datasets or a newly defined nested-validation benchmark.

## 8 Conclusion

Temporal popularity adaptation is useful but unsafe when treated as an
unconstrained score correction. DriftDiff separates a frozen stable path from a
lightweight adaptive path and adds an exact hierarchical preservation layer.
The approach gives significant gains on three datasets and zero protected-hit
violations on all four, while Twitter demonstrates that safe coverage does not
imply universal accuracy under future regime change.

## Author checklist before submission

- [x] Run the seed-21 corrected temporal readiness gate for three task baselines.
- [x] Attach the temporal residual and protected union to a strong frozen backbone.
- [x] Pass the seed-21 DyHGCN strong-backbone incremental-value gate.
- [x] Implement and pass the seed-21 direct internal DeDiff gate on two datasets.
- [x] Replicate dynamic DeDiff over the remaining four seeds.
- [ ] Run corrected DyHGCN plus the adapter for the remaining four seeds.
- [ ] Replicate the adapter on a second strong backbone.
- [x] Add train+validation drift and five-seed validation-only stress tables.
- [ ] Confirm the turnover/influx failure hypothesis on a new untouched dataset.
- [x] Report anchor, residual, and fusion parameters and latency separately.
- [x] Add the five-seed inference-chain ablation and multi-prefix safety audit.
- [x] Add the five-seed no-preservation-loss retraining ablation.
- [x] Add static-mask and no-mask structural retraining ablations.
- [ ] Audit all causal wording inherited from CIKM.
- [ ] Verify WWW page limit, anonymity, artifact, and GenAI-disclosure rules at
      submission time.

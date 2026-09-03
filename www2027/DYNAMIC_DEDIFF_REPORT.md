# Environment-Conditioned Dynamic DeDiff

## Motivation and direct relationship

The output-only logit adapter safely wrapped corrected DeDiff but selected the
exact anchor fallback on Christianity and Android. This experiment therefore
moves temporal adaptation inside the original CIKM decomposition instead of
adding another score calibrator.

For a graph `A`, DeDiff's one-layer causal path materializes `(A D) X`, where
`D` is the learned dense debiasing matrix and `X` is the user embedding table.
The new implementation uses the algebraically equivalent association

`A (D X)`

and adds a past-conditioned rank-eight correction before graph propagation:

`Z_e = D X + U diag(g(s_e)) V^T X`,

`H_e^c = GCN(A_e Z_e)`,

`H_e^b = GCN(A_e X - A_e Z_e)`.

Here `s_e` contains 28 cumulative-plus-recent statistics computed strictly
before environment `e`. The final layer of `g` is zero initialized. The
original DeDiff checkpoint is frozen and only `U`, `V`, and `g` are trained.
The original source files remain untouched.

The graph products use sparse matrices and never materialize `A D`. A separate
optional SVD path replaces `D` itself by two frozen rank-256 factors. The exact
original anchor remains available as epoch zero and is used if no validation
candidate improves mean MAP@100.

## Correctness audit

Without SVD compression and with the temporal gate at zero, the associative
rewrite has a maximum real-logit difference below `1.5e-6` on both datasets and
100% exact Top-100 row agreement. Unit tests cover zero correction, gradient
flow, dense associative equivalence, sparse/dense graph equivalence, and SVD
factor projection.

No test Dataset, DataLoader, tensor, graph, or model forward is constructed.
Every result below is post-freeze, seed-21, validation-only descriptive
evidence.

## Seed-21 direct DeDiff improvement

| Dataset | Corrected DeDiff | Dynamic DeDiff | Delta MAP@100 | Delta worst MAP@100 | Selected epoch | New trainable parameters | Protected hits K=10/50/100 | Violations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Christianity | 0.093242 | 0.094483 | +0.001241 | -0.000338 | 4 | 27,696 | 7 / 33 / 48 | 0 / 0 / 0 |
| Android | 0.027461 | 0.027683 | +0.000223 | +0.000837 | 2 | 48,112 | 10 / 40 / 62 | 0 / 0 / 0 |

This is the first implemented variant that improves the corrected DeDiff
anchor itself on both tractable datasets. The claim must remain narrow:
Christianity improves mean MAP but slightly reduces worst-period MAP, and only
one seed has been evaluated. Android improves both mean and worst-period MAP.
The deterministic hierarchical union preserves every protected anchor hit.

## Five-seed validation summary

The rank-eight uncompressed configuration was then repeated for seeds 21, 42,
84, 126, and 168 with paired corrected DeDiff anchors.

| Dataset | DeDiff MAP@100 | Dynamic MAP@100 | Mean delta | Positive/negative seeds | p | Delta worst MAP@100 | p worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Christianity | 0.08301 ± 0.01058 | 0.08563 ± 0.01162 | +0.00262 | 4 / 0 | 0.06250 | +0.00327 | 0.12500 |
| Android | 0.02784 ± 0.00156 | 0.02835 ± 0.00107 | +0.00051 | 3 / 0 | 0.12500 | +0.00069 | 0.12500 |

The remaining seeds are exact epoch-zero fallbacks: one on Christianity and
two on Android. Thus neither dataset has a negative final mean-MAP seed.
Christianity worst MAP has three positive, one negative, and one flat seed.
The exact one-sided paired sign-flip tests do not reach 0.05; the correct claim
is consistent non-negative validation selection with positive average effect,
not statistical significance.

Across five seeds, hierarchical fusion protects 23/124/207 Christianity and
49/196/328 Android anchor hits at K=10/50/100 with zero violations.

## Rank-256 static-operator compression

| Dataset | Spectral energy retained | Dense `D` floats | Factor floats | Static reduction | Compressed dynamic MAP@100 | Worst MAP@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Christianity | 96.33% | 2,732,409 | 846,336 | 69.0% | 0.093800 | 0.088175 |
| Android | 98.63% | 8,579,041 | 1,499,648 | 82.5% | 0.027682 | 0.021641 |

Compression is essentially lossless relative to the uncompressed dynamic path
on Android. Christianity remains above the original anchor in mean MAP but
loses more worst-period accuracy and trails the uncompressed dynamic path.
Rank-256 compression is therefore a promising Android/scaling ablation, not the
default method.

## Efficiency result

On one validation batch of 32 cascades, the associative sparse path reduces
incremental peak allocation from 83.4 to 45.9 MiB on Christianity and from
205.7 to 90.7 MiB on Android. It does not provide a reliable small-dataset
latency speedup: measured forward time changes from 16.57 to 17.06 ms and from
18.26 to 20.06 ms, respectively. Sparse kernel overhead dominates at these
sizes. The correct claim is lower temporary memory and removal of the cubic
`A D` construction, not faster end-to-end inference.

The optional SVD factors remove most static `D` storage, but the benchmark
process also holds the original anchor for comparison, so persistent model
memory is reported analytically by factor counts rather than inferred from the
shared-process CUDA peak.

## Decision

Dynamic DeDiff now provides a defensible direct CIKM-to-WWW method lineage:

1. the CIKM decomposition and all learned anchor weights are preserved;
2. temporal environment conditioning is inserted into the debiasing operator;
3. the graph computation is rewritten without `A D` materialization;
4. an exact anchor fallback and protected-hit guarantee remain available.

This passes a five-seed validation development gate but not a significance or
new-test gate. Next steps are a second scalable strong backbone and evaluation
on a newly untouched temporal benchmark. The already consumed test split
cannot be used to choose ranks, epochs, or losses.

Primary artifacts:

- `artifacts/postfreeze_dynamic_dediff/dediff_christian_rank8_seed21.json`
- `artifacts/postfreeze_dynamic_dediff/dediff_android_rank8_seed21.json`
- `artifacts/postfreeze_dynamic_dediff/dediff_christian_base256_temporal8_seed21.json`
- `artifacts/postfreeze_dynamic_dediff/dediff_android_base256_temporal8_seed21.json`
- `artifacts/postfreeze_dynamic_dediff/benchmark_christian_seed21.json`
- `artifacts/postfreeze_dynamic_dediff/benchmark_android_seed21.json`
- `artifacts/postfreeze_dynamic_dediff_summary.json`

# Temporal Strong-Baseline Readiness Report

## Status

This is a **post-freeze, validation-only, single-seed readiness gate**. Twelve
full runs were completed: DyHGCN, MS-HGAT, and DisenIDP on Christianity,
Android, Douban, and Twitter with seed 21. The held-out test partition was not
materialized, evaluated, or used for selection in any run. The aggregate
protocol audit passes.

These numbers are not yet a paper baseline table: they use one seed, a
community integration rather than three author-official repositories, and a
matched Adam training protocol rather than separately tuned original
optimizers. Their purpose is to decide whether the current DriftDiff backbone
is competitive enough to justify five-seed expansion. It is not.

## Reproducible source and adapter

BuzzBloom is pinned at commit
`56db5ddb517d9aeda28be82cab5ebdb482be7897` (MIT license; archive SHA-256
`A92FA3BC1D5C9AE068F17C96DC099CF32C7020542DF60460591BA6432C3FADAF`). It is a
community integration and is not represented as each method author's official
implementation. The pinned source is unchanged under `third_party/buzzbloom`.

`baselines/buzzbloom_temporal.py` replaces the upstream data and runner logic
with:

- timestamp-tie-preserving 70/10/20 chronological splitting;
- train-only diffusion graph/hypergraph construction;
- two contiguous validation environments;
- checkpoint selection by mean validation MAP@100;
- maximum prefix length 50 and metrics over real users only;
- no test Dataset, DataLoader, tensor, forward pass, or selection access.

MS-HGAT's output-equivalent nested Python loops for previous-user masking are
replaced at runtime with a batched scatter. A unit test proves exact entrywise
equality with the unchanged upstream implementation. This reduces a
Christianity 50-step epoch from about three minutes to about nine seconds.

All methods use dimension 64, batch size 64, 50 optimizer steps per epoch,
Adam at `1e-3` with weight decay `1e-4`, at most 10 epochs, at least 5 epochs,
and patience 3. The model-specific defaults from BuzzBloom are retained.

## Seed-21 validation results

The delta columns compare against the existing seed-21 hierarchical DriftDiff
path under its frozen validation evaluator. A positive delta favors the strong
baseline.

| Dataset | Method | Selected epoch | MAP@100 | Worst MAP@100 | Delta MAP | Delta worst |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Christianity | DyHGCN | 3 | 0.08582 | 0.08075 | +0.00254 | +0.00307 |
| Christianity | MS-HGAT | 9 | 0.09479 | 0.07884 | +0.01150 | +0.00116 |
| Christianity | DisenIDP | 6 | **0.09647** | **0.08920** | +0.01319 | +0.01152 |
| Android | DyHGCN | 3 | **0.02859** | 0.01971 | +0.00964 | +0.00489 |
| Android | MS-HGAT | 5 | 0.02825 | 0.02152 | +0.00930 | +0.00670 |
| Android | DisenIDP | 10 | 0.02829 | **0.02269** | +0.00935 | +0.00787 |
| Douban | DyHGCN | 6 | **0.05649** | **0.05445** | +0.01288 | +0.01093 |
| Douban | MS-HGAT | 8 | 0.04565 | 0.04442 | +0.00204 | +0.00089 |
| Douban | DisenIDP | 9 | 0.04601 | 0.04535 | +0.00240 | +0.00182 |
| Twitter | DyHGCN | 10 | **0.19237** | **0.18708** | +0.17866 | +0.17436 |
| Twitter | MS-HGAT | 10 | 0.02893 | 0.02129 | +0.01522 | +0.00857 |
| Twitter | DisenIDP | 10 | 0.12541 | 0.11965 | +0.11170 | +0.10693 |

Every one of the 12 baseline settings beats the existing seed-21 DriftDiff
path in both mean and worst MAP@100. This is too consistent and too large on
Twitter to be handled as a missing comparison table. It invalidates the old
backbone as the primary WWW contribution.

## Interpretation and decision

The exact protected-union result remains useful, but its claim must be
backbone-agnostic. The next method should freeze a competitive anchor such as
DyHGCN or DisenIDP, learn a small past-conditioned logit residual, and apply the
same hierarchical protected union. The necessary claim is then incremental:
the adapter must improve or preserve a strong anchor under natural drift while
retaining zero protected-hit violations.

The follow-up strong-backbone adapter has now passed its seed-21 validation
gate on frozen DyHGCN: three datasets improve and Android selects the exact
anchor fallback. Five-seed expansion is therefore justified for DyHGCN plus
the adapter, but not for the discarded lightweight backbone. See
`STRONG_BACKBONE_ADAPTER_REPORT.md`.

## Limitations before paper reporting

1. Only seed 21 has been run; no uncertainty or significance claim is valid.
2. BuzzBloom is a community integration. Official repositories must be checked
   where available.
3. Adam is matched across the three baselines, not tuned per method; original
   schedule sensitivity remains to be audited.
4. Held-out leakage is prevented, but the community models differ in how they
   use the complete training-period graph. This is not identical to
   DriftDiff's rolling-environment training and must be disclosed or aligned.
5. No new test result is authorized by this readiness gate.

Artifacts:

- `artifacts/postfreeze_temporal_baselines/*.json`
- `artifacts/postfreeze_temporal_baseline_summary_seed21.json`
- `checkpoints/postfreeze_temporal_baselines/*.pt`
- `run_postfreeze_temporal_baselines.py`
- `summarize_temporal_baselines.py`

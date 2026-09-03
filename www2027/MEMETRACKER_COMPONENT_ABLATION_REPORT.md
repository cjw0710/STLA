# MemeTracker Component Ablation Report

## Scope and protocol

This is a post-confirmation, validation-only analysis. It does not reopen or
reuse the sealed MemeTracker test payload, does not change any frozen
checkpoint or one-shot test result, and is not presented as preregistered
confirmatory evidence. All 30 new runs certify `test_materialized=false`,
`test_evaluated=false`, `test_used_for_selection=false`, and
`confirmatory_test_reused=false`.

Each configuration uses the same frozen DyHGCN or DisenIDP anchor and the same
five paired seeds (21, 42, 84, 126, 168). The reported metric is hierarchical
union validation MAP@100, mean ± sample standard deviation.

## Results

| Configuration | DyHGCN | DisenIDP |
|---|---:|---:|
| Frozen anchor | 0.11940 ± 0.00325 | 0.10672 ± 0.00372 |
| Historical candidate feature only | 0.11965 ± 0.00333 | 0.10916 ± 0.00301 |
| No environment context | 0.12222 ± 0.00354 | 0.11156 ± 0.00356 |
| No cascade-prefix context | 0.12242 ± 0.00364 | 0.11129 ± 0.00400 |
| Full STLA | 0.12237 ± 0.00346 | 0.11160 ± 0.00337 |

The historical-only candidate descriptor trails full STLA by 0.00273 on
DyHGCN and 0.00244 on DisenIDP. Full STLA is higher in all five paired seeds
for both backbones (one-sided exact sign-flip p=0.03125). One DyHGCN
historical-only run selects epoch zero, giving the exact anchor fallback.

Environment context supplies a small, consistent DyHGCN increment: full is
higher by 0.00015 in 5/5 seeds (p=0.03125). It is essentially neutral for
DisenIDP (+0.00004 on average, mixed 2/5). Prefix context is likewise modest
and backbone-dependent: removing it changes DyHGCN by +0.00005 and DisenIDP by
-0.00031 relative to full, with mixed paired directions.

The defensible interpretation is therefore narrow: the richer set of temporal
candidate statistics is the main source of the validation gain on MemeTracker;
environment and prefix context make smaller, architecture-dependent changes.
The evidence does not support claiming that every context component is
uniformly necessary.

All configurations retain zero hierarchical-union guarantee violations at
K=10, 50, and 100, as expected from the deterministic construction.

## Artifacts

- Summary: `www2027/artifacts/posttest_validation_ablation_summary.json`
- Per-seed results: `www2027/artifacts/posttest_validation_ablation/`
- Checkpoints: `www2027/checkpoints/posttest_validation_ablation/`
- Runner: `www2027/run_memetracker_adapter_ablations.py`
- Summarizer: `www2027/summarize_memetracker_adapter_ablations.py`


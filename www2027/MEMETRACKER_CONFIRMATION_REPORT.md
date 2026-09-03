# Untouched MemeTracker Confirmation Report

## Status

The confirmatory experiment is complete. MemeTracker was prepared from the
pinned BuzzBloom release, selected without opening its test JSON, hash-frozen,
and evaluated exactly once over ten model-seed pairs. Both frozen backbones
show significant mean and worst-period MAP@100 gains in all five seeds. No
protected anchor hit is displaced at K=10, 50, or 100.

## Deterministic timestamp recovery and split

The released MemeTracker timestamps omit a decimal point after the first six
digits: for example, `3383039575` is `338303.9575`. Restoring this delimiter
makes every retained cascade temporally nondecreasing. The conversion keeps
cascades of length 5--500, remaps released user IDs, and creates a
tie-preserving chronological 70/10/20 split.

| Item | Count |
| --- | ---: |
| Users | 4,709 |
| Directed graph edges | 77,126 |
| Retained cascades | 10,130 |
| Train | 7,091 |
| Validation | 1,013 |
| Sealed test | 2,026 |

The strict selection loader opens only `cascade_train.json` and
`cascade_valid.json`. A unit test deliberately replaces `cascade_test.json`
with invalid JSON and verifies that loader construction still succeeds. The
sealed test hash was recorded during preparation and checked again before
selection was frozen.

## Five-seed validation selection

| Backbone | Anchor MAP@100 | Adapted MAP@100 | Delta | Positive | p | Delta worst | Positive worst | p worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DyHGCN | 0.11940 ± 0.00325 | 0.12237 ± 0.00346 | +0.00298 | 5/5 | 0.03125 | +0.00405 | 5/5 | 0.03125 |
| DisenIDP | 0.10672 ± 0.00372 | 0.11160 ± 0.00337 | +0.00488 | 5/5 | 0.03125 | +0.00556 | 5/5 | 0.03125 |

These results are used only to select the zero-initialized adapter checkpoint
for each already frozen anchor. They are not the confirmatory claim.

## Hash-frozen one-shot test

Before test materialization, one immutable manifest recorded SHA-256 values for
all ten anchor checkpoints, all ten adapter checkpoints, their result JSONs,
the split manifest, the sealed test file, and the evaluator source. The
evaluator verifies every entry before opening the test payload and refuses to
overwrite an existing result.

| Backbone | Anchor MAP@100 | Adapted MAP@100 | Delta | Positive | p | Anchor worst | Adapted worst | Delta worst | p worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DyHGCN | 0.09868 ± 0.00236 | 0.10631 ± 0.00242 | +0.00764 | 5/5 | 0.03125 | 0.09344 ± 0.00164 | 0.10109 ± 0.00206 | +0.00765 | 0.03125 |
| DisenIDP | 0.08846 ± 0.00296 | 0.09781 ± 0.00353 | +0.00935 | 5/5 | 0.03125 | 0.08563 ± 0.00290 | 0.09477 ± 0.00368 | +0.00914 | 0.03125 |

The exact one-sided paired sign-flip test is applied to the five seed-wise
deltas. With five non-zero pairs, the minimum attainable p-value is 0.03125.

## Protected-hit audit

| Cutoff | DyHGCN protected hits | DisenIDP protected hits | Total | Violations |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 764 | 1,026 | 1,790 | 0 |
| 50 | 2,431 | 3,624 | 6,055 | 0 |
| 100 | 4,327 | 6,227 | 10,554 | 0 |

The guarantee is conditional: it preserves protected targets already retrieved
by the corresponding frozen anchor. It does not guarantee retrieval for a
target missed by the anchor, demographic fairness, or preservation of every
unprotected head/recent-active hit.

## Reproducibility artifacts

- `dataset/memetracker/split_manifest.json`
- `artifacts/postfreeze_strong_adapter_memetracker_summary.json`
- `artifacts/memetracker_pretest_selection_manifest.json`
- `artifacts/memetracker_one_shot/*.json`
- `artifacts/memetracker_one_shot_summary.json`
- `prepare_memetracker.py`
- `freeze_memetracker_selection.py`
- `evaluate_memetracker_strong_test.py`
- `run_memetracker_one_shot.py`
- `summarize_memetracker_test.py`

All 63 unit tests passed immediately before the selection manifest was frozen.

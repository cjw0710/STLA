# STLA: Subgroup-Safe Temporal Logit Adaptation

Research code for **STLA: Backbone-Agnostic, Subgroup-Safe Temporal Logit
Adaptation for Web Information Diffusion**. STLA is a lightweight temporal
logit adapter for frozen information-diffusion predictors. It uses past-only
environment and cascade-prefix features, an explicit epoch-zero fallback, and
a deterministic hierarchical protected-union rule at K = 10/50/100.

The repository also retains the original DeDiff implementation used for the
paper's compatibility experiments.

## Repository layout

```text
.
|-- main.py, model.py, module.py   # Original DeDiff implementation
|-- dataLoader.py, graph.py         # Shared data and graph utilities
|-- www2027/
|   |-- models/                     # STLA adapter and backbone wrappers
|   |-- training/                   # Leakage-safe objectives and protocol
|   |-- data/                       # Chronological split/rolling-graph code
|   |-- metrics/                    # Ranking and protected-union evaluation
|   |-- tests/                      # Regression and protocol tests
|   |-- config/                     # Paper-faithful frozen configuration
|   |-- paper/                      # LaTeX manuscript and figure sources
|   `-- README.md                   # Full experiment commands and reports
|-- requirements.txt
`-- REPRODUCTION.md
```

The main STLA implementation is in
`www2027/models/temporal_logit_adapter.py`; training on frozen DyHGCN and
DisenIDP anchors is implemented by `www2027/train_strong_logit_adapter.py`.
The hierarchical safety layer is in `www2027/metrics/ranking.py`.

## Environment

- Python 3.10
- PyTorch 2.0.1
- Remaining pinned dependencies are listed in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

## Tests

Run the regression suite from the repository root:

```bash
python -m unittest discover -s www2027/tests -v
```

## Reproduction

Dataset files and trained checkpoints are intentionally excluded from Git.
Place the prepared datasets under `dataset/`, then follow `REPRODUCTION.md`
and `www2027/README.md` for the smoke test, complete training matrices,
sealed-test protocol, ablations, and paper build instructions.

# Third-party baseline sources

## BuzzBloom

The `buzzbloom/` directory is a pinned snapshot of the community BuzzBloom
integration framework. It is used only as a source of model implementations
for DyHGCN, MS-HGAT, and DisenIDP. It is **not** represented as the original
authors' official implementation of each method.

- Source: <https://github.com/data-science-lab-core/BuzzBloom>
- Pinned commit: `56db5ddb517d9aeda28be82cab5ebdb482be7897`
- Commit date: 2025-10-22
- Downloaded archive SHA-256:
  `A92FA3BC1D5C9AE068F17C96DC099CF32C7020542DF60460591BA6432C3FADAF`
- License: MIT; the upstream `LICENSE` is retained in `buzzbloom/LICENSE`.

The source is kept unchanged. `www2027/baselines/buzzbloom_temporal.py`
provides the local temporal-protocol adapter. In particular, it replaces the
upstream count-only 80/10/10 split and test-coupled runner with:

1. a timestamp-tie-preserving 70/10/20 chronological split;
2. training-only graph and hypergraph construction;
3. two contiguous validation environments selected by mean MAP@100;
4. no test Dataset, DataLoader, forward pass, or selection access;
5. a 50-user prefix cap and metrics over real users only.

The adapter also replaces MS-HGAT's output-equivalent nested Python loops for
the previous-user mask with a batched scatter. A unit test compares every mask
entry with the unchanged upstream method. This is a semantics-preserving
runtime patch, not a model change.

Before a paper submission, numerical claims should also be cross-checked
against each method's official implementation when one is available. The
adapter results are currently post-freeze, validation-only descriptive
baselines and cannot be used to retune the frozen main method.

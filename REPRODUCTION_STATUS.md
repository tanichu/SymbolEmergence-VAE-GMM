# Inter-GMM+VAE reproduction work in progress

Upstream: is0383kk/SymbolEmergence-VAE-GMM, commit
`5401e43cbf6de923a353147dd10a5a7c0513471f`. Original README and license are retained.
This branch is not a completed reproduction or an independently audited release.

Reference: https://arxiv.org/html/2205.12392v2 (version 2).
The published 2023 article is also identified by DOI 10.1080/01691864.2023.2260856;
any differences between versions remain to be checked.

## Initial source alignment

- Algorithm 1: proposals are temporary; rejection retains the listener word.
- Algorithm 2: returned words must be assigned to the listener before its GMM update.
- These two bookkeeping discrepancies are corrected, without replacing the VAE/GMM
  model by a feature-observation model.
- Gaussian proposal normalization and acceptance use log-space arithmetic.
- NumPy is seeded in addition to torch.

## Remaining work (not silently repaired)

- Section 3.2 describes feedback between GMM and VAE; the current outer loop runs
  once and does not return its data-dependent prior arrays. Multi-round feedback,
  VAE checkpoint continuation and covariance approximation need an explicit specification.
- The VAE currently has a diagonal recognition distribution, and the unused feedback
  path keeps only covariance diagonals. A full-covariance prior requires the appropriate
  KL, not an undocumented diagonal replacement.
- Section 4.1 and the current defaults differ in data selection, GMM prior scale,
  update counts and number of mutual-inference rounds. No reported performance has
  been reproduced; do not compare against published scores yet.
- Centralized Gibbs control and held-out evaluation remain to be implemented.
- Existing reported maxima, saved parameters, and terminal private-word metrics need
  consistent readouts. Per-round scientific results have not been generated here.
- Independent audit is required before making publication claims.

## Branch separation

`reproduce-inter-gmm-vae` is for source-aligned corrections and tests.
`cpc-phys-image` starts at the untouched upstream commit and is reserved for the
new study. No unpublished manuscript, Vault notes or current-study code is included
in this public fork by this initial correction.

## Decoder limitation

The research author reports that this model's decoder did not work adequately on
general natural images, motivating MNIST and Fruits360 in earlier work. This is
author-provided history, not a new benchmark result. MNIST is the initial integration
target. Natural-image reconstruction requires a separate feasibility evaluation;
successful naming must not be inferred from decoder quality, or vice versa.

## Tests

`python -m pytest tests/test_word_updates.py -q`
Synthetic tests cover temporary proposals, rejection, listener-state persistence,
the local no-communication update, log-space ratios and Gaussian probabilities.
They do not validate image learning, full mutual inference or convergence.

2026-10-24 local result: 7 passed (10.96 s), Python 3.13, NumPy 2.4.0,
SciPy 1.16.3, torch 2.6.0+cu124. The single-object integration fixture emits
two sklearn warnings because kappa is undefined for its single constant label;
no kappa value is asserted or treated as a scientific result. The process exited
0; pyreadline also printed a console destructor error during interpreter shutdown.
No image training or MNIST download was run.

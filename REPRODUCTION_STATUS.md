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

## Mutual-inference repair (2026-10-24)

The GMM now supplies each agent's own word-conditioned mean and **full covariance**
to the next VAE round. The loss uses analytic KL from diagonal recognition Gaussian
to full prior Gaussian, with a batch sum in both initial and feedback rounds.
VAE weights/Adam state and GMM/private-word state continue rather than restart.
Extraction is batched and order-checked. MODEL.md records the blockwise approximation
and Adam-state choice; neither is claimed to recover undocumented historical details.
Use --mutual-iterations 1 to disable subsequent feedback, or a larger value to test it.

## Remaining work (not silently repaired)

- Full-prior transfer, checkpoint continuation and numerical KL tests are now
  implemented; a full performance reproduction and independent audit remain pending.
- Section 4.1 and the current defaults differ in data selection, GMM prior scale,
  update counts and number of mutual-inference rounds. No reported performance has
  been reproduced; do not compare against published scores yet.
- Centralized Gibbs and a named reference-scale profile are now implemented;
  full trials and held-out evaluation remain pending. See EVAL.md.
- Existing reported maxima, saved parameters, and terminal private-word metrics need
  consistent readouts. Per-round scientific results have not been generated here.
- Independent audit is required before making publication claims.

### Update: preliminary MNIST comparison completed, 2026-10-24

The historical pending statements above describe earlier stages. The repaired,
cached implementation completed four methods on seeds 0..4, using 10,000 balanced
images, 100 VAE epochs and 100 naming sweeps per block, five blocks, batch 128,
prior scale .05. Mean +/- sample SD of terminal outcomes (not training maxima):

| Method | ARI A | ARI B | Kappa |
| --- | --- | --- | --- |
| Central Gibbs | .755 +/- .078 | .755 +/- .078 | N/A |
| MHNG | .774 +/- .057 | .774 +/- .058 | .971 +/- .008 |
| All-accept | .627 +/- .015 | .618 +/- .007 | .770 +/- .012 |
| No-comm. | .556 +/- .064 | .628 +/- .038 | .001 +/- .043 |

MHNG exceeded both baselines in agent-averaged ARI on all five seeds, and word
agreement followed MHNG > All-accept > No-comm. Central Gibbs and baseline ARI
were lower than the original Table 1, and the mean MHNG/Gibbs ranking reversed.
This is preliminary qualitative agreement, not exact numerical reproduction,
an equivalence test, or an independently audited release. No-MI benefit is untested.

Thirty tests passed on the CUDA experiment snapshot, with two pre-existing
single-label fixture warnings. Optional caches preserve the model/update budget;
round-off differences in stochastic trajectories remain possible. CUDA settings
used deterministic algorithms, deterministic cuDNN, disabled benchmark and TF32,
and CUBLAS_WORKSPACE_CONFIG=:4096:8. AMD/ROCm smoke runs had small internal MHNG
differences despite identical evaluation metrics; flags alone do not certify
bitwise reproducibility across executions or devices.

`python deterministic_entry.py --paper-mnist --mode -1 --seed 0 --cache-images
--cache-prior --gpu-images --epoch-sync --run-dir NEW_RUN` applies those settings.
Raw datasets/checkpoints are not included in Git.

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

That paragraph describes the initial patch only. Subsequent verification on
2026-10-24: all 12 synthetic/integration tests passed, with the same two warnings
from the old single-label word fixture. A real MNIST smoke run (256 images, seed 2,
2 outer rounds, 1 VAE epoch and 2 MH sweeps per round, batch 32) completed on CUDA.
This checks execution and feedback, not categorization performance. No published
score is claimed. Data and generated models are kept outside the repository.

Example smoke command:

```
python main.py --samples 256 --batch-size 32 --vae-iter 1 --mh-iter 2 --mutual-iterations 2 --seed 2 --run-dir PATH_TO_NEW_RUN
python check_feedback_run.py PATH_TO_NEW_RUN
```

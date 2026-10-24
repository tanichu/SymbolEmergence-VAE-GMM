# Development decisions (append only)

2026-10-24: Use log-space Gaussian arithmetic without clipping small positive
acceptance probabilities. Non-finite log ratios raise rather than silently reject.
Remove the determinant epsilon from normalized Gaussian proposal probabilities;
use slogdet and retain the existing precision matrices without a covariance floor.
The finite well-conditioned case is checked against direct Gaussian densities.
No scientific hyperparameter or VAE prior is changed in this first patch.

2026-10-24: Keep --mode 0 as an independent local Gibbs word update. The speaker
state is updated only in that no-communication path; communicating proposals are
temporary. GMM/VAE feedback is explicitly still incomplete.

2026-10-24 (mutual-inference repair): Full prior covariances are now transferred
and used in a closed-form KL with diagonal recognition covariance. The image
recognition family is unchanged. Each agent's own private terminal words index
its priors. Continue GMM/word state and VAE/Adam state across blocks. Adam-state
continuation is not specified by the reference; it is an explicit implementation
choice, not a claimed recovered historical setting. VAE initially uses N(0,I).
Use fixed sequential image order, reject shuffled/dropped loaders, extract in
multiple batches under no_grad. No covariance jitter or fallback is introduced.
Small MNIST runs check plumbing, not the full published scores or convergence.

## 2026-10-24: direction-local MH likelihood cache

During one speaking direction, listener parameters and observations are fixed.
Cache D×K Gaussian log densities using precision matrices directly, omitting only
the category-independent dimension constant. Rebuild on every directional call;
never reuse across a GMM update. Keep proposal/uniform RNG calls in the original
order, including uniforms for identical proposal/current labels. No clipping,
jitter, precision reduction, or acceptance-rule change. Round-off can differ
from scipy's inverse-covariance path; long-chain bitwise identity is not promised.
The unchanged legacy kernel is retained for development differential checks.

## 2026-10-24: opt-in image and VAE prior caches

`--cache-images` materializes exactly the existing fixed-rotation transform once,
preserving CPU torch RNG during construction and consuming the original angle
draw on each retrieval. Sequential sample order and pixel values are unchanged;
this cache is not for stochastic augmentation. `--cache-prior` prepares FP64
precision and logdet once per agent/outer training block, with gradients retained
through recognition mean/logvar. No batch size, model, loss weight, or training
budget changes. Both options default off; no running experiment is migrated.

## 2026-10-24: opt-in GPU residency and epoch synchronization

`--gpu-images` requires the fixed image cache, sequential complete single-worker
loader, and CUDA. Its batch iterator creates the original loader iterator to
preserve its base-seed draw, then reproduces fixed-rotation CPU RNG draws in the
original order; images are sliced from one resident tensor. Labels remain CPU.
`--epoch-sync` transfers detached scalar losses once per epoch and sums the
Python floats in the original order (not a changed GPU reduction). Non-finite
losses still fail the run before checkpoint saving, but detection is delayed
until the epoch boundary: intermediate optimizer state may be invalid and must
be discarded on failure. This option changes failure-detection latency, not the
finite-data objective. Both options default off. No running seed is migrated.

## 2026-10-24: validation precision clarified by author

Bitwise equality is not a required scientific acceptance criterion. Preserve
model/update semantics, check objective/gradient numerical differences and actual
learning behavior. Existing exact-equality checks are useful controlled diagnostic
results, not a requirement to force long stochastic trajectories to match.
Do not tune inference to make final outcomes numerically identical. Future
numerical tolerances must be stated before judging the corresponding test.

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

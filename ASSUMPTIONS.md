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

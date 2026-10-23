# Mutual-inference repair: development specification

Reference: arXiv:2205.12392v2, section 3.1 equations 1--4, section 3.2,
Algorithm 1/2 and Appendix A.2. This is a working implementation, not certified
reproduction of published scores or a claim of exact Gibbs sampling of the VAE.

## Model and objective

Uniform categorical word, agent-specific normal-Wishart Gaussian codebooks,
and a private neural image decoder. The recognition distribution is diagonal
Gaussian; the GMM prior is full Gaussian. Train each VAE with pixel BCE plus
KL(q_diag(z|image) || N(mu_private_word, inverse(precision_private_word))).
The KL uses the full prior precision, log determinant and quadratic term; do
not replace a full prior by its covariance diagonal. Initial VAE pretraining
uses a standard normal prior. Labels are used only by existing evaluation.

## Mutual inference

Initial independent VAE training -> draw local latents -> initialize GMM once ->
MH/GMM sweeps -> gather each agent's own terminal word-conditioned means and
full covariances -> continue that agent's VAE -> redraw local latents -> continue
GMM/MH. Repeat. Latents are frozen during an MH block: this is a blockwise neural
approximation to the generative model, not every line of Algorithm 2 performed
as exact conditional sampling after every utterance.

Each feedback array is indexed in the same stable dataset order as the private
word array. Training and extraction loaders must be sequential and not drop
samples. An unknown order fails rather than silently mixing images and priors.
No cross-agent latent, gradient, or codebook is used to construct a local prior.

## Constants and choices

- Network: unchanged MNIST architecture, latent dimension 12, image 1x28x28.
- VAE Adam learning rate 0.001, unchanged from upstream. Loss is a batch sum.
- Default VAE/MH epochs stay at upstream 50/50 (not paper's full reproduction
  budget). Default outer rounds 5, matching the number of MI rounds in section 4.1.
- Default GMM prior stays beta=1, m=0, W=0.1 I, nu=dimension. It is not silently
  changed to the paper's reported scale. This discrepancy remains open.
- Continue VAE weights AND Adam state across rounds. The reference does not
  specify Adam-state handling; this is an explicit continuation choice.
- Continue GMM parameters and private words across outer rounds; no refitting
  to the other agent's labels. Initial pretraining is independent.
- Smoke run: 256 sequential MNIST training images, seed 2, batch 32, 1 VAE epoch,
  2 MH sweeps per outer round, 2 outer rounds. Not a performance experiment.
- Artifact destinations are run-specific; no overwrite of an existing run.
- Full Gaussian KL uses float64 arithmetic; gradients flow to recognition mean
  and log variance. No covariance clipping, jitter or failure-to-value conversion.

## Verification

Compare KL with torch.distributions for non-diagonal priors and standard normals;
check gradient finiteness, correct agent/word indexing, sequential-loader contract,
multi-batch extraction, and checkpoint continuation. The MNIST smoke must save
used-prior arrays and source hashes and show that round 1 actually consumes round
0 GMM feedback. Original reported ARI values are not test expectations.

2026-10-24: first specification for full-prior feedback and persistent MI state.

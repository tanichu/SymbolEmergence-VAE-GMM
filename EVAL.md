# MNIST qualitative replication plan (2026-10-24)

Development study; not a frozen publication claim. No human rh lock has been
created. Independent audit and a completed experiment are required before claiming
replication. The current study's unpublished manuscript is outside this repository.

## Scope and protocol

Reference is arXiv:2205.12392v2 sections 3.2 and 4.1. The named --paper-mnist profile
uses 10,000 training images, 1,000 per digit, 0/45-degree observations, latent
dimension 12, uniform 10-word prior, normal-Wishart scale .05 I, mean precision 1,
degrees of freedom 12, 100 VAE epochs and 100 MH/GMM rounds per outer iteration,
5 mutual-inference rounds. Batch size remains configurable (default 128), not
silently optimized after looking at test labels. Data selection takes the first
1,000 indices of each digit, then sorts indices; this historical index choice is
not claimed to be exactly the original dataset subset.

Methods: mode -1 MHNG, 0 independent local Gibbs, 1 always accept, 2 centralized
Gibbs from the likelihood product. All use the same model/prior and local update
counts. Central has a common word, so kappa is not a meaningful communication
metric and is stored as NaN. Central does not use test labels to sample words.

Primary target: 16 seeds, numbered 0..15, all four methods. Independent VAE
initialization, no label alignment. No seed-specific restart/early stopping or
best-seed selection. Report paired seed differences and full ranges, not only
published-score proximity. Report terminal ARI (agent-averaged), raw kappa for
distributed methods and complete saved learning curves; do not use maxima over
training as the main result. Original raw ARI is invariant to label permutation;
supervised display relabeling is not used to compute it.

Secondary comparison: --disable-mi runs one initial VAE/GMM block. This reproduces
the operational no-feedback setting but has less total training than the five-block
setting. Any difference is not a pure budget-matched effect of prior feedback.
Run the four conditions without feedback on the same 16 seeds after the main
comparison, subject to pilot cost and audit. A matched-budget frozen-VAE control
would answer a different question and requires a separately stated protocol.

## Qualitative interpretation

Compare categorization, agreement and the effect of additional mutual inference.
Report reversals or absent effects as failures of qualitative replication rather
than modifying code or selecting seeds to force the reference's ordering. Similar
means alone do not establish equivalence to centralized inference. A single pilot
is only an existence/smoke check. No target numeric scores are embedded in tests.

## Scheduling and provenance

First a balanced 1,000-image, two-block, 2-VAE-epoch/5-GMM-round pilot on kinkaku;
record elapsed time, device, source hashes and completion. Verify full-prior
transfer using check_feedback_run.py. Then estimate the full matrix before
launching the long jobs. Checkpoint and source data stay in the dedicated run
directory, not public GitHub. Queue time and environment setup are separate.

Independent audit: isolated code-only reconstruction, fresh different-family
auditor, separate false-positive review, verify quoted lines and tests. Full-run
results will need the same audit claims revisited after results exist.

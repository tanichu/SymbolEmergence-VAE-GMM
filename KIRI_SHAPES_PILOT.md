# Five-shape, single-object multi-view feasibility pilot

Author-approved replacement for the initial three-category/multi-instance pilot.
No requirement for five independent specimens per category at this stage.

| Category | Scan ID | Shape cue |
| --- | --- | --- |
| Banana | A143 | Single curved elongated fruit, not a bunch |
| Pear | A112 | Narrow neck / wider base |
| Apple | A34 | Rounded with stem indentation |
| Dragon fruit | A107 | Oval with projecting bracts |
| Watermelon | A98 | Elongated ellipsoid, striped surface |

Use Mid LOD, source albedo and the existing renderer's common neutral background,
lighting, isotropic centering/rescaling, RGB128 output and RGB64 input. Keep color
and texture: shape is varied by object choice, not isolated as the sole cue.
Known K=5, latent12, identical RGB VAE/GMM/prior across methods. One scan per
category means category identity is confounded with object identity. This is
same-object multi-view naming, not category generalization to new specimens.

Each scan: 36 azimuths (0,10,...,350) x elevations10/25/40 =108 pairs.
A sees base pose, B sees azimuth+60/elevation+15. Five scans yield540 pairs,
1080 RGB images. Split by base azimuth index modulo6: remainder0 validation,
remainder3 test, others training. Thus360 train,90 validation,90 test pairs.
Same scan can appear across splits; no identical camera pose may cross splits,
including between agent A and B. The60-degree offset preserves the azimuth
split groups. This tests unseen views, never unseen physical instances.

Fixed one-seed0 comparison: Central Gibbs/MHNG/All-accept/No-comm., batch32,
VAE50 epochs +30 naming sweeps per block,3 blocks. Increase from the earlier
10/10x2 smoke is specified before outcomes, to permit more image learning;
it is not a promised convergence budget. Adam .001, normal-Wishart scale .05,
remaining model/prior/update semantics unchanged. No score-selected retries.
All methods retain private independent VAE initialization. Save all results.

Main diagnostics: real input/reconstruction, losses, terminal training ARI,
private-word usage and kappa (undefined for identical constant words, not1).
Held-out view readout and manuscript F_d remain separate explicit work; do not
use validation/test in VAE or GMM training. Reproduction branch is unchanged.

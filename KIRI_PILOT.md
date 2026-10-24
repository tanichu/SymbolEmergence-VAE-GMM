# KIRI multi-view fruit pilot (development)

This is a feasibility experiment, not the manuscript's final demonstration.
Reuse the repaired Inter-GMM+VAE word/GMM updates and full Gaussian feedback.
Keep this work on `cpc-phys-image`, separate from the MNIST reproduction branch.

## Data protocol

Three filename categories, visually screened for obvious cuts/mislabels:

| Category | Train scan IDs | Validation | Test |
| --- | --- | --- | --- |
| Apple | A34, A59, A73 | A81 | A99 |
| Orange | A49, A61, A62 | A64 | A67 |
| Mango | A29, A57, A114 | A115 | A130 |

These are distinct scan IDs, not certified independent physical fruit specimens.
All LODs of a scan belong to one split; use Mid only. Do not use cut/peeled
variants as extra independent specimens. No data or original assets go into Git.

Each scan has 12 azimuths (0,30,...,330 degrees), two base elevations (20,40).
Agent A sees that pose, B sees azimuth+60/elevation+15 degrees of the same scan.
Thus 24 paired observations per scan: 216 train pairs, 72 validation, 72 test.
Render RGB 128x128, then resize to RGB 64x64 for training. Normalize each object
isotropically and center it; fixed neutral background/lighting across categories.
No labels, category-specific backgrounds or marks enter the learning objective.
Category-balanced sampling and known vocabulary K=3 are explicit assumptions.

## Model and budget

Private CNN VAEs: RGB64, latent dimension12, four stride-two encoder stages and
mirrored decoder; diagonal recognition, standard-normal first block, then each
agent's private full Gaussian word-conditioned prior. Pixel BCE + full Gaussian
KL, batch sum, Adam .001. BCE on continuous rendered pixels is a reconstruction
surrogate, not a calibrated physical image likelihood or manuscript F_d.

Encoder channels are 3->32->64->128->256 with kernel4/stride2/padding1 and ReLU,
yielding 256x4x4. Linear diagonal-Gaussian heads have dimension12. The decoder
reverses this channel schedule from the linear 4096 features, ending in sigmoid.
No pretrained weights, dropout or normalization layers are added.

One fixed seed0, four methods, VAE10 epochs / naming10 sweeps per block, two
blocks, batch32, normal-Wishart scale .05 and the existing remaining constants.
No retries selected by ARI. Settings are chosen for short execution, not claimed
convergence. Compare reconstruction and finite losses before interpreting naming.
Validation/test remain out of training. A future held-out naming readout must be
specified separately; the first run's private-word ARI is training-only.

The current pilot covers Same object only. Same category and Random pairing,
full five-seed evaluation and empirical joint-prior F_d remain separate work.
No replacement of images with thumbnails, fake reconstructions or pretrained
supervised representations is permitted.

## License review

Official pack page declares CC0 and asks not to redistribute/sell the pack as-is:
https://www.kiriengine.app/3d-tools/fruit-and-veg-asset-pack-by-kiri-engine
The bundled README contains Scatter folder instructions, not an extra license.
The public EULA describes the application, not a replacement asset license.
Respect the pack's non-redistribution request: publish scripts/manifests only,
keep source assets private and credit KIRI in research reports voluntarily.
CC0 does not clear third-party marks or offer ownership warranties; avoid stickers.

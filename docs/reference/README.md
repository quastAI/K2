# Reference scripts from AAC (2026-11-17/18)

These are throwaway analysis scripts from the AAC session, kept for their method. They read AAC's pixel checkpoints and import AAC modules, so they do not run in K2 and are not part of the package. Rewrite what a step needs; do not import from here.

- `predbasis.py`: the lag-1 canonical (lagged-CCA) basis of `(z_t, z_{t+1})`, with the whitened latent floored at an eigenvalue of 1e-2. Measures how many directions a factor needs (k90) in this basis against PCA and a random rotation. This is the method behind E5.
- `changebasis.py`: the spectrum of the change. Per canonical direction, the predictor's error over the change (1 = copying), and where each factor's change sits.
- `decodedtiles.py`: the decoder as a probe. Classifies decoded map tiles by the nearest mean real sprite per block type. This found that probes undersell rare classes.
- `unevenness.py`: how heavy-tailed the gate's error per patch is (the share of the total in the worst 1% and 5%; the mean gate under a batch-mean scale against a median scale).

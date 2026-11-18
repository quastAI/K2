import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/Users/julian.quast/Documents/Kailash/AAC/H-JEPA/h_jepa')
sys.path.insert(0, '/Users/julian.quast/Documents/Kailash/AAC/H-JEPA')
from sieve import check, sample as sm, gate_analysis as ga
from sieve.watch import load_model

D = Path('/Users/julian.quast/Documents/Kailash/AAC/data')
R = D / 'runs/V4_seed42/ckpts/craftax'
part = sm._read(D / 'hf/craftax-classic-pqn-rnn-v1/data/craftax_expert_val.h5', 64, 4, 32, 1)
t, t1 = part.pairs()
lab = ga.labels(part, t, t1)
stratum, kind = lab['stratum'], lab['kind']
rng = np.random.default_rng(0)
for label, path in [('V4 stage-1 epoch 8', R / 'V4/seed42/V4_epoch_8_object.ckpt'),
                    ('V4 stage-2 epoch 4', R / 'V4_stage2_from8/seed42/V4_stage2_from8_epoch_4_object.ckpt')]:
    model = load_model(path).eval(); jepa = model.get_level(1)
    z = check.encode(jepa, part.pixels, 'cpu')
    p = check.predict(jepa, z[t], part.action[t], 'cpu')
    g = ga.decoded_difference(jepa.decoder, p, z[t1], 'cpu')          # (pairs, 81): the diff error per patch
    flat = g.ravel()
    m = flat.mean()
    r = flat / m
    G = 0.1 + 0.9 * np.exp(-r)
    srt = np.sort(flat)[::-1]
    mass = srt.cumsum() / srt.sum()
    n = len(flat)
    print(f'\n=== {label}: {n} patches, mean g {m:.2e}, median g / mean {np.median(r):.2f}')
    print(f'share of the total error carried by the worst 1% / 5% / 20% of patches: {mass[int(.01*n)]:.2f} / {mass[int(.05*n)]:.2f} / {mass[int(.2*n)]:.2f}')
    print('patches by g / mean:  <0.1: %.2f   0.1-0.5: %.2f   0.5-1: %.2f   1-3: %.2f   >3: %.2f' % tuple(
        ((r >= lo) & (r < hi)).mean() for lo, hi in [(0, .1), (.1, .5), (.5, 1), (1, 3), (3, 1e9)]))
    print('gate:  mean %.2f   share >0.8: %.2f   0.5-0.8: %.2f   0.2-0.5: %.2f   <0.2: %.2f' % (
        G.mean(), (G > .8).mean(), ((G > .5) & (G <= .8)).mean(), ((G > .2) & (G <= .5)).mean(), (G <= .2).mean()))
    # what a robust scale (the median) would do instead of the mean
    G2 = 0.1 + 0.9 * np.exp(-flat / (2 * np.median(flat)))
    print('with the median in place of the mean (scale 2 x median): mean gate %.2f   share <0.2: %.2f' % (G2.mean(), (G2 <= .2).mean()))
    names = ga.STRATUM_NAMES
    print('who carries the worst 5%:', end=' ')
    top = flat >= srt[int(.05 * n)]
    sel = stratum.ravel()[top]
    cnt = np.bincount(sel, minlength=len(names))
    order = np.argsort(-cnt)[:6]
    print(', '.join(f'{names[i]} {cnt[i]/top.sum():.2f}' for i in order))
    kk = kind.ravel()[top & (stratum.ravel() < ga.NUM_STRATA)]
    print('  of those on map tiles: entering %.2f, changed %.2f, unchanged %.2f' % tuple((kk == k).mean() for k in range(3)))

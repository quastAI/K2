import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, '/Users/julian.quast/Documents/Kailash/AAC/H-JEPA/h_jepa')
sys.path.insert(0, '/Users/julian.quast/Documents/Kailash/AAC/H-JEPA')
from sieve import concentration as cn, check, sample as sm
from sieve.watch import load_model

D = Path('/Users/julian.quast/Documents/Kailash/AAC/data')
R = D / 'runs/V4_seed42/ckpts/craftax'
EPS = 1e-2
BINS = [(-9.0, 0.3, 'foreseen <0.3'), (0.3, 0.6, '0.3-0.6'), (0.6, 0.9, '0.6-0.9'), (0.9, 99.0, '>=0.9 (copy-level)')]

part = sm._read(D / 'hf/craftax-classic-pqn-rnn-v1/data/craftax_expert_val.h5', 128, 4, 32, 1)
y, names = cn.factor_targets(part)
t, t1 = part.pairs()
frame_train, _ = cn.split_by_episode(part)
tr = frame_train[t]; te = ~tr


def sym_pow(m, power):
    vals, vecs = np.linalg.eigh(m)
    return (vecs * (vals + EPS * vals.mean()) ** power) @ vecs.T


for label, path in [('stage1_epoch8', R / 'V4/seed42/V4_epoch_8_object.ckpt'),
                    ('stage2_epoch4', R / 'V4_stage2_from8/seed42/V4_stage2_from8_epoch_4_object.ckpt')]:
    model = load_model(path).eval(); jepa = model.get_level(1)
    z = check.encode(jepa, part.pixels, 'cpu').astype(np.float64)
    p = check.predict(jepa, z[t].astype(np.float32), part.action[t], 'cpu').astype(np.float64)
    z0, z1, yy = z[t], z[t1], y[t1]
    delta, err = z1 - z0, z1 - p
    n = tr.sum()
    Md = delta[tr].T @ delta[tr] / n
    Me = err[tr].T @ err[tr] / n
    Wd, Wd_inv = sym_pow(Md, -0.5), sym_pow(Md, 0.5)
    lam, Q = np.linalg.eigh(Wd @ Me @ Wd)             # lam: error over change, per whitened direction (1 = copying)
    print(f'\n=== {label}: overall error over copy {np.trace(Me)/np.trace(Md):.2f}')
    print('directions of the change by predictability (error / change): ' + '   '.join(f'{b[2]}: {int(((lam>=b[0])&(lam<b[1])).sum())}' for b in BINS))
    # linear read of every factor from the latent (levels), fitted on train pairs
    mu = z1[tr].mean(0)
    a = z1[tr] - mu
    gram = a.T @ a
    wr = np.linalg.solve(gram + cn.LAM * np.trace(gram) / 192 * np.eye(192), a.T @ (yy[tr] - yy[tr].mean(0)))
    pred_te = (z1[te] - mu) @ wr
    r2 = 1 - ((pred_te - (yy[te] - yy[tr].mean(0))) ** 2).sum(0) / ((yy[te] - yy[te].mean(0)) ** 2).sum(0)
    wt = Q.T @ Wd_inv @ wr                              # the read, in the whitened change directions
    print(f"{'factor':24}{'R2':>5}{'change foreseen':>16}{'(direct)':>9}   share of the factor's CHANGE variance on directions with error/change " + ' / '.join(b[2] for b in BINS))
    rows = []
    for j, nm in enumerate(names):
        if r2[j] < cn.HELD: continue
        e2 = wt[:, j] ** 2
        shares = [e2[(lam >= lo) & (lam < hi)].sum() / e2.sum() for lo, hi, _ in BINS]
        via = 1 - (e2 * lam).sum() / e2.sum()
        d_read, e_read = delta[te] @ wr[:, j], err[te] @ wr[:, j]
        direct = 1 - (e_read ** 2).mean() / max((d_read ** 2).mean(), 1e-12)
        rows.append((nm, r2[j], via, direct, shares))
    for nm, r, via, direct, sh in sorted(rows, key=lambda r: -r[2]):
        print(f'{nm:24}{r:5.2f}{via:16.2f}{direct:9.2f}   ' + ' / '.join(f'{s:.2f}' for s in sh))

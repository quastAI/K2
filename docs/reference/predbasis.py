import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, '/Users/julian.quast/Documents/Kailash/AAC/H-JEPA/h_jepa')
sys.path.insert(0, '/Users/julian.quast/Documents/Kailash/AAC/H-JEPA')
from sieve import concentration as cn, check, sample as sm
from sieve.watch import load_model

D = Path('/Users/julian.quast/Documents/Kailash/AAC/data')
R = D / 'runs/V4_seed42/ckpts/craftax'
EPS = 1e-2          # eigenvalue floor of the whitening, in units of the mean eigenvalue
BINS = [(0.9, 9.0, '>=0.9'), (0.5, 0.9, '0.5-0.9'), (0.2, 0.5, '0.2-0.5'), (-9.0, 0.2, '<0.2')]

part = sm._read(D / 'hf/craftax-classic-pqn-rnn-v1/data/craftax_expert_val.h5', 128, 4, 32, 1)
y, names = cn.factor_targets(part)
t, t1 = part.pairs()
frame_train, _ = cn.split_by_episode(part)
tr = frame_train[t]; te = ~tr


def whitener(cov):
    vals, vecs = np.linalg.eigh(cov)
    return (vecs * (vals + EPS * vals.mean()) ** -0.5) @ vecs.T, vecs[:, ::-1]


def shares(zw_tr, y_tr, basis, r2dir):
    a = (zw_tr - zw_tr.mean(0)) @ basis
    ya = y_tr - y_tr.mean(0)
    gram = a.T @ a
    w = np.linalg.solve(gram + cn.LAM * np.trace(gram) / len(gram) * np.eye(len(gram)), a.T @ ya)
    contrib = w ** 2 * (np.diag(gram) / len(a))[:, None]
    out = np.zeros((len(BINS), y_tr.shape[1]))
    for i, (lo, hi, _) in enumerate(BINS):
        m = (r2dir >= lo) & (r2dir < hi)
        out[i] = contrib[m].sum(0) / np.maximum(contrib.sum(0), 1e-30)
    return out, w


for label, path in [('stage1_epoch8', R / 'V4/seed42/V4_epoch_8_object.ckpt'),
                    ('stage2_epoch4', R / 'V4_stage2_from8/seed42/V4_stage2_from8_epoch_4_object.ckpt')]:
    t0 = time.time()
    model = load_model(path).eval()
    jepa = model.get_level(1)
    z = check.encode(jepa, part.pixels, 'cpu').astype(np.float64)
    p = check.predict(jepa, z[t].astype(np.float32), part.action[t], 'cpu').astype(np.float64)
    z0, z1, yy = z[t], z[t1], y[t1]
    mu0, mu1 = z0[tr].mean(0), z1[tr].mean(0)
    W0, _ = whitener(np.cov(z0[tr], rowvar=False))
    W1, Vp = whitener(np.cov(z1[tr], rowvar=False))
    # (a) lagged CCA, no action
    cross = (z0[tr] - mu0).T @ (z1[tr] - mu1) / tr.sum()
    _, rho, vt = np.linalg.svd(W0 @ cross @ W1)
    R_cca, r2_cca = vt.T, rho ** 2
    # (b) the trained predictor's error spectrum in the target's whitened space
    err = (z1 - p)[tr]
    s = W1 @ (err.T @ err / len(err)) @ W1
    lam, Q = np.linalg.eigh(s)
    R_model, r2_model = Q, 1 - lam                      # most predictable first
    zw = (z1 - mu1) @ W1
    pw = (p - mu1) @ W1
    print(f'\n=== {label}  ({time.time()-t0:.0f} s)  pairs {tr.sum()} fit / {te.sum()} test')
    for nm, r2d in [('cca (no action)', r2_cca), ('model (trained predictor)', r2_model)]:
        print(f'{nm:28} directions with predictability >=0.9: {int((r2d>=0.9).sum()):3d}   0.5-0.9: {int(((r2d>=0.5)&(r2d<0.9)).sum()):3d}   0.2-0.5: {int(((r2d>=0.2)&(r2d<0.5)).sum()):3d}   <0.2: {int((r2d<0.2).sum()):3d}')
    # k90 in four bases of the same whitened space, held-out R^2
    rng = np.random.default_rng(0)
    rands = []
    for _ in range(3):
        q, r = np.linalg.qr(rng.standard_normal((192, 192))); rands.append(q * np.sign(np.diag(r)))
    bases = {'model': R_model, 'cca': R_cca, 'pca': Vp, 'rand': rands}
    k90 = {}
    for nm, b in bases.items():
        cs = [cn.curves(zw[tr], yy[tr], zw[te], yy[te], bb) for bb in (b if isinstance(b, list) else [b])]
        k90[nm] = np.nanmedian(np.stack([cn.axes_needed(c['r2']) for c in cs]), 0)
        r2full = cs[0]['r2'][-1]
    sh, wfull = shares(zw[tr], yy[tr], R_model, r2_model)
    # direct check: the linear read of the predicted latent against the read of the true latent, on the test pairs
    a_te = (zw[te] - zw[tr].mean(0)) @ R_model
    p_te = (pw[te] - zw[tr].mean(0)) @ R_model
    ya = yy[tr] - yy[tr].mean(0)
    wf = wfull
    read_true, read_pred = a_te @ wf, p_te @ wf
    pred_direct = 1 - ((read_true - read_pred) ** 2).mean(0) / np.maximum(read_true.var(0), 1e-12)
    held = [j for j, nme in enumerate(names) if r2full[j] >= cn.HELD]
    print(f"\n{'factor':24}{'R2':>5}{'foreseen':>9}   share of the factor's weight on directions of predictability {'/'.join(b[2] for b in BINS)}   k90: model  cca  pca  rand")
    for j in sorted(held, key=lambda j: -r2full[j]):
        print(f"{names[j]:24}{r2full[j]:5.2f}{pred_direct[j]:9.2f}   " + ' / '.join(f'{sh[i, j]:.2f}' for i in range(len(BINS))) + f"      {k90['model'][j]:5.0f}{k90['cca'][j]:5.0f}{k90['pca'][j]:5.0f}{k90['rand'][j]:5.0f}")
    summ = {nm: float(np.median([k90[nm][j] for j in held])) for nm in k90}
    print('\nmedian k90 over the held factors:', {k: round(v) for k, v in summ.items()})

import sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, '/Users/julian.quast/Documents/Kailash/AAC/H-JEPA/h_jepa')
sys.path.insert(0, '/Users/julian.quast/Documents/Kailash/AAC/H-JEPA')
from sieve import check, sample as sm
from sieve.decoder import patchify, uint8_to_unit
from sieve.watch import load_model
from probing import targets as tg

t0 = time.time()
D = Path('/Users/julian.quast/Documents/Kailash/AAC/data')
R = D / 'runs/V4_seed42/ckpts/craftax'
part = sm._read(D / 'hf/craftax-classic-pqn-rnn-v1/data/craftax_expert_val.h5', 32, 4, 32, 1)
t, t1 = part.pairs()
blocks, mobs = tg.tile_maps(torch.from_numpy(part.state))
blocks, mobs = blocks.numpy(), mobs.numpy()
PLAYER = tg.PLAYER_TILE[0] * tg.VIEW_COLS + tg.PLAYER_TILE[1]
bright = part.daylight > 0.9
clean = ~mobs.any(-1)                                           # no mob on the tile
clean[:, PLAYER] = False                                        # not the player's tile
true = patchify(uint8_to_unit(part.pixels)).numpy()[:, :63]     # (N, 63, 147): the real tile crops
# prototypes: the mean real crop of every block type (bright frames, no mob, not the player's tile)
use = bright[:, None] & clean
protos, ids = [], []
for b in range(tg.NUM_BLOCKS):
    m = use & (blocks == b)
    if m.sum() >= 5:
        protos.append(true[m].mean(0)); ids.append(b)
protos, ids = np.stack(protos), np.array(ids)
print('prototypes for', [tg.BLOCK_NAMES[i] for i in ids], f'({time.time()-t0:.0f} s)')


def classify(crops):
    d = ((crops[:, None, :] - protos[None]) ** 2).mean(-1)
    return ids[d.argmin(1)]


def recall(crops_all, frame_idx, labels_all, use_mask):
    out = {}
    m = use_mask[frame_idx]
    lab = labels_all[frame_idx]
    pred = classify(crops_all[m]) if m.any() else np.array([])
    truth = lab[m]
    for b in ids:
        s = truth == b
        out[b] = (float((pred[s] == b).mean()) if s.sum() else float('nan'), int(s.sum()))
    return out


def decode(model, frames_idx, predicted=False):
    jepa = model.get_level(1)
    z = check.encode(jepa, part.pixels[frames_idx], 'cpu')
    if predicted:
        z = check.predict(jepa, z, part.action[frames_idx], 'cpu')
    return check.decoded_frames(jepa.decoder, z, 'cpu')[:, :63]


# ceiling: the nearest prototype on the REAL crops; then the decoded crops of each model; then the decoded prediction of the next frame
res = {}
res['real crops (ceiling)'] = recall(true.reshape(len(part), 63, 147).reshape(-1, 63, 147), np.arange(len(part)), blocks, use)
frames_t = t
for key, path in (('stage 1 e8 D(z)', R / 'V4/seed42/V4_epoch_8_object.ckpt'), ('stage 2 e4 D(z)', R / 'V4_stage2_from8/seed42/V4_stage2_from8_epoch_4_object.ckpt')):
    model = load_model(path).eval()
    allidx = np.arange(len(part))
    dz = decode(model, allidx)
    res[key] = recall(dz, allidx, blocks, use)
    pred_next = decode(model, t, predicted=True)                # decoded prediction of frame t+1, scored on frame t+1's tiles
    res[key.replace('D(z)', 'D(prediction)')] = recall(pred_next, t1, blocks, use)
    print(key, f'({time.time()-t0:.0f} s)')
cols = list(res)
print(f"\n{'tile':16}{'n':>7}" + ''.join(f'{c:>24}' for c in cols))
order = sorted(ids, key=lambda b: -res[cols[0]][b][1])
for b in order:
    print(f'{tg.BLOCK_NAMES[b]:16}{res[cols[0]][b][1]:7d}' + ''.join(f'{res[c][b][0]:24.2f}' for c in cols))

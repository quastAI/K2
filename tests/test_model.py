import numpy as np
import torch

from k2 import layout as L
from k2.data import Frames, flat_state
from k2.losses import SIGReg, token_losses, variant_loss
from k2.model import Decoder, LinearDecoder
from k2.train import Config
from k2.model import World

from .conftest import synthetic_table


def small(variant, **kw):
    return Config(variant=variant, width=32, layers=1, heads=2, queries=2, query_dim=16, latent=32, pred_width=32,
                  pred_layers=1, dec_layers=1, sigreg_proj=64, device="cpu", **kw)


def batch(k=2, context=2, b=8):
    fr = Frames(synthetic_table())
    pool = fr.starts(k, context)
    return fr, *fr.steps(pool[:b], k, context)


def test_frames_and_flat_state():
    t = synthetic_table()
    fr = Frames(t)
    f = fr.frames(torch.arange(5))
    np.testing.assert_array_equal(flat_state(f).numpy(), L.expand(L.take(t, slice(0, 5))))


def test_steps_stay_inside_episodes():
    fr, f, m = batch(k=4, context=3, b=64)
    assert f["block"].shape == (64, 4, L.TILES) and m.shape == (64, 3, 4)
    assert (m >= 0).all()


def test_shapes_and_rollout():
    torch.manual_seed(0)
    cfg = small("E1", k=2, context=2)
    model = World(cfg)
    assert model.decoder is None and World(small("E0")).predictor is None
    _, f, m = batch()
    z = model.encoder(f)
    assert z.shape == (8, 3, 32)
    out = Decoder(32, 32, 1, 2)(z)
    assert out["block"].shape == (8, 3, L.TILES, L.NUM_BLOCKS) and out["light"].shape == (8, 3)
    assert token_losses(out, f).shape == (8, 3, L.NUM_TOKENS)
    torch.nn.init.normal_(model.predictor.out.weight, std=0.1)
    one = model.predictor(z[:, :1], m[:, :1])[:, -1]
    roll = model.predictor.rollout(z[:, :1], m)
    assert roll.shape == (8, 2, 32)
    torch.testing.assert_close(roll[:, 0], one)
    # the second step sees the first prediction fed back, with the context of C = 2
    two = model.predictor(torch.stack([z[:, 0], one], 1), m)[:, -1]
    torch.testing.assert_close(roll[:, 1], two)
    assert LinearDecoder(32)(z)["mob"].shape == (8, 3, L.TILES, L.NUM_MOBS)


def test_sigreg_prefers_gaussian():
    torch.manual_seed(0)
    s = SIGReg(num_proj=256)
    gauss = s(torch.randn(1, 512, 16))
    collapsed = s(torch.randn(1, 512, 16) * 0.01)
    assert gauss < collapsed


def test_variants_train_and_e2_detaches_the_target():
    torch.manual_seed(0)
    _, f, m = batch()
    for v in ("E0", "E1", "E2"):
        cfg = small(v, k=2, context=2)
        model = World(cfg)
        loss, logs = variant_loss(model, f, m, cfg, SIGReg(num_proj=64))
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None for p in model.encoder.parameters() if p.requires_grad)
    # E1 against E2: the prediction loss reaches the target's encoding only in E1
    for v, flows in (("E1", True), ("E2", False)):
        cfg = small(v, k=2, context=2, sigreg_weight=0.0)
        model = World(cfg)
        torch.nn.init.normal_(model.predictor.out.weight, std=0.1)
        f_last = {key: x.clone() for key, x in f.items()}
        f_last["light"].requires_grad_(True)
        loss, _ = variant_loss(model, f_last, m, cfg, SIGReg(num_proj=64))
        loss.backward()
        grad_target = f_last["light"].grad[:, -1].abs().sum()  # the last frame is only ever a target
        assert (grad_target > 0) == flows

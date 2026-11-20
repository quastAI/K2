import numpy as np
import torch

from k2 import layout as L
from k2.data import Frames, flat_state
from k2.losses import token_losses, variant_loss
from k2.model import Decoder, LinearDecoder, Predictor
from k2.train import Config
from k2.model import World

from .conftest import synthetic_table


def small(variant, **kw):
    return Config(variant=variant, width=32, layers=1, heads=2, queries=2, query_dim=16, latent=32, pred_width=32,
                  pred_layers=1, dec_layers=1, device="cpu", **kw)


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
    assert World(small("E0")).predictor is None
    pred = Predictor(32, k=2, context=2, width=32, layers=1, heads=2)
    _, f, m = batch()
    z = World(small("E0")).encoder(f)
    assert z.shape == (8, 3, 32)
    out = Decoder(32, 32, 1, 2)(z)
    assert out["block"].shape == (8, 3, L.TILES, L.NUM_BLOCKS) and out["light"].shape == (8, 3)
    assert token_losses(out, f).shape == (8, 3, L.NUM_TOKENS)
    torch.nn.init.normal_(pred.out.weight, std=0.1)
    one = pred(z[:, :1], m[:, :1])[:, -1]
    roll = pred.rollout(z[:, :1], m)
    assert roll.shape == (8, 2, 32)
    torch.testing.assert_close(roll[:, 0], one)
    # the second step sees the first prediction fed back, with the context of C = 2
    two = pred(torch.stack([z[:, 0], one], 1), m)[:, -1]
    torch.testing.assert_close(roll[:, 1], two)
    assert LinearDecoder(32)(z)["mob"].shape == (8, 3, L.TILES, L.NUM_MOBS)


def test_readout_whitens_its_input():
    """A latent stretched along one direction reads the same after whitening: zero mean, unit covariance; a direction
    below 1e-6 of the largest variance is not blown up to unit variance (it would amplify numerical noise)."""
    from k2.readouts import Whitened

    torch.manual_seed(0)
    z = torch.randn(4096, 8) @ torch.diag(torch.tensor([50.0, 1, 1, 1, 1, 1, 1, 0.5])) + 3.0
    w = Whitened(LinearDecoder(8), z).white(z)
    assert torch.allclose(w.mean(0), torch.zeros(8), atol=1e-3)
    assert torch.allclose(torch.cov(w.T), torch.eye(8), atol=2e-2)
    z[:, -1] = 3.0 + 1e-5 * torch.randn(4096)  # variance 1e-10 of the largest
    w = Whitened(LinearDecoder(8), z).white(z)
    assert torch.cov(w.T).diagonal().min() < 0.01


def test_e0_trains():
    torch.manual_seed(0)
    _, f, m = batch()
    cfg = small("E0", k=2, context=2)
    model = World(cfg)
    loss, logs = variant_loss(model, f, m, cfg)
    loss.backward()
    assert torch.isfinite(loss) and logs["recon"] == loss.item()
    assert all(p.grad is not None for p in model.encoder.parameters() if p.requires_grad)

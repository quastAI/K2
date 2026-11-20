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
        if v != "E0":  # the gradient balance: prediction against SIGReg, on the encoder
            assert logs["grad_pred"] > 0 and logs["grad_rest"] > 0
            assert abs(logs["grad_rest_over_pred"] - logs["grad_rest"] / logs["grad_pred"]) < 1e-6
    # the prediction loss reaches the target's encoding in E1, not in E2
    grads = {}
    for v in ("E1", "E2"):
        torch.manual_seed(1)
        cfg = small(v, k=2, context=2, sigreg_weight=0.0)
        model = World(cfg)
        torch.nn.init.normal_(model.predictor.out.weight, std=0.1)
        f_last = {key: x.clone() for key, x in f.items()}
        f_last["light"].requires_grad_(True)
        loss, _ = variant_loss(model, f_last, m, cfg, SIGReg(num_proj=64))
        loss.backward()
        grads[v] = f_last["light"].grad[:, -1]  # the last frame is only ever a target
    assert grads["E1"].abs().sum() > 0 and grads["E2"].abs().sum() == 0


def test_e3_grace_model_two_stages():
    """Stage 1 (error-weighted reconstruction + detached prediction) before the switch, stage 2 (decoded prediction +
    gated reconstruction + detached prediction) from it on; every module gets a gradient, the logged keys are the same."""
    torch.manual_seed(0)
    _, f, m = batch()
    keys = {}
    for step in (0, 5):
        cfg = small("E3", k=2, context=2, switch=5)
        model = World(cfg)
        loss, logs = variant_loss(model, f, m, cfg, step=step, diag=True)
        loss.backward()
        assert torch.isfinite(loss)
        assert logs["stage"] == (1 if step < 5 else 2)
        assert 0.0 <= logs["gate_open"] <= 1.0 and 0.0 <= logs["gate_shut"] <= 1.0
        for part in (model.encoder, model.predictor, model.decoder):
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in part.parameters())
        keys[step] = set(logs)
    assert keys[0] == keys[5]  # one csv header across the switch
    assert "grad_rest_over_pred" in keys[0]
    # without diag, stage 1 skips the gate's diagnostic terms
    cfg = small("E3", k=2, context=2, switch=5)
    _, logs = variant_loss(World(cfg), f, m, cfg, step=0, diag=False)
    assert "gate_open" not in logs


def test_e3_error_weights_are_label_free_and_capped():
    from k2.losses import error_weights

    loss = torch.tensor([0.1, 0.1, 0.1, 10.0])
    w = error_weights(loss, 1.0, 20.0)
    assert w[-1] > w[0]  # the badly reconstructed token counts more
    assert w.max() <= 20.0 + 1e-6
    w0 = error_weights(loss, 0.0, 20.0)
    assert torch.allclose(w0, torch.ones(4))  # power 0 is the plain mean

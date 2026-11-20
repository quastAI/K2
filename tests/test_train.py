import json

import numpy as np

from k2 import layout, report
from k2.readouts import run_readouts
from k2.train import Config, read_log, train


def test_train_resume_readouts_report(cache_dir, tmp_path):
    kw = dict(width=32, layers=1, heads=2, queries=2, query_dim=16, latent=32, pred_width=32, pred_layers=1,
              dec_layers=1, sigreg_proj=64, batch=8, log_every=1, save_every=2, device="cpu", cache=str(cache_dir))
    ctl = Config(variant="E1", k=2, steps=2, checkpoints=(0,), out=str(tmp_path / "E1"), **kw)
    train(ctl)
    cfg = Config(variant="E2", k=2, steps=2, checkpoints=(0,), out=str(tmp_path / "E2"), control=ctl.out, **kw)
    train(cfg)
    cfg.steps = 4
    train(cfg)  # resumes from last.pt at step 2
    log = read_log(cfg.out)
    np.testing.assert_array_equal(log["step"], [1, 2, 3, 4])
    for run in (ctl.out, cfg.out):
        run_readouts(run, "cpu", decoder_steps=2, n_train=64, n_eval=64, horizon=2)
    res = json.loads((tmp_path / "E2" / "readouts.json").read_text())
    # every checkpoint file is read out, the end of the first, shorter run included
    assert set(res) == {"0", "2", "4"} and res["4"]["kept"] is None  # no oracle table in the synthetic cache
    text = report.write_report(cfg.out)
    assert "E1 (control)" in text and "What this does not show" in text


def test_kept_against_predictable_counts_and_report(cache_dir, tmp_path):
    """The 2 x 2 per kind and class, from a synthetic oracle: counts add up, unchanged apart, the report prints them."""
    import torch

    from k2.cache import load
    from k2.data import Frames
    from k2.model import Decoder, Encoder
    from k2.readouts import kept_vs_predictable
    from k2.report import kept_lines

    val = Frames(load("val", cache_dir))
    rows = np.arange(0, 60)
    start = layout.token_values(layout.take(val.table, rows))
    rng = np.random.default_rng(0)
    p = np.where(rng.random(start.shape) < 0.3, rng.choice([0.5, 0.8, 0.95], start.shape), 1.0).astype(np.float32)
    mode = np.where(rng.random(start.shape) < 0.3, start + 1, start)
    oracle = {"k": np.int64(1), "n_keys": np.int64(64), "rows": rows, "p": p, "mode": mode}
    torch.manual_seed(0)
    enc, dec = Encoder(32, 1, 2, 2, 16, 32).eval(), Decoder(32, 32, 1, 2).eval()
    out = kept_vs_predictable(enc, dec, val, oracle)
    assert out["lines"] == ["1", "0.9", "0.75"]
    chg = (mode != start) | (p < 1)
    a = out["all"]
    assert a["changing"]["n"] == chg.sum() and a["unchanged"]["n"] == (~chg).sum()
    for line, c in a["changing"]["by_line"].items():
        assert sum(c.values()) == chg.sum()
    one = a["changing"]["by_line"]["1"]
    assert one["pred_kept"] + one["pred_dropped"] == ((p == 1) & chg).sum()
    assert sum(g["changing"]["n"] for g in out["kinds"].values()) >= chg.sum()  # a tile counts for its block and mobs
    text = "\n".join(kept_lines(out, out, "E1"))
    assert "| kind |" in text and "| class |" in text and "E1, all agreed" in text and "all agreed" in text


def test_e3_switches_stage_and_resumes_across_the_switch(cache_dir, tmp_path):
    kw = dict(width=32, layers=1, heads=2, queries=2, query_dim=16, latent=32, pred_width=32, pred_layers=1,
              dec_layers=1, sigreg_proj=64, batch=8, log_every=1, save_every=1, device="cpu", cache=str(cache_dir))
    cfg = Config(variant="E3", k=2, steps=3, switch=2, checkpoints=(0, 2), out=str(tmp_path / "E3"), **kw)
    train(cfg)
    cfg.steps = 5
    train(cfg)  # resumes in stage 2
    log = read_log(cfg.out)
    np.testing.assert_array_equal(log["step"], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(log["stage"], [1, 1, 2, 2, 2])
    run_readouts(cfg.out, "cpu", decoder_steps=2, n_train=64, n_eval=64, horizon=2)
    assert "E3" in report.write_report(cfg.out)

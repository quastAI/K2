import json

import numpy as np

from k2 import report
from k2.readouts import run_readouts
from k2.train import Config, read_log, train


def test_train_resume_readouts_report(cache_dir, tmp_path):
    kw = dict(width=32, layers=1, heads=2, queries=2, query_dim=16, latent=32, pred_width=32, pred_layers=1,
              dec_layers=1, sigreg_proj=64, batch=8, log_every=1, save_every=2, device="cpu", cache=str(cache_dir))
    ctl = Config(variant="E3", k=2, steps=2, checkpoints=(0,), out=str(tmp_path / "E3"), **kw)
    train(ctl)
    cfg = Config(variant="E4", k=2, steps=2, checkpoints=(0,), out=str(tmp_path / "E4"), control=ctl.out, **kw)
    train(cfg)
    cfg.steps = 4
    train(cfg)  # resumes from last.pt at step 2
    log = read_log(cfg.out)
    np.testing.assert_array_equal(log["step"], [1, 2, 3, 4])
    for run in (ctl.out, cfg.out):
        run_readouts(run, "cpu", decoder_steps=2, n_train=64, n_eval=64, horizon=2)
    res = json.loads((tmp_path / "E4" / "readouts.json").read_text())
    assert set(res) == {"0", "4"} and res["4"]["kept"] is None  # no oracle table in the synthetic cache
    text = report.write_report(cfg.out)
    assert "E3 (control)" in text and "What this does not show" in text

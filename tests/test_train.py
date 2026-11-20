import json

import numpy as np

from k2 import report
from k2.readouts import run_readouts
from k2.train import Config, read_log, train


def test_train_resume_readouts_report(cache_dir, tmp_path):
    kw = dict(width=32, layers=1, heads=2, queries=2, query_dim=16, latent=32, pred_width=32, pred_layers=1,
              dec_layers=1, batch=8, log_every=1, save_every=2, device="cpu", cache=str(cache_dir))
    ctl = Config(variant="E0", k=2, steps=2, checkpoints=(0,), out=str(tmp_path / "E0"), **kw)
    train(ctl)
    cfg = Config(variant="E0", k=2, steps=2, checkpoints=(0,), out=str(tmp_path / "E0b"), control=ctl.out, **kw)
    train(cfg)
    cfg.steps = 4
    train(cfg)  # resumes from last.pt at step 2
    log = read_log(cfg.out)
    np.testing.assert_array_equal(log["step"], [1, 2, 3, 4])
    for run in (ctl.out, cfg.out):
        run_readouts(run, "cpu", decoder_steps=2, n_train=64, n_eval=64, horizon=2)
    res = json.loads((tmp_path / "E0b" / "readouts.json").read_text())
    # every checkpoint file is read out, the end of the first, shorter run included
    assert set(res) == {"0", "2", "4"} and "prediction" not in res["4"]  # E0 has no predictor
    assert res["4"]["latent"]["rank"] >= 1 and res["4"]["version"] == 3
    own, sens = res["4"]["own"], res["4"]["sensitivity"]
    assert 0 <= own["frames_exact"] <= own["factors_exact"] <= 1
    assert sens["unit"] > 0 and "mob zombie, 1 away" in sens["edits"] and "item wood ±1" in sens["edits"]
    # readouts of an older version are computed again, current ones are kept
    res["2"]["version"] = 1
    res["4"]["latent"]["rank"] = 12345
    (tmp_path / "E0b" / "readouts.json").write_text(json.dumps(res))
    run_readouts(cfg.out, "cpu", decoder_steps=2, n_train=64, n_eval=64, horizon=2)
    again = json.loads((tmp_path / "E0b" / "readouts.json").read_text())
    assert again["2"]["version"] == 3 and again["4"]["latent"]["rank"] == 12345
    assert set(json.loads((tmp_path / "E0b" / "readouts_v1.json").read_text())) == {"2"}  # the old one is kept
    text = report.write_report(cfg.out)
    assert "E0 (control)" in text and "What this does not show" in text


def test_e1_linear_resolution_readout(cache_dir, tmp_path):
    from k2.readouts import run_resolution

    cfg = Config(variant="E1", arch="linear", k=2, steps=3, latent=8, batch=16, checkpoints=(0,), log_every=1,
                 device="cpu", cache=str(cache_dir), out=str(tmp_path / "E1"))  # fmt: skip
    train(cfg)
    res = run_resolution(cfg.out, "cpu", mlp_steps=5)
    assert set(res) == {"0", "3"}
    assert 0 <= res["3"]["own"]["frames_exact"] <= res["3"]["own"]["factors_exact"] <= 1
    for part in ("holds", "holds_mlp", "foresees", "copy"):
        assert "iron|quadrant" in res["3"][part] and "mobs|view" in res["3"][part]
    assert res["3"]["latent"]["rank"] >= 1

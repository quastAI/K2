"""The guard: replaying with the true keys reproduces the stored frames (needs AAC's val file; skipped without it)."""

import numpy as np
import pytest

from k2 import cache, sim

pytestmark = pytest.mark.skipif(not cache.h5_path("val").exists(), reason="AAC's val file is not here")


def test_replay_contract(tmp_path):
    table = cache.build("val", out_dir=tmp_path, max_episodes=2, sample=16, tasks=1, task_offset=8)
    full = cache.load_full("val", tmp_path)
    assert sim.check_episodes(table, [0, 1]) == []
    assert sim.check_steps(table, full, full["sample"][:8]) == []
    assert len(full["task_start"]) == 1
    np.testing.assert_array_equal(full["task_goal"] - full["task_start"], 8)

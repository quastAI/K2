"""Write notebooks/E0.ipynb .. E4.ipynb from one template; they differ only in the configuration cell.

    python notebooks/make_notebooks.py
"""

from pathlib import Path

import nbformat as nbf

VARIANTS = {  # id: (control, title)
    "E0": (None, "capacity check (autoencoder only)"),
    "E1": ("E2", "prediction with the gradient through the target + SIGReg (A3 analogue)"),
    "E2": ("E1", "prediction with a detached target + SIGReg (A4 analogue)"),
    "E3": ("E4", "detached prediction + decoded prediction + reconstruction (Sieve stage 1 analogue)"),
    "E4": ("E3", "E3 with the reconstruction gated per token (V5's relative gate)"),
}

INTRO = """# {id}: {title}

Runs this variant end to end: data check, training, readouts, report, and level-2 planning at k > 1. The notebooks
E0 to E4 share the code in `k2/` and differ only in the configuration cell (`notebooks/make_notebooks.py` writes them).

**Before running on a pod:** clone to `/root/K2`, make the venv there, upload the cache built on the Mac
(`data/cache`: `train.npz`, `val.npz`, `val_full.npz`, `oracle_k*.npz`) to `/root/K2/data/cache`, and point `RUNS` at
the network volume. The training is minutes long: run it only after a go-ahead."""

SETUP = """import os, subprocess, sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from k2 import cache, readouts, report, train

print("torch threads:", train.limit_threads())"""

CONFIG = """VARIANT = "{id}"
CONTROL = {control!r}  # the variant this run is read against (E0 is the reference for all)
K = 1  # stride; step 5 of the order of work runs K = 8
CONTEXT = 1  # C
STEPS = 8000  # about one epoch at K = 1
CACHE = ROOT / "data/cache"  # the local disk
RUNS = ROOT / "runs"  # on a pod: a folder on the volume, e.g. Path("/workspace/K2/runs")
PLAN = K > 1  # level-2 planning, with the executor check (oracle subgoals)

cfg = train.Config(
    variant=VARIANT, k=K, context=CONTEXT, steps=STEPS, cache=str(CACHE), out=str(RUNS / f"{{VARIANT}}_k{{K}}"),
    control=str(RUNS / f"{{CONTROL}}_k{{K}}") if CONTROL else None,
)
cfg"""

DATA = """import numpy as np

for split in ("train", "val"):
    t = cache.load(split, CACHE)
    src, n = np.unique(t["ep_source"], return_counts=True)
    print(f"{split}: {len(t['action'])} frames, {len(t['ep_len'])} episodes, sources {dict(zip(src.tolist(), n.tolist()))}")
t = cache.load("train", CACHE)
print("level-2 steps to train on:", len(cache.starts(t, cfg.k * cfg.context, cfg.sources)))
md = CACHE / f"oracle_k{K}.md"
print(md.read_text() if md.exists() else f"No oracle table for k = {K}: run `python -m k2.oracle --k {K}` on the Mac.")"""

TRAIN = "train.train(cfg)"

READ = "results = readouts.run_readouts(cfg.out)"

PLANNING = """if PLAN:
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}  # the image's CUDA path crashes JAX
    subprocess.run([sys.executable, "-m", "k2.plan", cfg.out, "--split", "val", "--device", cfg.device], cwd=ROOT, env=env, check=True)"""

REPORT = """from IPython.display import Markdown

Markdown(report.write_report(cfg.out))"""


def notebook(vid, control, title):
    nb = nbf.v4.new_notebook()
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb.cells = [
        md(INTRO.format(id=vid, title=title)),
        code(SETUP),
        md("## Configuration"),
        code(CONFIG.format(id=vid, control=control)),
        md("## Data check"),
        code(DATA),
        md("## Training"),
        code(TRAIN),
        md("## Readouts\n\nA fresh decoder and a linear probe per checkpoint, kept against predictable, prediction, geometry."),
        code(READ),
        md("## Planning (K > 1)\n\nJAX on the CPU in a subprocess, without `LD_LIBRARY_PATH`."),
        code(PLANNING),
        md("## Report"),
        code(REPORT),
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


if __name__ == "__main__":
    here = Path(__file__).parent
    for vid, (control, title) in VARIANTS.items():
        nbf.write(notebook(vid, control, title), here / f"{vid}.ipynb")
        print(here / f"{vid}.ipynb")

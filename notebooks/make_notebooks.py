"""Write notebooks/E0.ipynb .. E3.ipynb from one template; they differ only in the configuration cell.

    python notebooks/make_notebooks.py
"""

from pathlib import Path

import nbformat as nbf

VARIANTS = {  # id: (control, title)
    "E0": (None, "capacity check (autoencoder only)"),
    "E1": ("E2", "prediction with the gradient through the target + SIGReg (A3 analogue)"),
    "E2": ("E1", "prediction with a detached target + SIGReg (A4 analogue)"),
    "E3": ("E0", "the Grace model: reconstruction first, then keep what the predictor foresees"),
}
STEPS = {  # id: the STEPS line's value
    "E3": "16000  # stage 1 until SWITCH, stage 2 after it (about four epochs at K = 1, batch 512)",
}
SETTINGS = {  # id: (extra lines before CACHE, extra arguments of train.Config)
    "E3": (
        "SWITCH = 10000  # stage 2 starts here (fresh optimiser state, same rate); raise SWITCH and STEPS to extend stage 1\n"
        "CHECKPOINTS = (0, 3000, SWITCH)  # and the last step\n",
        "\n    switch=SWITCH, checkpoints=CHECKPOINTS,",
    ),
}

INTRO = """# {id}: {title}

Runs this variant end to end: data check, training, readouts, report, and level-2 planning at k > 1. The notebooks
E0 to E3 share the code in `k2/` and differ only in the configuration cell (`notebooks/make_notebooks.py` writes them).

**On a pod:** unzip `k2_pod.zip` into `/root` (the local disk, not `/workspace`; the zip holds the code and the cache
built on the Mac), open this notebook with the pod's Python kernel and run the cells in order. The first cell installs
`requirements.txt` into the kernel's Python (torch from the image stays), the tests cell checks the code and the
cache on this machine, and `RUNS` below should point at the network volume. The training is minutes long: run it
only after a go-ahead."""

ENV = """import subprocess, sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()


def run(cmd, **kw):
    \"\"\"A command whose output shows in the cell (the tail of it) and which fails the cell if it fails.\"\"\"
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    print((r.stdout + r.stderr)[-3000:])
    if r.returncode:
        raise RuntimeError(f"failed with exit code {r.returncode}: {' '.join(map(str, cmd))}")


# jax 0.4.38 CPU wheels next to the image's torch; nothing changes when everything is installed already
run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "-r", ROOT / "requirements.txt"])
print("requirements are installed in", sys.executable, "(if pip changed numpy or torch just now, restart the kernel)")"""

SETUP = """import os, subprocess, sys

sys.path.insert(0, str(ROOT))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from k2 import cache, readouts, report, train

print("torch threads:", train.limit_threads())"""

TESTS = """# the code and the cache work on this machine (about 20 s; the replay test skips itself without AAC's HDF5 file)
env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}  # the image's CUDA path crashes JAX
env["JAX_PLATFORMS"] = "cpu"
run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT, env=env)"""

CONFIG = """VARIANT = "{id}"
CONTROL = {control!r}  # the variant this run is read against (E0 is the reference for all)
K = 1  # stride; step 5 of the order of work runs K = 8
CONTEXT = 1  # C
STEPS = {steps}
{settings}CACHE = ROOT / "data/cache"  # the local disk
RUNS = ROOT / "runs"  # on a pod: a folder on the volume, e.g. Path("/workspace/K2/runs")
PLAN = K > 1  # level-2 planning, with the executor check (oracle subgoals)

cfg = train.Config(
    variant=VARIANT, k=K, context=CONTEXT, steps=STEPS, cache=str(CACHE), out=str(RUNS / f"{{VARIANT}}_k{{K}}"),
    control=str(RUNS / f"{{CONTROL}}_k{{K}}") if CONTROL else None,{args}
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
        md("## Environment\n\nInstalls the requirements into this kernel's Python (a pip call that does nothing when they are there)."),
        code(ENV),
        code(SETUP),
        md("## Tests\n\n`pytest` on the Mac-built code and a tiny synthetic cache: a failure here is an environment problem, not a result."),
        code(TESTS),
        md("## Configuration"),
        code(CONFIG.format(id=vid, control=control, steps=STEPS.get(vid, "8000  # about two epochs at K = 1 (batch 512)"),
                           settings=SETTINGS.get(vid, ("", ""))[0], args=SETTINGS.get(vid, ("", ""))[1])),
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

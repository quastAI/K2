"""Write notebooks/<ID>.ipynb from one template; the variants differ only in the configuration cell.

    python notebooks/make_notebooks.py
"""

from pathlib import Path

import nbformat as nbf

VARIANTS = {  # id: (control, title)
    "E0": (None, "capacity reference (autoencoder only)"),
}
INTRO = """# {id}: {title}

Runs this variant end to end: data check, training, readouts, report. The notebooks share the code in `k2/` and
differ only in the configuration cell (`notebooks/make_notebooks.py` writes them).

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
CONTROL = {control!r}  # the variant this run is read against
K = 1  # the run's stride: E0 reconstructs single frames, so K only names the run (this continues runs/E0_k1)
STRIDE = 4  # the level-2 stride the readouts measure in
CONTEXT = 1  # C
STEPS = 20000  # resumes from runs/E0_k1/last.pt (step 12000)
CHECKPOINTS = (0, 1000, 3000, 8000, 12000, 16000)  # and the last step; files already there are kept
CACHE = ROOT / "data/cache"  # the local disk
RUNS = ROOT / "runs"  # the zip brings runs/E0_k1; copy it to the volume afterwards

cfg = train.Config(
    variant=VARIANT, k=K, context=CONTEXT, steps=STEPS, cache=str(CACHE), out=str(RUNS / f"{{VARIANT}}_k{{K}}"),
    control=str(RUNS / f"{{CONTROL}}_k{{K}}") if CONTROL else None, checkpoints=CHECKPOINTS,
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

READ = "results = readouts.run_readouts(cfg.out, stride=STRIDE)"

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
        code(CONFIG.format(id=vid, control=control)),
        md("## Data check"),
        code(DATA),
        md("## Training"),
        code(TRAIN),
        md("## Readouts\n\nA fresh decoder and a linear probe per checkpoint, prediction against copying, the raw latent's shape."),
        code(READ),
        md("## Report"),
        code(REPORT),
    ]
    return finish(nb)


def finish(nb):
    """Fixed cell ids (nbformat draws random ones, which would change every notebook on every write) and the kernel."""
    for i, cell in enumerate(nb.cells):
        cell["id"] = f"cell-{i:02d}"
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


E1_IDEA = r"""# E1: does a gradient through the target coarse-grain?

## Question and hypothesis

The goal (README) asks for a latent that holds as much about the state as stays predictable at stride k = 4, with
what is unpredictable coarsened (a mob's tile becomes its quadrant), not deleted. A JEPA ranks by persistence: it
keeps slow content and its harmonics and shrinks the rest (README, Concepts, R1, R2). E1 replaces the anti-collapse
"spread" by "information": the reconstruction of the input, which a harmonic does not improve.

**Hypothesis.** Reconstruction plus a scale-free prediction penalty, *with* the prediction's gradient reaching the
target latent, coarse-grains: it keeps a coarse, predictable version of content whose fine version is unpredictable.

| gradient through the target | reconstruction | expected | known from |
|---|---|---|---|
| from step 0 | yes | **coarse-grain** (if there is no race) | untested |
| after a warm-up | yes | coarse-grain (if there is a race) | untested |
| never (stop-gradient) | yes | no selection: everything kept | E0, E3 stage 1 |
| from step 0 | no | delete | AAC A3 |
| never | no | control (collapse onto the most predictable directions) | |

## What we want to learn

1. **The core claim:** with reconstruction holding the content, does the target gradient coarse-grain (row 1) instead
   of keeping everything (row 3) or deleting (row 4)?
2. **The race:** does it happen at all once the reconstruction is there (row 1 against row 2, the early checkpoints)?
3. **lam as the granularity:** which content stays at tile, quadrant or count level as lam grows (R2: the level is a
   choice, so we want the curve, not one point).
4. **The actions:** are the facing and the view's shift foreseen (R3), which the ideal latent without actions loses (R1)?
5. **Go or no-go for stage 2:** does the mechanism hold in the linear model before we spend GPU time on the deep one?

## The objective

    L = rec_weight * sum_tokens w * CE(D(z_t), s_t)   +   lam * E[ e^T C^-1 e ],   e = P(z_t, m_t) - z_{t+4}

- **Reconstruction** of every frame, summed over the 82 tokens (nats per frame), weighted per token by its own error
  (label-free, as in Grace). It measures what z knows about the state; a harmonic adds nothing to it.
- **Prediction** in the coordinates that whiten the target batch (C: its covariance, regularised by 1e-4 of the mean
  variance), **summed** over the directions. Shrinking z or one direction buys nothing; a direction the predictor
  cannot foresee costs about 1; adding predictable directions does not dilute the others.
- **lam** is a price in nats per frame: a fully unpredictable direction stays only if it carries more than about lam
  nats per frame. The game has no timescale gap (R2), so lam is a choice; we sweep it.
- **The actions** enter the predictor (R3: the facing and the view's shift need them).
- **The target gradient** (`target_from`): the step from which the prediction's gradient reaches z_{t+4} (0: from the
  start; -1: never). It is the lever: without it nothing pushes z_{t+4} itself to be predictable.

## Shortcuts and safeguards

1. Shrink the latent: closed by the whitening.
2. Keep unpredictable content at a tiny variance in one direction: closed per direction by the full whitening, down to
   the regularisation (1e-4 of the mean variance). Below it, it can slip through: we watch the raw latent's shape.
3. Full whitening makes every one of the D directions count: spare directions fill with harmonics (harmless) or with
   unpredictable content (penalised). D and lam are chosen together.
4. **Early pruning (the race):** the predictor starts bad, so everything looks unpredictable and the target gradient
   may delete before the predictor has learned. This spoiled ideas before (AAC A3, Grace). E1 runs the target gradient
   from step 0 *and*, as a variant, after a fixed warm-up; dense early checkpoints show what disappears and whether it
   comes back. No automatic switch (the old E3's plateau rule never fired).
5. The egocentric view shifts with every move: predictable with the actions, but the predictor must learn the shift;
   if it learns it late, the target gradient may delete the map.
6. Rare content is cheap for a plain reconstruction: the error weights counter it.

## What may look negative although the idea holds

- Read only per tile, coarse-graining looks like loss: we read tile, quadrant and count in view.
- E1 should hold less than E0: the criterion is "coarse but predictable", not "everything".
- One lam says little: too large deletes everything, too small is an autoencoder.
- Mobs are only moderately predictable even when coarse (R3): at any lam they blur or go; that is the tolerance, not
  a failure of the idea.
- Too short training (E0 needed more than 12k steps), one seed.

## Stages

1. **Linear (this machine, CPU):** a linear encoder, a predictor bilinear in the actions (a Koopman model with inputs),
   a linear decoder. Five arms x lam. Free of deep-network confounds; R1 is its reference. If the mechanism fails here,
   a deep model is unlikely to rescue it.
2. **Deep (GPU), only after stage 1 is read:** the token encoder, the same arms at the lam chosen from stage 1.

Readouts per checkpoint (`readouts.run_resolution`, labels for evaluation only): R² on val of what z **holds**, what
the predicted latent **foresees** at t+4, and what **copying** z_t gives, per group at tile / quadrant / view; the
raw latent's participation ratio."""

E1_STAGE1 = """from IPython.display import Markdown
from k2 import readouts, report, train

CACHE = ROOT / "data/cache"
RUNS = ROOT / "runs/E1_linear"
K, LATENT, STEPS = 4, 64, 5000
WARMUP = 1000  # the warm-up variant: the target gradient from this step on
LAMBDAS = (0.1, 1.0, 10.0)
ARMS = {  # name: (target_from, rec_weight)
    "target + rec": (0, 1.0),
    "warm-up target + rec": (WARMUP, 1.0),
    "stop-grad + rec": (-1, 1.0),
    "target, no rec": (0, 0.0),
    "stop-grad, no rec": (-1, 0.0),
}


def runs():
    \"\"\"{label: Config}: the arms with reconstruction at every lam, the ones without at lam 1 (lam only scales them).\"\"\"
    out = {}
    for arm, (target_from, rec) in ARMS.items():
        for lam in (LAMBDAS if rec > 0 else (1.0,)):
            name = f"{arm}, lam {lam:g}"
            out[name] = train.Config(variant="E1", arch="linear", k=K, latent=LATENT, steps=STEPS, lr=1e-3, warmup=100,
                                     batch=512, lam=lam, rec_weight=rec, target_from=target_from,
                                     checkpoints=(0, 100, 300, 1000, 3000), device="cpu", cache=str(CACHE),
                                     out=str(RUNS / name.replace(" ", "_").replace(",", "").replace("+", "and")))
    return out


print(len(runs()), "runs")"""

E1_TRAIN1 = """for cfg in runs().values():  # a finished run returns at once
    train.train(cfg)"""

E1_READ1 = """for cfg in runs().values():
    readouts.run_resolution(cfg.out, "cpu", stride=K)
labels = {name: cfg.out for name, cfg in runs().items()}
Markdown("### What z holds (end)\\n" + report.resolution_table(labels, "holds")
         + "\\n### What the predicted latent foresees at t + 4 (end)\\n" + report.resolution_table(labels, "foresees")
         + "\\n### Copying z_t (end)\\n" + report.resolution_table(labels, "copy"))"""

E1_RACE1 = """# the race: what z holds over the first steps, for the arms with a target gradient from step 0
early = {n: c.out for n, c in runs().items() if n.startswith(("target", "warm-up"))}
Markdown("\\n".join(f"### step {s}\\n" + report.resolution_table(early, "holds", str(s)) for s in (100, 300, 1000, 3000)))"""

E1_STAGE1B_MD = """## Stage 1b: can tiles be coarsened? (linear, D = 256)

At D = 64 no arm could represent tiles (stage 1), so coarse-graining against deletion was not testable. At D = 256 the
linear ideal holds quadrants (R1). Two runs, 10000 steps each: the strongest selector of stage 1 (target gradient after
a 1000-step warm-up, reconstruction, lam 10) and its reference (lam 0: a plain linear autoencoder of the same size; its
predictor gets no gradient, so only its `holds` is read).

**Read, fixed before the runs** (mobs against the reference):
- **coarse-grain:** the mob tile clearly below the reference, mob quadrant and count in view close to it;
- **delete:** mobs clearly below the reference at all three levels;
- **no selection:** mobs about equal at all three levels.

Ores and stations, being predictable, should stay or rise."""

E1_STAGE1B = """RUNS_B = ROOT / "runs/E1_linear256"
runs_b = {
    "warm-up target + rec, lam 10": train.Config(variant="E1", arch="linear", k=K, latent=256, steps=10000, lr=1e-3,
        warmup=100, lam=10.0, rec_weight=1.0, target_from=WARMUP, checkpoints=(0, 100, 300, 1000, 3000, 6000),
        device="cpu", cache=str(CACHE), out=str(RUNS_B / "warm-up_target_and_rec_lam_10")),
    "reference, lam 0": train.Config(variant="E1", arch="linear", k=K, latent=256, steps=10000, lr=1e-3,
        warmup=100, lam=0.0, rec_weight=1.0, target_from=-1, checkpoints=(0, 100, 300, 1000, 3000, 6000),
        device="cpu", cache=str(CACHE), out=str(RUNS_B / "reference_lam_0")),
}  # fmt: skip
for cfg in runs_b.values():  # about 8.5 minutes each on the Mac CPU (0.051 s per step, measured)
    train.train(cfg)
    readouts.run_resolution(cfg.out, "cpu", stride=K)
labels_b = {n: c.out for n, c in runs_b.items()}
Markdown("### holds\\n" + report.resolution_table(labels_b, "holds")
         + "\\n### foresees (the reference's predictor is untrained)\\n" + report.resolution_table(labels_b, "foresees"))"""

E1_STAGE2 = """RUN_STAGE2 = False  # only after stage 1 is read: set LAM2 from it, then True (GPU)
LAM2 = 1.0
RUNS2 = ROOT / "runs/E1_deep"  # on a pod: a folder on the volume
runs2 = {arm: train.Config(variant="E1", arch="deep", k=K, latent=256, steps=16000, lam=LAM2, rec_weight=rec,
                           target_from=target_from, checkpoints=(0, 100, 300, 1000, 3000, 8000),
                           cache=str(CACHE), out=str(RUNS2 / arm.replace(" ", "_").replace(",", "").replace("+", "and")))
         for arm, (target_from, rec) in ARMS.items()}
if RUN_STAGE2:
    for cfg in runs2.values():
        train.train(cfg)
        readouts.run_resolution(cfg.out, stride=K)
    Markdown(report.resolution_table({a: c.out for a, c in runs2.items()}, "holds")
             + report.resolution_table({a: c.out for a, c in runs2.items()}, "foresees"))"""


def e1_notebook():
    nb = nbf.v4.new_notebook()
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb.cells = [
        md(E1_IDEA),
        md("## Environment\n\nInstalls the requirements into this kernel's Python (a pip call that does nothing when they are there)."),
        code(ENV),
        code(SETUP),
        md("## Tests"),
        code(TESTS),
        md("## Stage 1: the mechanism, linear (CPU)\n\n11 runs of 5000 steps, about 17 minutes on the Mac (0.018 s per step, measured)."),
        code(E1_STAGE1),
        code(E1_TRAIN1),
        md("## Stage 1: readouts"),
        code(E1_READ1),
        md("## Stage 1: the race\n\nWhat disappears in the first steps, and whether it comes back."),
        code(E1_RACE1),
        md(E1_STAGE1B_MD),
        code(E1_STAGE1B),
        md("## Stage 2: deep (GPU)\n\nOnly after stage 1 is read; the warm-up arm only if stage 1 or the first deep checkpoints show a race."),
        code(E1_STAGE2),
    ]
    return finish(nb)


if __name__ == "__main__":
    here = Path(__file__).parent
    for vid, (control, title) in VARIANTS.items():
        nbf.write(notebook(vid, control, title), here / f"{vid}.ipynb")
        print(here / f"{vid}.ipynb")
    nbf.write(e1_notebook(), here / "E1.ipynb")
    print(here / "E1.ipynb")

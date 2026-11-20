"""Training of one variant: Adam, linear warmup then a constant learning rate, batch 512.

The run directory holds `config.json`, `log.csv` (every `log_every` steps), the checkpoints `ckpt_<step>.pt` at the
steps in `checkpoints` and at the end, and `last.pt` (every `save_every` steps; a restart resumes from it).

    python -m k2.train --variant E0 --k 1 --steps 12000 --out runs/E0_k1
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .cache import CACHE_DIR, load
from .data import Frames
from .losses import variant_loss
from .model import World

VARIANTS = ("E0",)


@dataclass
class Config:
    variant: str = "E0"
    k: int = 1  # stride
    context: int = 1  # C
    steps: int = 8000  # one epoch at k = 1 (about 2M level-2 steps) is about 3.9k steps of batch 512
    batch: int = 512
    lr: float = 3e-4
    warmup: int = 500
    # encoder
    width: int = 128
    layers: int = 3
    heads: int = 4
    queries: int = 4
    query_dim: int = 64
    latent: int = 256
    # predictor, decoder
    pred_width: int = 256
    pred_layers: int = 3
    dec_layers: int = 2
    # data: all sources (0 expert, 3 early checkpoint), as AAC's level 1; an open decision
    sources: tuple = (0, 3)
    checkpoints: tuple = (0, 100, 300, 1000, 3000)
    log_every: int = 50
    save_every: int = 1000
    seed: int = 0
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    cache: str = str(CACHE_DIR)
    out: str = "runs/E0_k1"
    control: str | None = None  # the run directory of this variant's control, for the report

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {self.variant}")
        self.sources = tuple(self.sources)
        self.checkpoints = tuple(self.checkpoints)


def limit_threads() -> int:
    """Limit torch and OpenMP to the container's CPU quota (RunPod shows more CPUs in `nproc` than it grants)."""
    import os

    n = os.cpu_count() or 1
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            n = max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    torch.set_num_threads(n)
    return n


def save_config(cfg: Config, out: Path):
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=1))


def _config(d: dict) -> Config:
    """A Config from a stored dict; settings the Config no longer has (in older runs) are dropped."""
    names = {f.name for f in dataclasses.fields(Config)}
    return Config(**{k: v for k, v in d.items() if k in names})


def load_config(run_dir) -> Config:
    return _config(json.loads((Path(run_dir) / "config.json").read_text()))


def checkpoint_steps(cfg: Config) -> list[int]:
    return sorted({s for s in cfg.checkpoints if s < cfg.steps} | {cfg.steps})


def load_model(path, device="cpu") -> tuple[Config, World]:
    """A checkpoint's config and model (eval mode)."""
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = _config({**blob["config"], "device": str(device)})
    model = World(cfg).to(device)
    model.load_state_dict(blob["model"])
    return cfg, model.eval()


def train(cfg: Config) -> Path:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out)
    torch.manual_seed(cfg.seed)
    gen = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    frames = Frames(load("train", cfg.cache), device)
    pool = frames.starts(cfg.k, cfg.context, cfg.sources)
    model = World(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(cfg.warmup, 1)))
    step = 0
    if (out / "last.pt").exists():
        blob = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        sched.load_state_dict(blob["sched"])
        try:
            gen.set_state(blob["gen"])
        except RuntimeError:  # saved on another device (CUDA and CPU generator states differ)
            gen.manual_seed(cfg.seed + blob["step"])
            print("batch generator reseeded: last.pt was saved on another device")
        step = blob["step"]
        print(f"resumed at step {step}")
    keep = set(checkpoint_steps(cfg))
    log_path = out / "log.csv"
    writer, log_file = None, open(log_path, "a", newline="")
    t0 = time.time()

    def save(path):
        torch.save({"config": dataclasses.asdict(cfg), "model": model.state_dict(), "step": step}, path)

    try:
        while True:
            if step in keep and step <= cfg.steps and not (out / f"ckpt_{step:06d}.pt").exists():
                save(out / f"ckpt_{step:06d}.pt")
            if step >= cfg.steps:
                break
            model.train()
            idx = torch.randint(len(pool), (cfg.batch,), device=device, generator=gen)
            f, m = frames.steps(pool[idx], cfg.k, cfg.context)
            diag = (step + 1) % cfg.log_every == 0 or step == 0
            loss, logs = variant_loss(model, f, m, cfg, step=step, diag=diag)
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f"loss is {loss.item()} at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % cfg.log_every == 0 or step == 1:
                row = {"step": step, "loss": loss.item(), "lr": sched.get_last_lr()[0],
                       "sec": round(time.time() - t0, 1), **logs}  # fmt: skip
                if writer is None:  # a resumed run keeps its file's columns (newer logged values are left out)
                    fresh = log_path.stat().st_size == 0
                    names = list(row) if fresh else next(csv.reader(open(log_path)))
                    writer = csv.DictWriter(log_file, fieldnames=names, extrasaction="ignore")
                    if fresh:
                        writer.writeheader()
                writer.writerow(row)
                log_file.flush()
            if step % cfg.save_every == 0 or step == cfg.steps:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "gen": gen.get_state(), "step": step}, out / "last.pt")  # fmt: skip
    finally:
        log_file.close()
    print(f"{cfg.variant}: {cfg.steps} steps in {time.time() - t0:.0f} s -> {out}")
    return out


def read_log(run_dir) -> dict[str, np.ndarray]:
    with open(Path(run_dir) / "log.csv") as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) if r.get(k) not in (None, "") else np.nan for r in rows]) for k in rows[0]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in dataclasses.fields(Config):
        if f.name in ("sources", "checkpoints"):
            p.add_argument(f"--{f.name}", type=int, nargs="+")
        elif f.name != "control":
            kind = type(f.default) if f.default is not dataclasses.MISSING else str
            p.add_argument(f"--{f.name}", type=kind)
    p.add_argument("--control")
    args = {k: v for k, v in vars(p.parse_args(argv)).items() if v is not None}
    limit_threads()
    train(Config(**args))


if __name__ == "__main__":
    main()

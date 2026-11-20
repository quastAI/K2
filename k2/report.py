"""The markdown report of a run, read against its control and with what it does not show.

    python -m k2.report runs/E0_k1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import layout as L
from .train import load_config, read_log

ROLE = {
    "E0": "capacity reference: what this encoder can hold under reconstruction alone",
    "E1": "reconstruction + lam x whitened prediction error: does a gradient through the target coarse-grain?",
}
NOT_SHOWN = {
    "E0": "E0 has no predictor: it shows what this encoder can hold, not what a predictive loss keeps.",
    "E1": "One lam, one latent size, one seed; whitening is regularised (content below 1e-4 of the mean variance can "
    "slip past the prediction penalty).",
}
COMMON = [
    "One seed per run.",
    "Held-out frames are the val worlds (all sources).",
    "The readouts read z whitened on their train frames; the fresh decoder (2000 steps on 65k train frames) gives a "
    "lower bound on what the latent holds, not on what a planner using the raw z sees. Classes marked `(low ceiling)` "
    "cannot be read even from the raw state; they are left out of the summaries.",
]
SUMMARY_KEYS = ("rare", "blocks", "mobs", "items", "vitals")


def _f(x, digits=3):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def _load(run_dir):
    p = Path(run_dir) / "readouts.json"
    return json.loads(p.read_text()) if p.exists() else {}


def write_report(run_dir) -> str:
    run_dir = Path(run_dir)
    cfg = load_config(run_dir)
    res = _load(run_dir)
    steps = sorted(res, key=int)
    ctl = _load(cfg.control) if cfg.control else {}
    ctl_cfg = load_config(cfg.control) if ctl else None
    ctl_last = ctl[max(ctl, key=int)] if ctl else None
    ctl_name = f"{ctl_cfg.variant} (control)" if ctl_cfg else None
    lines = [
        f"# {cfg.variant}, k = {cfg.k}, C = {cfg.context}",
        "",
        f"Role: {ROLE[cfg.variant]}. {cfg.steps} steps, batch {cfg.batch}, sources {list(cfg.sources)}.",
    ]
    if (run_dir / "log.csv").exists():
        log = read_log(run_dir)
        keys = [k for k in ("loss", "pred_over_copy", "recon", "grad_rest_over_pred", "z_std") if k in log]
        lines += ["", "Training, last logged step " + str(int(log["step"][-1])) + ": "
                  + ", ".join(f"{k} {_f(log[k][-1])}" for k in keys) + "."]  # fmt: skip
    if not steps:
        return _write(run_dir, lines + ["", "No readouts yet."], cfg)
    last = res[steps[-1]]

    lines += ["", "## 1. What the latent holds (fresh decoder on whitened z, val)", ""]
    head = ["class"] + [f"step {s}" for s in steps] + ([ctl_name] if ctl_last else [])
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    names = list(SUMMARY_KEYS) + L.class_order(list(last["fresh"]["classes"]))
    for n in names:
        get = lambda r: r["fresh"]["summary"].get(n) if n in SUMMARY_KEYS else r["fresh"]["classes"].get(n)  # noqa: E731
        label = f"**{n}**" if n in SUMMARY_KEYS else n + (" (low ceiling)" if n in L.LOW_CEILING else "")
        row = [label] + [_f(get(res[s])) for s in steps]
        row += [_f(get(ctl_last))] if ctl_last else []
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "Linear probe, last step: " + ", ".join(f"{k} {_f(last['linear']['summary'][k])}" for k in SUMMARY_KEYS) + "."]
    lines += ["", "The raw latent (participation ratio, rank, median slowness over one level-2 step): " + ", ".join(
        f"step {s} {_f(res[s]['latent']['participation_ratio'], 1)}, {res[s]['latent']['rank']}, "
        f"{_f(res[s]['latent']['slowness_median'])}" for s in steps if "latent" in res[s]) + "."]  # fmt: skip

    if "own" in last:
        lines += ["", "## Capacity (the model's own decoder, val)", "",
                  "Frames reconstructed exactly (all 145 factors) / factors exact: " + ", ".join(
                      f"step {s} {_f(res[s]['own']['frames_exact'])} / {_f(res[s]['own']['factors_exact'])}"
                      for s in steps if "own" in res[s]) + ".", ""]  # fmt: skip
        lines += ["| class | own decoder, last step |", "|---|---|"]
        for n in list(SUMMARY_KEYS) + L.class_order(list(last["own"]["classes"])):
            v = last["own"]["summary"].get(n) if n in SUMMARY_KEYS else last["own"]["classes"].get(n)
            label = f"**{n}**" if n in SUMMARY_KEYS else n + (" (low ceiling)" if n in L.LOW_CEILING else "")
            lines.append(f"| {label} | {_f(v)} |")
    if "sensitivity" in last:
        sens = last["sensitivity"]
        lines += ["", "## What the planner's distance sees (last step)", "",
                  "How far the raw latent moves when one factor of a val state is edited (median), in units of the median "
                  f"distance between frames one level-2 step (k = {sens.get('k', cfg.k)}) apart ({_f(sens['unit'])}).", "",
                  "| edit | distance in steps |", "|---|---|"]  # fmt: skip
        lines += [f"| {name} | {_f(v, 2)} |" for name, v in sens["edits"].items()]

    lines += ["", "## 2. Prediction (last step)", ""]
    if last.get("prediction"):
        hs = list(last["prediction"])
        lines += ["Error of the decoded rollout over the error of copying the last context frame (below 1 beats copying); "
                  "in brackets the error of decoding the true latent.", ""]  # fmt: skip
        head = ["class"] + [f"{h}" for h in hs]
        lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        cls = last["prediction"][hs[0]]["classes"]
        for n in L.class_order(list(cls)):
            row = [n]
            for h in hs:
                c = last["prediction"][h]["classes"].get(n)
                row.append(f"{_f(c['pred_over_copy'], 2)} ({_f(c['err_decoded_true'], 2)})" if c else "-")
            lines.append("| " + " | ".join(row) + " |")
        lines += ["", "Latent error over copying (median): " + ", ".join(
            f"{h} {_f(last['prediction'][h]['latent_pred_over_copy'])}" for h in hs) + "."]  # fmt: skip
    else:
        lines.append("No predictor (E0).")
    return _write(run_dir, lines, cfg)


def resolution_table(runs: dict, part: str = "holds", step: str | None = None) -> str:
    """A markdown table from `<run>/resolution.json`, one row per run ({label: run_dir}), at `step` (default: the last):
    R^2 of `part` (holds, foresees, copy) per group at tile / quadrant / view, the items, vitals and facing, and the
    raw latent's participation ratio."""
    groups = ("terrain", "ores and stations", "mobs")
    head = ["run", "step", *[f"{g} (tile / quadrant / view)" for g in groups], "items", "vitals", "facing", "PR"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for label, run_dir in runs.items():
        path = Path(run_dir) / "resolution.json"
        if not path.exists():
            continue
        res = json.loads(path.read_text())
        s = step if step is not None else max(res, key=int)
        r = res.get(s, {}).get(part)
        if r is None:
            continue
        cells = [" / ".join(_f(r[f"{g}|{x}"], 2) for x in ("tile", "quadrant", "view")) for g in groups]
        cells += [_f(r[f"{n}|-"], 2) for n in ("items", "vitals", "facing")]
        lines.append(f"| {label} | {s} | " + " | ".join(cells) + f" | {_f(res[s]['latent']['participation_ratio'], 1)} |")
    return "\n".join(lines) + "\n"


def _write(run_dir, lines, cfg) -> str:
    lines += ["", "## What this does not show", "", f"- {NOT_SHOWN[cfg.variant]}", *[f"- {c}" for c in COMMON]]
    text = "\n".join(lines) + "\n"
    (Path(run_dir) / "report.md").write_text(text)
    return text


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    print(write_report(p.parse_args(argv).run))


if __name__ == "__main__":
    main()

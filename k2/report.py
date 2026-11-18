"""The markdown report of a run, read against its control and with what it does not show.

    python -m k2.report runs/E2_k1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import layout as L
from .train import load_config, read_log

ROLE = {
    "E0": "capacity check: can this encoder hold every factor, rare ones included? (autoencoder only)",
    "E1": "pruning reference: prediction with the gradient through the target, + SIGReg (A3 analogue)",
    "E2": "E1 with a detached target (A4 analogue)",
    "E3": "E2 until the predictor plateaus on every latent dimension, then the target's gradient ramps up to E1's",
}
NOT_SHOWN = {
    "E0": "E0 has no predictor: it says nothing about what a predictive loss keeps, only what this encoder can hold "
    "under reconstruction. Its scores are the reference for the other variants, not a pass mark.",
    "E1": "One SIGReg weight (0.1). A loss of content here can come from the target path or from its interplay with "
    "that weight; E2 separates only the first.",
    "E2": "One SIGReg weight (0.1). Keeping content with a detached target does not mean it is kept for the right "
    "reason: the kept-against-predictable table says whether what is kept is predictable.",
    "E3": "One plateau rule (window, tolerance) and one ramp length. A plateau of the training error on every dimension "
    "does not mean the latent holds everything it should; and if the ramp starts late, the steps after it may be too "
    "few to show whether the target path then prunes as in E1.",
}
COMMON = [
    "One seed per run.",
    "Held-out frames are the val worlds (all sources); test is kept for the planning headline.",
    "The fresh decoder (2000 steps on 65k train frames) gives a lower bound on what the latent holds.",
    "The oracle's cut is read from the data (the least-populated level of p in [0.5, 1)), not fixed in advance.",
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
    if cfg.variant == "E3":
        ramp = run_dir / "ramp.json"
        rule = f"every dimension improving by less than {cfg.plateau_tol} over {cfg.plateau_window} steps"
        if ramp.exists():
            r = json.loads(ramp.read_text())
            lines += ["", f"Target-gradient ramp: the predictor plateaued ({rule}) at step {r['trigger']}; the gradient "
                      f"through the target rises from 0 to 1 until step {r['full']}."]  # fmt: skip
        else:
            lines += ["", f"Target-gradient ramp: never started ({rule}): this run is E2."]
    if (run_dir / "log.csv").exists():
        log = read_log(run_dir)
        keys = [k for k in ("loss", "pred_over_copy", "recon", "sigreg", "target_grad", "plateau_share", "z_std") if k in log]  # fmt: skip
        lines += ["", "Training, last logged step " + str(int(log["step"][-1])) + ": "
                  + ", ".join(f"{k} {_f(log[k][-1])}" for k in keys) + "."]  # fmt: skip
    if not steps:
        return _write(run_dir, lines + ["", "No readouts yet."], cfg)
    last = res[steps[-1]]

    lines += ["", "## 1. What the latent keeps (fresh decoder, val)", ""]
    head = ["class"] + [f"step {s}" for s in steps] + ([ctl_name] if ctl_last else [])
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    names = list(SUMMARY_KEYS) + L.class_order(list(last["fresh"]["classes"]))
    for n in names:
        get = lambda r: r["fresh"]["summary"].get(n) if n in SUMMARY_KEYS else r["fresh"]["classes"].get(n)  # noqa: E731
        row = [f"**{n}**" if n in SUMMARY_KEYS else n] + [_f(get(res[s])) for s in steps]
        row += [_f(get(ctl_last))] if ctl_last else []
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "Linear probe, last step: " + ", ".join(f"{k} {_f(last['linear']['summary'][k])}" for k in SUMMARY_KEYS) + "."]

    lines += ["", "## 2. Kept against predictable (last step)", ""]
    if last.get("kept"):
        lines += ["Share of the predictable factors kept / share of the unpredictable factors kept (counts in brackets).", ""]
        head = ["class", "predictable kept", "unpredictable kept"] + ([f"{ctl_name}: pred / unpred"] if ctl_last and ctl_last.get("kept") else [])
        lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        kept = last["kept"]
        for n in ["all"] + L.class_order([c for c in kept if c != "all"]):
            k = kept[n]
            row = [n, f"{_f(k['share_pred_kept'])} ({k['pred_kept'] + k['pred_dropped']})",
                   f"{_f(k['share_unpred_kept'])} ({k['unpred_kept'] + k['unpred_dropped']})"]  # fmt: skip
            if len(head) == 4:
                c = ctl_last["kept"].get(n)
                row.append(f"{_f(c['share_pred_kept'])} / {_f(c['share_unpred_kept'])}" if c else "-")
            lines.append("| " + " | ".join(row) + " |")
        lines += ["", "Over the checkpoints (all factors): " + ", ".join(
            f"step {s} {_f(res[s]['kept']['all']['share_pred_kept'])} / {_f(res[s]['kept']['all']['share_unpred_kept'])}"
            for s in steps if res[s].get("kept")) + "."]  # fmt: skip
    else:
        lines.append(f"No oracle table for k = {cfg.k} (`python -m k2.oracle --k {cfg.k}` on the Mac).")

    lines += ["", "## 3. Prediction (last step)", ""]
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

    lines += ["", "## 4. Geometry along expert paths", "", "| step | Spearman (distance, steps to go) | closer share |", "|---|---|---|"]
    lines += [f"| {s} | {_f(res[s]['geometry']['spearman'])} | {_f(res[s]['geometry']['closer_share'])} |" for s in steps]
    if ctl_last:
        g = ctl_last["geometry"]
        lines.append(f"| {ctl_name} | {_f(g['spearman'])} | {_f(g['closer_share'])} |")

    plan = run_dir / "planning.json"
    if plan.exists():
        lines += ["", "## 5. Planning", "", *planning_lines(json.loads(plan.read_text()))]
    return _write(run_dir, lines, cfg)


def planning_lines(p: dict) -> list[str]:
    lines = ["| method | success | steps used | subgoals reached |", "|---|---|---|---|"]
    for name, r in p["methods"].items():
        lines.append(f"| {name} | {_f(r['success'])} ({r['tasks']} tasks) | {_f(r['steps'], 1)} | {_f(r.get('subgoal_reached'))} |")
    return lines + ["", f"Tasks: {p['split']}, goal {p['offset']} steps ahead, budget {p['budget']} steps."]


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

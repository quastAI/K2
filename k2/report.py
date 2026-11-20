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
    "E3": "the Grace model: stage 1 (error-weighted reconstruction + detached prediction, no SIGReg) until the switch, "
    "then stage 2 (decoded prediction + prediction-gated reconstruction + detached prediction): keep what the predictor "
    "foresees, release the rest",
}
NOT_SHOWN = {
    "E0": "E0 has no predictor: it says nothing about what a predictive loss keeps, only what this encoder can hold "
    "under reconstruction. Its scores are the reference for the other variants, not a pass mark.",
    "E1": "One SIGReg weight. At 0.0225 (AAC's 0.09 scaled to batch 512) SIGReg's encoder gradient was 7 to 2000 times "
    "the prediction's (`grad_rest_over_pred`), so the target path E2 removes hardly acted; a comparison with E2 means "
    "something only at a weight where the two balance (Q2).",
    "E2": "One SIGReg weight (as E1, the same caveat). Keeping content with a detached target does not mean it is kept "
    "for the right reason: the kept-against-target table says whether what is kept is determined or needed.",
    "E3": "One switch step and one gate tolerance (tau, in nats per token), both heuristics; the stage-1 length is not "
    "tuned. The gate releases what the predictor cannot foresee now, including goal factors it has not learned yet. "
    "Stage 2 has no term on the latent's scale (z_std grew from 0.25 to 5.3 in the first run). The checkpoints "
    "before the switch are stage 1, the ones after it stage 2: read them as the two stages, not as one curve.",
}
COMMON = [
    "One seed per run.",
    "Held-out frames are the val worlds (all sources); test is kept for the planning headline.",
    "The readouts read z whitened on their train frames; the fresh decoder (2000 steps on 65k train frames) gives a "
    "lower bound on what the latent holds. Classes the readout cannot read even from the raw state (`(low ceiling)`) "
    "are shown but left out of the summaries.",
    "The target is D ∪ G: determined (the oracle's replays agree, at three lines: all agreed, 0.9, 0.75) or a goal "
    "factor (block types, items, vitals, sleeping). All agreed is noisy at its edge: a factor that deviates in a few "
    "percent of the replays can still pass. Each tile is two factors, its block and its mobs; outside the target are "
    "mostly the mob factors.",
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
        keys = [k for k in ("loss", "pred_over_copy", "recon", "sigreg", "grad_rest_over_pred", "z_std") if k in log]
        lines += ["", "Training, last logged step " + str(int(log["step"][-1])) + ": "
                  + ", ".join(f"{k} {_f(log[k][-1])}" for k in keys) + "."]  # fmt: skip
    if not steps:
        return _write(run_dir, lines + ["", "No readouts yet."], cfg)
    last = res[steps[-1]]

    lines += ["", "## 1. What the latent keeps (fresh decoder on whitened z, val)", ""]
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
    lines += ["", "The latent (participation ratio, rank, median slowness over one level-2 step): " + ", ".join(
        f"step {s} {_f(res[s]['latent']['participation_ratio'], 1)}, {res[s]['latent']['rank']}, "
        f"{_f(res[s]['latent']['slowness_median'])}" for s in steps if "latent" in res[s]) + "."]  # fmt: skip

    lines += ["", "## 2. Kept against the target D ∪ G (last step)", ""]
    if last.get("kept"):
        lines += kept_lines(last["kept"], ctl_last["kept"] if ctl_last and ctl_last.get("kept") else None, ctl_name)
        lines += ["", "Over the checkpoints (all changing factors, all agreed): " + ", ".join(
            f"step {s} {_pair(res[s]['kept']['all']['changing']['by_line']['1'])}" for s in steps if res[s].get("kept")) + "."]  # fmt: skip
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


LINE_NAMES = {"1": "all agreed", "0.9": "p >= 0.9", "0.75": "p >= 0.75"}


def _share(kept: int, n: int) -> str:
    return "-" if n == 0 else f"{kept / n:.2f} ({n})"


def _pair(c: dict) -> str:
    """Share of the target factors kept / of the other ones kept, counts in brackets."""
    t, o = c["target_kept"], c["other_kept"]
    return f"{_share(t, t + c['target_dropped'])} / {_share(o, o + c['other_dropped'])}"


def kept_lines(kept: dict, ctl: dict | None, ctl_name: str | None) -> list[str]:
    """Kept against the target per kind and per class: the unchanged factors kept, the changing ones kept, and per line
    of the oracle the share of the target / other changing factors kept (`*`: fewer than 30 changing)."""
    from .oracle import FEW

    lines = [
        f"Oracle of {kept['n_keys']} fresh-key replays. Changing = some replay ends elsewhere than the start frame; "
        "the unchanged factors are determined and listed apart (kept share, count). For the changing ones, per line: "
        "target kept / other kept, counts in brackets. Target = determined at that line (the main line is all replays "
        f"agreed) or a goal factor. `*`: fewer than {FEW} changing factors, read with care.",
    ]
    for title, key, order in (("kind", "kinds", list(L.KINDS)), ("class", "classes", None)):
        group = kept[key]
        names = [n for n in order if n in group] if order else L.class_order(list(group))
        names = ["all"] + names if key == "kinds" else names
        get = lambda d, n: d["all"] if n == "all" else d[key][n]  # noqa: E731
        head = [title, "unchanged kept", "changing kept"] + [LINE_NAMES[x] for x in kept["lines"]]
        if ctl:
            head.append(f"{ctl_name}, all agreed")
        lines += ["", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for n in names:
            g = get(kept, n)
            row = [n + (" *" if g["changing"]["n"] < FEW else ""), _share(g["unchanged"]["kept"], g["unchanged"]["n"]),
                   _share(g["changing"]["kept"], g["changing"]["n"])]  # fmt: skip
            row += [_pair(g["changing"]["by_line"][x]) for x in kept["lines"]]
            if ctl:
                c = get(ctl, n) if n == "all" or n in ctl[key] else None
                row.append(_pair(c["changing"]["by_line"]["1"]) if c else "-")
            lines.append("| " + " | ".join(row) + " |")
    return lines


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

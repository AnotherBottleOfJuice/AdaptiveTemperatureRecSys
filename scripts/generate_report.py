#!/usr/bin/env python3
"""Generate the markdown experiment report (the four tables) from mlruns/mlflow.db.

Tables produced, given an experiment name:
  1. Windowed stats   - max metric per epoch window (default 10), all metrics.
  2. Final comparison - last window, Recall/NDCG/Hitrate with Delta% vs the
                        sibling run that shares everything except the learning rate.
  3. Prefix max (fine)- running-max recall at several epoch checkpoints, Delta% vs
                        the previous checkpoint (default step 5).
  4. Prefix max (E15) - running-max recall at E<=15 and E<=30, Delta% between them.

Usage:
    python scripts/generate_report.py linear_tau_30e
    python scripts/generate_report.py constant_tau_30e --db mlruns/mlflow.db
    python scripts/generate_report.py linear_tau_30e --window 10 --fine-steps 5,10,15,20,25,30
"""
import argparse
import ast
import sqlite3
from collections import defaultdict

import numpy as np

METRICS = ["valid/recall", "valid/ndcg", "valid/hitrate", "valid/coverage"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", help="MLflow experiment name, e.g. linear_tau_30e")
    p.add_argument("--db", default="mlruns/mlflow.db", help="path to mlflow.db")
    p.add_argument("--window", type=int, default=10,
                   help="epoch window size for the windowed table (default 10)")
    p.add_argument("--fine-steps", default="5,10,15,20,25,30",
                   help="comma epoch checkpoints for the fine prefix-max table")
    p.add_argument("--coarse-steps", default="15,30",
                   help="comma epoch checkpoints for the coarse prefix-max table")
    p.add_argument("--baseline-exp", default="constant_tau_30e",
                   help="experiment holding the baseline run (default constant_tau_30e)")
    p.add_argument("--baseline", default="tau=0.045",
                   help="baseline config label, e.g. 'tau=0.045' or 'min=0.035, max=0.06'")
    p.add_argument("--baseline-lr", default="0.003",
                   help="baseline learning rate (default 0.003)")
    return p.parse_args()


def load_experiment(conn, name):
    """Return {run_uuid: {'params': {...}, 'curves': {metric: [v_per_epoch]}}}."""
    c = conn.cursor()
    c.execute("SELECT experiment_id FROM experiments WHERE name=?", (name,))
    row = c.fetchone()
    if row is None:
        names = [r[0] for r in c.execute("SELECT name FROM experiments").fetchall()]
        raise SystemExit(f"Experiment '{name}' not found. Available: {names}")
    exp_id = row[0]

    c.execute(
        "SELECT run_uuid FROM runs WHERE experiment_id=? AND lifecycle_stage='active' "
        "ORDER BY start_time", (exp_id,))
    run_ids = [r[0] for r in c.fetchall()]

    runs = {}
    for ru in run_ids:
        params = dict(c.execute(
            "SELECT key, value FROM params WHERE run_uuid=?", (ru,)).fetchall())
        curves = {}
        for m in METRICS:
            vals = [v for (v,) in c.execute(
                "SELECT value FROM metrics WHERE run_uuid=? AND key=? ORDER BY step",
                (ru, m)).fetchall()]
            curves[m] = vals
        runs[ru] = {"params": params, "curves": curves}
    return runs


def config_label(params):
    """Human label for a config, independent of the learning rate.

    Returns (base_label, lr) so runs can be grouped and lr-compared.
    """
    lr = params.get("learning_rate", "?")
    tau_cls = params.get("tau_class_name", "")
    try:
        targs = ast.literal_eval(params.get("tau_json_args", "{}"))
    except (ValueError, SyntaxError):
        targs = {}

    if tau_cls == "LinearTau" or (targs.get("tau_min") is not None):
        base = f"min={targs.get('tau_min')}, max={targs.get('tau_max')}"
    elif tau_cls == "ConstantTau" or (targs.get("initial_tau") is not None):
        base = f"tau={targs.get('initial_tau')}"
    else:
        base = params.get("tau_json_args", tau_cls or "config")
    return base, lr


def group_runs(runs):
    """{(base_label, lr): [curve_dicts]}, preserving first-seen order of bases."""
    groups = defaultdict(list)
    base_order, lr_order = [], []
    for r in runs.values():
        base, lr = config_label(r["params"])
        groups[(base, lr)].append(r["curves"])
        if base not in base_order:
            base_order.append(base)
        if lr not in lr_order:
            lr_order.append(lr)
    return groups, base_order, sorted(lr_order)


def agg(curves_list, metric, lo, hi):
    """mean,std of the max-in-window [lo:hi] across seeds."""
    vals = [max(c[metric][lo:hi]) for c in curves_list if c[metric][lo:hi]]
    return (np.mean(vals), np.std(vals)) if vals else (float("nan"), float("nan"))


def prefix_agg(curves_list, metric, upto):
    vals = [max(c[metric][:upto]) for c in curves_list if c[metric][:upto]]
    return (np.mean(vals), np.std(vals)) if vals else (float("nan"), float("nan"))


def window_mean_agg(curves_list, metric, lo, hi):
    """mean,std of the mean-in-window [lo:hi] across seeds."""
    vals = [np.mean(c[metric][lo:hi]) for c in curves_list if c[metric][lo:hi]]
    return (np.mean(vals), np.std(vals)) if vals else (float("nan"), float("nan"))


def fmt(m, s):
    return f"{m:.4f}±{s:.4f}"


def last_window_finals(curves_list, metrics, window):
    """{metric: (mean, std)} over the last `window` epochs (max-in-window)."""
    ne = max_epochs(curves_list)
    lo = max(0, ne - window)
    return {m: agg(curves_list, m, lo, ne) for m in metrics}


def max_epochs(curves_list):
    return max((len(c["valid/recall"]) for c in curves_list), default=0)


# ---------------------------------------------------------------- table 1
def table_windowed(groups, base_order, lr_order, window):
    out = ["## Windowed stats (max in window, mean±std over seeds)", ""]
    out.append("| Config | n | Part | Recall | NDCG | Hitrate | Coverage |")
    out.append("|---|---|---|---|---|---|---|")
    for base in base_order:
        for lr in lr_order:
            cl = groups.get((base, lr))
            if not cl:
                continue
            n = len(cl)
            ne = max_epochs(cl)
            bounds = [(i, min(i + window, ne)) for i in range(0, ne, window)]
            for j, (lo, hi) in enumerate(bounds):
                cfg = f"lr={lr}, {base}" if j == 0 else ""
                ncol = str(n) if j == 0 else ""
                part = f"E{lo + 1}–{hi}"
                cells = " | ".join(fmt(*agg(cl, m, lo, hi)) for m in METRICS)
                out.append(f"| {cfg} | {ncol} | {part} | {cells} |")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------- table 2
def table_final_comparison(groups, base_order, lr_order, window):
    out = ["## Final results (last window, Δ% vs sibling lr)", ""]
    out.append("| Config | lr | Recall | NDCG | Hitrate |")
    out.append("|---|---|---|---|---|")
    metrics3 = ["valid/recall", "valid/ndcg", "valid/hitrate"]
    for base in base_order:
        present_lrs = [lr for lr in lr_order if (base, lr) in groups]
        # mean of last window per (lr, metric) for delta + bolding
        last = {}
        for lr in present_lrs:
            cl = groups[(base, lr)]
            ne = max_epochs(cl)
            lo = max(0, ne - window)
            last[lr] = {m: agg(cl, m, lo, ne) for m in metrics3}
        best_lr = max(present_lrs, key=lambda lr: last[lr]["valid/recall"][0]) \
            if present_lrs else None
        for j, lr in enumerate(present_lrs):
            cfg = base if j == 0 else ""
            cells = []
            for m in metrics3:
                mean, std = last[lr][m]
                others = [last[o][m][0] for o in present_lrs if o != lr]
                txt = fmt(mean, std)
                if others:
                    ref = np.mean(others)
                    pct = (mean - ref) / ref * 100
                    txt += f" ({'+' if pct >= 0 else ''}{pct:.1f}%)"
                if m == "valid/recall" and lr == best_lr and len(present_lrs) > 1:
                    txt = f"**{txt}**"
                cells.append(txt)
            out.append(f"| {cfg} | {lr} | " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------- table vs baseline
def load_baseline(conn, exp_name, base_label, lr, window):
    """Return ({metric: (mean, std)}, description) for the baseline run, or None."""
    runs = load_experiment(conn, exp_name)
    groups, _, _ = group_runs(runs)
    cl = groups.get((base_label, lr))
    if not cl:
        avail = sorted({f"lr={b_lr}, {b}" for (b, b_lr) in groups})
        raise SystemExit(
            f"Baseline 'lr={lr}, {base_label}' not found in '{exp_name}'. "
            f"Available: {avail}")
    finals = last_window_finals(cl, ["valid/recall", "valid/ndcg", "valid/hitrate"],
                                window)
    return finals, f"{exp_name}: lr={lr}, {base_label} (n={len(cl)})"


def table_vs_baseline(groups, base_order, lr_order, baseline, baseline_desc, window):
    metrics3 = ["valid/recall", "valid/ndcg", "valid/hitrate"]
    out = [f"## Final vs baseline — {baseline_desc}", ""]
    out.append("| Config | lr | Recall (Δ%) | NDCG (Δ%) | Hitrate (Δ%) |")
    out.append("|---|---|---|---|---|")
    for base in base_order:
        for lr in lr_order:
            cl = groups.get((base, lr))
            if not cl:
                continue
            finals = last_window_finals(cl, metrics3, window)
            cells = []
            for m in metrics3:
                mean, std = finals[m]
                ref = baseline[m][0]
                pct = (mean - ref) / ref * 100 if ref else float("nan")
                txt = f"{fmt(mean, std)} ({'+' if pct >= 0 else ''}{pct:.1f}%)"
                if m == "valid/recall" and mean > ref:
                    txt = f"**{txt}**"
                cells.append(txt)
            out.append(f"| {base} | {lr} | " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------- tables 3 & 4
def table_prefix(groups, base_order, lr_order, steps, title):
    out = [f"## {title}", ""]
    head = ["lr", "config"] + [
        (f"E≤{s}" if i == 0 else f"E≤{s} (Δ%)") for i, s in enumerate(steps)]
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "---|" * len(head))
    for base in base_order:
        for lr in lr_order:
            cl = groups.get((base, lr))
            if not cl:
                continue
            cells, prev = [], None
            for i, s in enumerate(steps):
                mean, std = prefix_agg(cl, "valid/recall", s)
                if i == 0 or prev is None or prev == 0:
                    cells.append(fmt(mean, std))
                else:
                    pct = (mean - prev) / prev * 100
                    cells.append(f"{fmt(mean, std)} ({'+' if pct >= 0 else ''}{pct:.1f}%)")
                prev = mean
            out.append(f"| {lr} | {base} | " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------- window-mean table
def table_window_mean(groups, base_order, lr_order, steps, title):
    """Like table_prefix, but each cell is the mean recall inside the window
    between consecutive checkpoints (not a running prefix max)."""
    bounds, prev = [], 0
    for s in steps:
        bounds.append((prev, s))
        prev = s
    out = [f"## {title}", ""]
    head = ["lr", "config"] + [
        (f"E{lo + 1}–{hi}" if i == 0 else f"E{lo + 1}–{hi} (Δ%)")
        for i, (lo, hi) in enumerate(bounds)]
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "---|" * len(head))
    for base in base_order:
        for lr in lr_order:
            cl = groups.get((base, lr))
            if not cl:
                continue
            cells, prev_mean = [], None
            for i, (lo, hi) in enumerate(bounds):
                mean, std = window_mean_agg(cl, "valid/recall", lo, hi)
                if i == 0 or prev_mean is None or prev_mean == 0:
                    cells.append(fmt(mean, std))
                else:
                    pct = (mean - prev_mean) / prev_mean * 100
                    cells.append(f"{fmt(mean, std)} ({'+' if pct >= 0 else ''}{pct:.1f}%)")
                prev_mean = mean
            out.append(f"| {lr} | {base} | " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out)


def main():
    args = parse_args()
    fine = [int(x) for x in args.fine_steps.split(",")]
    coarse = [int(x) for x in args.coarse_steps.split(",")]

    conn = sqlite3.connect(args.db)
    runs = load_experiment(conn, args.experiment)
    if not runs:
        raise SystemExit(f"No active runs in experiment '{args.experiment}'.")
    groups, base_order, lr_order = group_runs(runs)

    print(f"# {args.experiment}\n")
    print(table_windowed(groups, base_order, lr_order, args.window))
    print(table_final_comparison(groups, base_order, lr_order, args.window))
    baseline, baseline_desc = load_baseline(
        conn, args.baseline_exp, args.baseline, args.baseline_lr, args.window)
    print(table_vs_baseline(groups, base_order, lr_order,
                            baseline, baseline_desc, args.window))
    print(table_prefix(groups, base_order, lr_order, fine,
                       "Prefix max recall, fine steps (Δ% vs previous)"))
    print(table_window_mean(groups, base_order, lr_order, fine,
                            "Window mean recall, fine steps (Δ% vs previous)"))
    print(table_prefix(groups, base_order, lr_order, coarse,
                       "Prefix max recall, coarse steps (Δ% vs previous)"))


if __name__ == "__main__":
    main()

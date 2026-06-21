#!/usr/bin/env python3
"""Generate the markdown experiment report (the four tables) from mlruns/mlflow.db.

Tables produced, given one or more experiment names:
  1. Windowed stats   - max metric per epoch window (default 10), all metrics.
  2. Final results    - last window, Recall/NDCG/Hitrate per config (no comparison).
  3. Prefix max (fine)- running-max recall at several epoch checkpoints, Delta% vs
                        the previous checkpoint (default step 5).
  4. Prefix max (E15) - running-max recall at E<=15 and E<=30, Delta% between them.

Usage:
    python scripts/generate_report.py linear_tau_30e
    python scripts/generate_report.py constant_tau_30e --db mlruns/mlflow.db
    # pool several experiments and keep only one dataset:
    python scripts/generate_report.py constant_tau_30e constant_tau_30e_part2 \
        constant_tau_30e_part3 --dataset Amazon
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
    p.add_argument("experiment", nargs="+",
                   help="One or more MLflow experiment names to pool, e.g. "
                        "constant_tau_30e constant_tau_30e_part2 constant_tau_30e_part3")
    p.add_argument("--db", default="mlruns/mlflow.db", help="path to mlflow.db")
    p.add_argument("--dataset", default=None,
                   help="keep only runs whose data_dataset param contains this "
                        "substring, e.g. 'Amazon' (default: keep all)")
    p.add_argument("--window", type=int, default=10,
                   help="epoch window size for the windowed table (default 10)")
    p.add_argument("--fine-steps", default="5,10,15,20,25,30",
                   help="comma epoch checkpoints for the fine prefix-max table")
    p.add_argument("--coarse-steps", default="15,30",
                   help="comma epoch checkpoints for the coarse prefix-max table")
    return p.parse_args()


def load_experiment(conn, name, dataset=None):
    """Return {run_uuid: {'params': {...}, 'curves': {metric: [v_per_epoch]}}}.

    When `dataset` is given, runs whose `data_dataset` param does not contain that
    substring are dropped (used to keep only e.g. Amazon runs from a mixed experiment).
    """
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
        if dataset and dataset not in params.get("data_dataset", ""):
            continue
        curves = {}
        for m in METRICS:
            vals = [v for (v,) in c.execute(
                "SELECT value FROM metrics WHERE run_uuid=? AND key=? ORDER BY step",
                (ru, m)).fetchall()]
            curves[m] = vals
        runs[ru] = {"params": params, "curves": curves}
    return runs


def load_experiments(conn, names, dataset=None):
    """Pool runs from several experiments into one {run_uuid: ...} dict.

    Runs are keyed by run_uuid, so the report groups them purely by config
    (tau/lr) regardless of which experiment/part they came from.
    """
    runs = {}
    for name in names:
        runs.update(load_experiment(conn, name, dataset))
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

    if tau_cls == "MACLossTau":
        base = (f"bt={targs.get('base_threshold')}, "
                f"lc={targs.get('linear_coeff')}")
    elif tau_cls == "LinearTau" or (targs.get("tau_min") is not None):
        base = f"min={targs.get('tau_min')}, max={targs.get('tau_max')}"
        # shift/scale (ShiftedCosPerUserTau & friends) are part of the config
        # identity: without them every shift/scale combo collapses into one group.
        if targs.get("shift") is not None:
            base += f", shift={targs.get('shift')}"
        if targs.get("scale") is not None:
            base += f", scale={targs.get('scale')}"
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


def epochs_to_max_agg(curves_list, metric):
    """mean,std of the (1-indexed) epoch at which `metric` peaks, across seeds."""
    vals = [int(np.argmax(c[metric])) + 1 for c in curves_list if c[metric]]
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
                cfg = f"lr={lr}, {base}"
                ncol = str(n)
                part = f"E{lo + 1}–{hi}"
                cells = " | ".join(fmt(*agg(cl, m, lo, hi)) for m in METRICS)
                out.append(f"| {cfg} | {ncol} | {part} | {cells} |")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------- table 2
def table_final_comparison(groups, base_order, lr_order, window):
    out = ["## Final results (max over whole run, mean±std over seeds)", ""]
    out.append("| Config | lr | Recall | NDCG | Hitrate |")
    out.append("|---|---|---|---|---|")
    metrics3 = ["valid/recall", "valid/ndcg", "valid/hitrate"]
    for base in base_order:
        for lr in lr_order:
            cl = groups.get((base, lr))
            if not cl:
                continue
            ne = max_epochs(cl)
            cells = " | ".join(fmt(*agg(cl, m, 0, ne)) for m in metrics3)
            out.append(f"| {base} | {lr} | {cells} |")
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


# ---------------------------------------------------------------- epochs-to-max table
def table_epochs_to_max(groups, base_order, lr_order):
    """Mean (over seeds) number of epochs needed to reach the peak Recall / NDCG."""
    out = ["## Epochs to reach max (mean±std over seeds)", ""]
    out.append("| Config | lr | n | Epochs→max Recall | Epochs→max NDCG |")
    out.append("|---|---|---|---|---|")
    for base in base_order:
        for lr in lr_order:
            cl = groups.get((base, lr))
            if not cl:
                continue
            n = len(cl)
            rec = epochs_to_max_agg(cl, "valid/recall")
            ndcg = epochs_to_max_agg(cl, "valid/ndcg")
            out.append(
                f"| {base} | {lr} | {n} | {rec[0]:.1f}±{rec[1]:.1f} | "
                f"{ndcg[0]:.1f}±{ndcg[1]:.1f} |")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------- scale × shift
def tau_params(params):
    try:
        return ast.literal_eval(params.get("tau_json_args", "{}"))
    except (ValueError, SyntaxError):
        return {}


def group_shift_scale(runs):
    """{(tau_min, tau_max, lr): {(shift, scale): [curve_dicts]}}.

    Each leaf list holds the per-seed curves for one exact config, so the only
    aggregation that ever happens is across seeds — never across shift/scale/lr.
    Returns ({}, [], []) when the experiment has no shift/scale params.
    """
    groups = defaultdict(lambda: defaultdict(list))
    shifts, scales = set(), set()
    for r in runs.values():
        t = tau_params(r["params"])
        shift, scale = t.get("shift"), t.get("scale")
        if shift is None or scale is None:
            continue
        lr = r["params"].get("learning_rate", "?")
        groups[(t.get("tau_min"), t.get("tau_max"), lr)][(shift, scale)].append(
            r["curves"])
        shifts.add(shift)
        scales.add(scale)
    return groups, sorted(shifts), sorted(scales)


def table_shift_scale_pivot(runs, window, metric="valid/recall"):
    """Pivot of (mean±std over seeds) per (shift, scale), one block per min/max/lr.

    Reading down a column compares shifts at a fixed scale; reading across a row
    compares scales at a fixed shift. Every cell is a single config (only seeds
    pooled), so nothing is grouped by anything other than seed.
    """
    groups, shifts, scales = group_shift_scale(runs)
    if not groups:
        return ""
    name = metric.split("/")[-1].capitalize()
    out = [f"## {name} by shift × scale (mean±std over seeds)", ""]
    for (tmin, tmax, lr) in sorted(groups):
        cells = groups[(tmin, tmax, lr)]
        out.append(f"### min={tmin}, max={tmax}, lr={lr}")
        out.append("")
        out.append("| shift \\ scale | " + " | ".join(f"{s:g}" for s in scales) + " |")
        out.append("|" + "---|" * (len(scales) + 1))
        for shift in shifts:
            row = [f"{shift:g}"]
            for scale in scales:
                cl = cells.get((shift, scale))
                if cl:
                    ne = max_epochs(cl)
                    row.append(fmt(*agg(cl, metric, max(0, ne - window), ne)))
                else:
                    row.append("—")
            out.append("| " + " | ".join(row) + " |")
        out.append("")
    return "\n".join(out)


def table_shift_scale_flat(runs, window):
    """One row per exact config (seeds pooled), sorted by Recall descending.

    Lets you eyeball the best scale/shift combos directly across all min/max.
    """
    groups, _, _ = group_shift_scale(runs)
    if not groups:
        return ""
    metrics3 = ["valid/recall", "valid/ndcg", "valid/hitrate"]
    rows = []
    for (tmin, tmax, lr), cells in groups.items():
        for (shift, scale), cl in cells.items():
            finals = last_window_finals(cl, metrics3, window)
            rows.append((tmin, tmax, lr, shift, scale, len(cl), finals))
    rows.sort(key=lambda r: r[6]["valid/recall"][0], reverse=True)

    out = ["## All configs by scale & shift (last window, sorted by Recall)", ""]
    out.append("| min | max | shift | scale | lr | n | Recall | NDCG | Hitrate |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    best = rows[0][6]["valid/recall"][0] if rows else None
    for tmin, tmax, lr, shift, scale, n, finals in rows:
        recall = fmt(*finals["valid/recall"])
        if best is not None and finals["valid/recall"][0] == best:
            recall = f"**{recall}**"
        out.append(
            f"| {tmin} | {tmax} | {shift:g} | {scale:g} | {lr} | {n} | "
            f"{recall} | {fmt(*finals['valid/ndcg'])} | {fmt(*finals['valid/hitrate'])} |")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------- base × linear (MACLoss)
def group_base_linear(runs):
    """{(tau_min, tau_max, lr): {(base_threshold, linear_coeff): [curve_dicts]}}.

    MACLossTau sweeps base_threshold × linear_coeff; each leaf list holds the
    per-seed curves for one exact config, so seeds are the only thing pooled.
    Returns ({}, [], []) when no MACLossTau runs carry both params.
    """
    groups = defaultdict(lambda: defaultdict(list))
    thresholds, coeffs = set(), set()
    for r in runs.values():
        if r["params"].get("tau_class_name") != "MACLossTau":
            continue
        t = tau_params(r["params"])
        bt, lc = t.get("base_threshold"), t.get("linear_coeff")
        if bt is None or lc is None:
            continue
        lr = r["params"].get("learning_rate", "?")
        groups[(t.get("tau_min"), t.get("tau_max"), lr)][(bt, lc)].append(
            r["curves"])
        thresholds.add(bt)
        coeffs.add(lc)
    return groups, sorted(thresholds), sorted(coeffs)


def table_base_linear_pivot(runs, window, metric="valid/recall"):
    """Pivot of (mean±std over seeds) per (base_threshold, linear_coeff).

    Rows are base_threshold, columns are linear_coeff, one block per min/max/lr.
    """
    groups, thresholds, coeffs = group_base_linear(runs)
    if not groups:
        return ""
    name = metric.split("/")[-1].capitalize()
    out = [f"## {name} by base_threshold × linear_coeff (mean±std over seeds)", ""]
    for (tmin, tmax, lr) in sorted(groups):
        cells = groups[(tmin, tmax, lr)]
        out.append(f"### min={tmin}, max={tmax}, lr={lr}")
        out.append("")
        out.append("| base_threshold \\ linear_coeff | "
                   + " | ".join(f"{c:g}" for c in coeffs) + " |")
        out.append("|" + "---|" * (len(coeffs) + 1))
        for bt in thresholds:
            row = [f"{bt:g}"]
            for lc in coeffs:
                cl = cells.get((bt, lc))
                if cl:
                    ne = max_epochs(cl)
                    row.append(fmt(*agg(cl, metric, max(0, ne - window), ne)))
                else:
                    row.append("—")
            out.append("| " + " | ".join(row) + " |")
        out.append("")
    return "\n".join(out)


def main():
    args = parse_args()
    fine = [int(x) for x in args.fine_steps.split(",")]
    coarse = [int(x) for x in args.coarse_steps.split(",")]

    conn = sqlite3.connect(args.db)
    runs = load_experiments(conn, args.experiment, args.dataset)
    if not runs:
        raise SystemExit(
            f"No active runs in {args.experiment}"
            + (f" for dataset '{args.dataset}'." if args.dataset else "."))
    groups, base_order, lr_order = group_runs(runs)

    title = " + ".join(args.experiment)
    if args.dataset:
        title += f" ({args.dataset})"
    print(f"# {title}\n")
    # Scale × shift comparison (only emitted when the runs carry shift/scale).
    shift_scale_pivot = table_shift_scale_pivot(runs, args.window)
    if shift_scale_pivot:
        print(shift_scale_pivot)
        print(table_shift_scale_flat(runs, args.window))
    # base_threshold × linear_coeff pivot (only emitted for MACLossTau runs).
    base_linear_pivot = table_base_linear_pivot(runs, args.window)
    if base_linear_pivot:
        print(base_linear_pivot)
    print(table_epochs_to_max(groups, base_order, lr_order))
    print(table_windowed(groups, base_order, lr_order, args.window))
    print(table_final_comparison(groups, base_order, lr_order, args.window))
    print(table_prefix(groups, base_order, lr_order, fine,
                       "Prefix max recall, fine steps (Δ% vs previous)"))
    print(table_window_mean(groups, base_order, lr_order, fine,
                            "Window mean recall, fine steps (Δ% vs previous)"))
    print(table_prefix(groups, base_order, lr_order, coarse,
                       "Prefix max recall, coarse steps (Δ% vs previous)"))


if __name__ == "__main__":
    main()

"""Compare MACLossTau sweep results on Amazon: recall vs base_threshold and linear_coeff.

Reads from mlruns/mlflow.db (macloss_tau_30e experiment).
Groups runs by (tau_min, tau_max, lr); within each group aggregates across seeds.

Produces two figures:
  plots/macloss_tau_amazon_recall_vs_bias.png
  plots/macloss_tau_amazon_recall_vs_linear_coeff.png
"""

import ast
import sqlite3
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MLFLOW_DB = Path("mlruns/mlflow.db")
EXPERIMENT = "macloss_tau_30e"
METRIC = "valid/recall"


# ---------------------------------------------------------------- data loading

def load_runs(db: Path, experiment: str):
    conn = sqlite3.connect(db)
    c = conn.cursor()
    exp_id = c.execute(
        "SELECT experiment_id FROM experiments WHERE name=?", (experiment,)
    ).fetchone()[0]
    run_ids = [
        r[0]
        for r in c.execute(
            "SELECT run_uuid FROM runs WHERE experiment_id=? AND lifecycle_stage='active' "
            "ORDER BY start_time",
            (exp_id,),
        ).fetchall()
    ]

    records = []
    for ru in run_ids:
        params = dict(
            c.execute("SELECT key, value FROM params WHERE run_uuid=?", (ru,)).fetchall()
        )
        targs = ast.literal_eval(params.get("tau_json_args", "{}"))
        values = [
            v
            for (v,) in c.execute(
                "SELECT value FROM metrics WHERE run_uuid=? AND key=? ORDER BY step",
                (ru, METRIC),
            ).fetchall()
        ]
        if not values:
            continue
        records.append(
            {
                "tau_min": targs.get("tau_min"),
                "tau_max": targs.get("tau_max"),
                "lr": float(params.get("learning_rate", 0)),
                "base_threshold": targs.get("base_threshold"),
                "linear_coeff": targs.get("linear_coeff"),
                "recall_max": max(values),
            }
        )
    conn.close()
    return records


def aggregate(records):
    """Group by (tau_min, tau_max, lr, base_threshold, linear_coeff), aggregate seeds."""
    buckets = defaultdict(list)
    for r in records:
        key = (r["tau_min"], r["tau_max"], r["lr"], r["base_threshold"], r["linear_coeff"])
        buckets[key].append(r["recall_max"])
    return {
        k: (float(np.mean(v)), float(np.std(v)), len(v))
        for k, v in buckets.items()
    }


# ---------------------------------------------------------------- plotting helpers

MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _group_label(tau_min, tau_max, lr):
    return f"τ=[{tau_min:.3g},{tau_max:.3g}]  lr={lr:.3g}"


def plot_recall_vs_x(agg_data, x_key, line_key, x_label, line_label, out_path):
    """Generic: x-axis = x_key param, separate lines = line_key param, panels = tau group."""
    # Collect all group (tau_min, tau_max, lr) tuples in data order
    groups = sorted({(k[0], k[1], k[2]) for k in agg_data})
    n = len(groups)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.8), sharey=False, squeeze=False)
    axes = axes[0]

    key_order = {"base_threshold": 3, "linear_coeff": 4}
    x_idx = key_order[x_key]
    line_idx = key_order[line_key]

    for ax, (tmin, tmax, lr) in zip(axes, groups):
        subset = {k: v for k, v in agg_data.items() if k[0] == tmin and k[1] == tmax and k[2] == lr}

        line_vals = sorted({k[line_idx] for k in subset})
        for i, lv in enumerate(line_vals):
            pts = sorted(
                [(k[x_idx], v) for k, v in subset.items() if k[line_idx] == lv],
                key=lambda t: t[0],
            )
            if not pts:
                continue
            xs, ys_mean, ys_std = zip(*[(x, m, s) for x, (m, s, _) in pts])
            xs = list(xs)
            ys_mean = list(ys_mean)
            ys_std = list(ys_std)
            color = COLORS[i % len(COLORS)]
            marker = MARKERS[i % len(MARKERS)]
            ax.errorbar(
                xs,
                ys_mean,
                yerr=ys_std,
                label=f"{line_label}={lv:g}",
                color=color,
                marker=marker,
                markersize=7,
                linewidth=1.8,
                capsize=4,
                capthick=1.5,
            )

        ax.set_title(_group_label(tmin, tmax, lr), fontsize=10)
        ax.set_xlabel(x_label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, framealpha=0.7)

    axes[0].set_ylabel(f"Recall@100 (max over run, mean±std across seeds)")
    fig.suptitle(
        f"MACLossTau on Amazon Beauty — Recall vs {x_label}",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    Path(out_path).parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    records = load_runs(MLFLOW_DB, EXPERIMENT)
    agg_data = aggregate(records)

    plot_recall_vs_x(
        agg_data,
        x_key="base_threshold",
        line_key="linear_coeff",
        x_label="base_threshold  (bias)",
        line_label="linear_coeff",
        out_path="plots/macloss_tau_amazon_recall_vs_bias.png",
    )

    plot_recall_vs_x(
        agg_data,
        x_key="linear_coeff",
        line_key="base_threshold",
        x_label="linear_coeff",
        line_label="base_threshold",
        out_path="plots/macloss_tau_amazon_recall_vs_linear_coeff.png",
    )


if __name__ == "__main__":
    main()

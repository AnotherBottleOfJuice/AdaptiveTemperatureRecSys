"""Compare CosPerUserTau sweep results on Amazon: recall vs tau_min and tau_max.

Reads from mlruns/mlflow.db (cos_per_user_tau_30e experiment).
Groups runs by lr; within each group aggregates across seeds.

Produces two figures:
  plots/cos_per_user_tau_amazon_recall_vs_tau_min.png
  plots/cos_per_user_tau_amazon_recall_vs_tau_max.png
"""

import ast
import sqlite3
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MLFLOW_DB = Path("mlruns/mlflow.db")
EXPERIMENT = "cos_per_user_tau_30e"
METRIC = "valid/recall"

MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def load_runs(db, experiment):
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
        vals = [
            v
            for (v,) in c.execute(
                "SELECT value FROM metrics WHERE run_uuid=? AND key=? ORDER BY step",
                (ru, METRIC),
            ).fetchall()
        ]
        if not vals:
            continue
        records.append(
            {
                "lr": float(params.get("learning_rate", 0)),
                "tau_min": targs.get("tau_min"),
                "tau_max": targs.get("tau_max"),
                "recall_max": max(vals),
            }
        )
    conn.close()
    return records


def aggregate(records):
    buckets = defaultdict(list)
    for r in records:
        buckets[(r["lr"], r["tau_min"], r["tau_max"])].append(r["recall_max"])
    return {
        k: (float(np.mean(v)), float(np.std(v)), len(v))
        for k, v in buckets.items()
    }


def plot_recall_vs_x(agg_data, x_key, line_key, x_label, line_label, out_path):
    """x-axis = x_key, separate lines = line_key, panels = lr."""
    key_idx = {"tau_min": 1, "tau_max": 2}
    xi, li = key_idx[x_key], key_idx[line_key]

    lrs = sorted({k[0] for k in agg_data})
    n = len(lrs)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.8), sharey=False, squeeze=False)
    axes = axes[0]

    for ax, lr in zip(axes, lrs):
        subset = {k: v for k, v in agg_data.items() if k[0] == lr}
        line_vals = sorted({k[li] for k in subset})

        for i, lv in enumerate(line_vals):
            pts = sorted(
                [(k[xi], v) for k, v in subset.items() if k[li] == lv],
                key=lambda t: t[0],
            )
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys_mean = [p[1][0] for p in pts]
            ys_std = [p[1][1] for p in pts]
            ax.errorbar(
                xs, ys_mean, yerr=ys_std,
                label=f"{line_label}={lv:.3g}",
                color=COLORS[i % len(COLORS)],
                marker=MARKERS[i % len(MARKERS)],
                markersize=7,
                linewidth=1.8,
                capsize=4,
                capthick=1.5,
            )

        ax.set_title(f"lr={lr:.3g}", fontsize=10)
        ax.set_xlabel(x_label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, framealpha=0.7)

    axes[0].set_ylabel("Recall@100 (max over run, mean±std across seeds)")
    fig.suptitle(
        f"CosPerUserTau on Amazon Beauty — Recall vs {x_label}",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    Path(out_path).parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")
    plt.close(fig)


def main():
    records = load_runs(MLFLOW_DB, EXPERIMENT)
    agg_data = aggregate(records)

    plot_recall_vs_x(
        agg_data,
        x_key="tau_min", line_key="tau_max",
        x_label="tau_min", line_label="tau_max",
        out_path="plots/cos_per_user_tau_amazon_recall_vs_tau_min.png",
    )
    plot_recall_vs_x(
        agg_data,
        x_key="tau_max", line_key="tau_min",
        x_label="tau_max", line_label="tau_min",
        out_path="plots/cos_per_user_tau_amazon_recall_vs_tau_max.png",
    )


if __name__ == "__main__":
    main()

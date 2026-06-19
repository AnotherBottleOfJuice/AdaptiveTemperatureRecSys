"""Plot input logit (cosine similarity) vs resulting tau for a given config.

Reads a YAML experiment config, builds the *actual* Tau object(s) from the
sweep grid (so the curve always matches sasrec/model/model/tau.py, including
the clamping branches), evaluates tau over the logit domain, and shades the
real pos/neg logit ranges measured from mlruns/mlflow.db.

Usage:
    python scripts/plot_logits_vs_tau.py [config.yaml]

Defaults to configs/shifted_cos_per_user_tau_30e.yaml.
"""

import copy
import itertools
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sasrec.model.model import tau as tau_mod
from sasrec.model.model.tau import TauContext

CONFIG = Path(sys.argv[1] if len(sys.argv) > 1 else "configs/shifted_cos_per_user_tau_30e.yaml")
MLFLOW_DB = Path("mlruns/mlflow.db")


def eval_tau(tau_obj, logits):
    """Evaluate a Tau object as a function of a single per-element logit."""
    g = torch.as_tensor(logits, dtype=torch.float32).reshape(-1, 1)  # [N, 1] -> pos
    neg = torch.empty(g.shape[0], 1, 0)  # no in-batch negatives for the curve
    net = SimpleNamespace(tokens_passed=torch.tensor(0.0))
    ctx = TauContext(net=net, pos_logits=g, neg_logits=neg, batch=None)
    out = torch.as_tensor(tau_obj(ctx)).detach().reshape(-1).numpy()
    if out.size == 1:  # logit-independent tau (Constant/Param) -> broadcast
        out = np.full(g.shape[0], float(out))
    return out


def build_taus(raw):
    """Return (combos, keys): combos is [(params_dict, tau_obj)], keys the swept tau params."""
    base = dict(raw["tau"]["json_args"])
    cls = getattr(tau_mod, raw["tau"]["class_name"])
    grid = raw.get("sweep", {}).get("grid", {})
    tau_grid = {
        k.split("tau.json_args.")[1]: v
        for k, v in grid.items()
        if k.startswith("tau.json_args.") and isinstance(v, list) and len(v) > 1
    }

    keys = list(tau_grid)
    combos = []
    for combo in itertools.product(*tau_grid.values()) if keys else [()]:
        params = dict(zip(keys, combo))
        args = copy.deepcopy(base)
        args.update(params)
        combos.append((params, cls(**args)))
    return combos, keys


def logit_ranges(db_path):
    """(lo, mean, hi) for pos and neg logit means from mlflow.db, or None."""
    if not db_path.exists():
        return None
    con = sqlite3.connect(db_path)
    try:
        def stat(key):
            row = con.execute(
                "SELECT MIN(metrics.value), AVG(metrics.value), MAX(metrics.value) FROM metrics "
                "JOIN tags ON metrics.run_uuid = tags.run_uuid "
                "WHERE metrics.key = ? AND tags.key = 'mlflow.runName' "
                "AND tags.value LIKE 'CosPerUser%'",
                (key,),
            ).fetchone()
            return row if row and row[0] is not None else None
        return {"pos": stat("train/pos_logit_mean"), "neg": stat("train/neg_logit_mean")}
    finally:
        con.close()


LINESTYLES = ["-", "--", "-.", ":"]


def main():
    raw = yaml.safe_load(CONFIG.read_text())
    combos, keys = build_taus(raw)
    ranges = logit_ranges(MLFLOW_DB)

    if ranges and ranges["neg"] and ranges["pos"]:
        op_lo, op_hi = ranges["neg"][0], ranges["pos"][2]
    else:
        op_lo, op_hi = -0.1, 0.4
    domain = np.linspace(op_lo - 0.1, op_hi + 0.15, 1200)

    # Small multiples: one panel per first swept key, color = 2nd key, style = 3rd key.
    facet_key = keys[0] if keys else None
    color_key = keys[1] if len(keys) > 1 else None
    style_key = keys[2] if len(keys) > 2 else None

    facet_vals = sorted({p.get(facet_key) for p, _ in combos}) if facet_key else [None]
    color_vals = sorted({p.get(color_key) for p, _ in combos}) if color_key else [None]
    style_vals = sorted({p.get(style_key) for p, _ in combos}) if style_key else [None]
    cmap = plt.get_cmap("viridis")
    color_of = {v: cmap(i / max(1, len(color_vals) - 1)) for i, v in enumerate(color_vals)}
    style_of = {v: LINESTYLES[i % len(LINESTYLES)] for i, v in enumerate(style_vals)}

    n = len(facet_vals)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 5.0), sharey=True, squeeze=False)
    axes = axes[0]

    for ax, fval in zip(axes, facet_vals):
        # measured logit bands first (behind the curves)
        if ranges:
            for which, c in (("neg", "steelblue"), ("pos", "seagreen")):
                r = ranges[which]
                if r:
                    ax.axvspan(r[0], r[2], color=c, alpha=0.12)
                    ax.axvline(r[1], color=c, ls=":", lw=1)

        for params, tau_obj in combos:
            if facet_key and params.get(facet_key) != fval:
                continue
            ax.plot(
                domain, eval_tau(tau_obj, domain),
                color=color_of[params.get(color_key)],
                ls=style_of[params.get(style_key)],
                lw=1.8,
            )
        title = f"{facet_key}={fval:g}" if facet_key else raw["tau"]["class_name"]
        ax.set_title(title)
        ax.set_xlabel("logit  (cosine similarity)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("tau applied to that logit")

    # one shared legend describing the color / style encoding
    handles = []
    if color_key:
        handles += [plt.Line2D([], [], color=color_of[v], lw=2, label=f"{color_key}={v:g}")
                    for v in color_vals]
    if style_key:
        handles += [plt.Line2D([], [], color="0.3", ls=style_of[v], lw=2, label=f"{style_key}={v:g}")
                    for v in style_vals]
    handles += [
        plt.Line2D([], [], color="seagreen", ls=":", lw=2, label="pos logit mean/range"),
        plt.Line2D([], [], color="steelblue", ls=":", lw=2, label="neg logit mean/range"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"{raw['tau']['class_name']}  —  {CONFIG.name}  (operating range)", fontsize=13)

    out = Path("plots") / f"{CONFIG.stem}_logits_vs_tau.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()

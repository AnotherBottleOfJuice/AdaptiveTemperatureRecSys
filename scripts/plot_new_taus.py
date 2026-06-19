"""Plot tau vs logit for the two new per-user temperatures.

Left panel:  DoubleDipCosPerUserTau -- shows the
    tau_max -> tau_min -> tau_mid -> tau_low
profile (extra sharpening for near-best logits), with the plain
ShiftedCosPerUserTau overlaid for reference.

Right panel: ConfidenceShiftedCosPerUserTau -- the per-user shifted cosine whose
shift is driven by model confidence (mean positive logit). One curve per
confidence level, so you can see the dip slide along the logit axis as the model
grows more confident (this is the "use confidence instead of time" variant of
ExpandedShiftedCosPerUserTau).

The measured train logit bands from mlruns/mlflow.db are shaded behind the curves.

Usage:
    python scripts/plot_new_taus.py
"""

import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sasrec.model.model.tau import (  # noqa: E402
    ConfidenceShiftedCosPerUserTau,
    DoubleDipCosPerUserTau,
    ShiftedCosPerUserTau,
    TauContext,
)

MLFLOW_DB = Path("mlruns/mlflow.db")

BASE = dict(initial_tau=0.45, num_epochs=30, num_tokens_per_epoch=4019032)
TAU_MIN, TAU_MAX = 0.04, 0.065


def logit_ranges(db_path):
    """(lo, mean, hi) of pos and neg logit means from CosPerUser runs, or None."""
    if not db_path.exists():
        return None
    con = sqlite3.connect(db_path)
    try:
        def stat(key):
            row = con.execute(
                "SELECT MIN(metrics.value), AVG(metrics.value), MAX(metrics.value) "
                "FROM metrics JOIN tags ON metrics.run_uuid = tags.run_uuid "
                "WHERE metrics.key = ? AND tags.key = 'mlflow.runName' "
                "AND tags.value LIKE 'CosPerUser%'",
                (key,),
            ).fetchone()
            return row if row and row[0] is not None else None
        return {"pos": stat("train/pos_logit_mean"), "neg": stat("train/neg_logit_mean")}
    finally:
        con.close()


def shade_bands(ax, ranges):
    if not ranges:
        return
    for which, c in (("neg", "steelblue"), ("pos", "seagreen")):
        r = ranges[which]
        if r:
            ax.axvspan(r[0], r[2], color=c, alpha=0.12)
            ax.axvline(r[1], color=c, ls=":", lw=1)


def main():
    ranges = logit_ranges(MLFLOW_DB)
    domain = np.linspace(-0.4, 1.0, 1400)
    logits = torch.as_tensor(domain, dtype=torch.float32)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)

    # ---- DoubleDip ------------------------------------------------------------
    ddip = DoubleDipCosPerUserTau(
        tau_min=TAU_MIN, tau_max=TAU_MAX, shift=0.05, scale=0.3,
        tau_mid=0.055, tau_low=0.045, **BASE,
    )
    shifted = ShiftedCosPerUserTau(
        tau_min=TAU_MIN, tau_max=TAU_MAX, shift=-0.4, scale=0.3, **BASE,
    )
    shade_bands(ax1, ranges)
    ax1.plot(domain, ddip.tau_for(logits).numpy(), color="crimson", lw=2.2,
             label="DoubleDip  (max→min→mid→low)")
    ax1.plot(
        domain,
        shifted(TauContext(net=SimpleNamespace(tokens_passed=torch.tensor(0.0)),
                           pos_logits=logits.reshape(-1, 1),
                           neg_logits=torch.empty(logits.shape[0], 1, 0),
                           batch=None)).reshape(-1).numpy(),
        color="0.4", lw=1.8, ls="--", label="ShiftedCos  (max→min→max)",
    )
    for y, lab in ((TAU_MIN, "tau_min"), (0.045, "tau_low"),
                   (0.055, "tau_mid"), (TAU_MAX, "tau_max")):
        ax1.axhline(y, color="0.8", lw=0.8, zorder=0)
        ax1.text(0.99, y, f" {lab}", va="center", ha="left", fontsize=7, color="0.5",
                 transform=ax1.get_yaxis_transform())
    ax1.set_title("DoubleDipCosPerUserTau")
    ax1.set_xlabel("logit  (cosine similarity)")
    ax1.set_ylabel("tau applied to that logit")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)

    # ---- ConfidenceShifted ----------------------------------------------------
    conf_tau = ConfidenceShiftedCosPerUserTau(
        tau_min=TAU_MIN, tau_max=TAU_MAX, min_shift=-0.1, max_shift=-0.4, scale=0.3,
        conf_min=0.13, conf_max=0.30, **BASE,
    )
    shade_bands(ax2, ranges)
    cmap = plt.get_cmap("viridis")
    confs = [0.13, 0.18, 0.23, 0.30]
    for i, c in enumerate(confs):
        ax2.plot(domain, conf_tau.tau_for(logits, c).numpy(),
                 color=cmap(i / (len(confs) - 1)), lw=2.0,
                 label=f"confidence (pos_mean) = {c:.2f}")
    ax2.set_title("ConfidenceShiftedCosPerUserTau\n(shift driven by confidence, not time)")
    ax2.set_xlabel("logit  (cosine similarity)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.3)

    handles = [
        plt.Line2D([], [], color="seagreen", ls=":", lw=2, label="pos logit mean/range"),
        plt.Line2D([], [], color="steelblue", ls=":", lw=2, label="neg logit mean/range"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("New per-user temperatures  —  tau vs logit", fontsize=13)

    out = Path("plots") / "new_taus_logits_vs_tau.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()

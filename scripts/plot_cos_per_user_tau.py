"""Plot CosPerUserTau: input logit (cosine similarity) vs resulting tau.

tau = (1 + cos(pi*(logit+1))) * (tau_max - tau_min) / 2 + tau_min
(see sasrec/model/model/tau.py :: CosPerUserTau)
"""

import os

import numpy as np
import matplotlib.pyplot as plt


def cos_per_user_tau(logit, tau_min, tau_max):
    return (1 + np.cos(np.pi * (logit + 1))) * (tau_max - tau_min) / 2 + tau_min


# cosine similarity range
s = np.linspace(-1.0, 1.0, 1000)

# combos from configs/cos_per_user_tau_30e.yaml sweep grid
combos = [
    (0.030, 0.055),
    (0.030, 0.060),
    (0.035, 0.055),
    (0.035, 0.060),
]

# actual operating range measured from mlruns (run dd82197c, tau_min=0.03/max=0.055)
NEG_LO, NEG_MEAN, NEG_HI = -0.063, -0.021, 0.049   # train/neg_logit_mean
POS_LO, POS_MEAN, POS_HI = 0.131, 0.247, 0.281     # train/pos_logit_mean
OP_LO, OP_HI = NEG_LO, POS_HI

fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(13, 5.5))

# ---- left: full cosine domain, with the real operating band shaded ----
for tau_min, tau_max in combos:
    ax_full.plot(s, cos_per_user_tau(s, tau_min, tau_max),
                 label=f"min={tau_min:.3g}, max={tau_max:.3g}")
ax_full.axvspan(OP_LO, OP_HI, color="orange", alpha=0.18,
                label="actual logit range\n(measured)")
ax_full.set_xlabel("logit  (cosine similarity)")
ax_full.set_ylabel("tau applied to that logit")
ax_full.set_title("Full domain [-1, 1]\n(the model only ever visits the shaded band)")
ax_full.legend(fontsize=8)
ax_full.grid(alpha=0.3)

# ---- right: zoom into the actual operating range ----
zs = np.linspace(OP_LO - 0.02, OP_HI + 0.02, 600)
for tau_min, tau_max in combos:
    ax_zoom.plot(zs, cos_per_user_tau(zs, tau_min, tau_max),
                 label=f"min={tau_min:.3g}, max={tau_max:.3g}")

tmin, tmax = combos[0]
ax_zoom.axvspan(NEG_LO, NEG_HI, color="steelblue", alpha=0.15)
ax_zoom.axvspan(POS_LO, POS_HI, color="seagreen", alpha=0.15)
for x, c, lab in [(NEG_MEAN, "steelblue", "neg mean\n(-0.02)"),
                  (POS_MEAN, "seagreen", "pos mean\n(+0.25)")]:
    y = cos_per_user_tau(x, tmin, tmax)
    ax_zoom.scatter([x], [y], color=c, zorder=5, s=35)
    ax_zoom.annotate(lab, (x, y), textcoords="offset points", xytext=(6, -22),
                     fontsize=8, color=c)
ax_zoom.set_xlabel("logit  (cosine similarity)")
ax_zoom.set_ylabel("tau applied to that logit")
ax_zoom.set_title("Operating range only\n(smooth & monotonic: negatives sharp, positives softer)")
ax_zoom.legend(fontsize=8)
ax_zoom.grid(alpha=0.3)

out = "plots/cos_per_user_tau_curve.png"
os.makedirs("plots", exist_ok=True)
fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"saved {out}")

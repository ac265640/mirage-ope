"""Render the prototype figure used in the Round 1 abstract from results/results.json."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "results"
NAVY, AMBER, RED, GREEN, GREY = "#12324a", "#b8740a", "#b03a2e", "#1e7a46", "#8a96a0"

res = json.loads((OUT / "results.json").read_text())
trap, sweep, conf = res["calm_week_trap"], res["support_sweep"]["rows"], res["confounding"]

plt.rcParams.update({"font.family": "Helvetica", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 9})
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), gridspec_kw={"width_ratios": [1.15, 1, 1.1]})

# A: calm-week trap
ax = axes[0]
names = ["Naive\nbefore/after", "IPS", "SNIPS", "DM", "DR"]
vals = [trap["uplift_naive_before_after"], trap["uplift_IPS"], trap["uplift_SNIPS"], trap["uplift_DM"], trap["uplift_DR"]]
cols = [RED, NAVY, NAVY, NAVY, NAVY]
ax.bar(names, vals, color=cols, width=0.62)
lo, hi = trap["uplift_DR_CI95"]
ax.errorbar([4], [trap["uplift_DR"]], yerr=[[trap["uplift_DR"] - lo], [hi - trap["uplift_DR"]]],
            color=AMBER, capsize=4, lw=1.6)
ax.axhline(trap["uplift_true"], color=GREEN, lw=1.6, ls="--")
ax.text(1.0, trap["uplift_true"] - 0.055, "true uplift (simulator)", color=GREEN, ha="left", fontsize=8)
ax.axhline(0, color="#333", lw=0.6)
ax.set_ylabel("uplift of new policy (ticks / decision)")
ax.set_title("A · Calm-week trap: naive says WIN, truth is a LOSS")
ax.tick_params(axis="x", labelsize=7.6)

# B: support stress test
ax = axes[1]
taus = [r["tau"] for r in sweep]
for k, c, m in [("IPS", NAVY, "o"), ("DM", GREY, "s"), ("DR", AMBER, "D")]:
    ax.plot(taus, [r[f"abserr_{k}"] for r in sweep], marker=m, color=c, lw=1.4, ms=4, label=k)
ax.set_xscale("log")
ax.invert_xaxis()
ax.set_xticks([1, 0.6, 0.4, 0.25, 0.15, 0.1]); ax.set_xticklabels(["1", ".6", ".4", ".25", ".15", ".1"]); ax.minorticks_off()
ax.set_xlabel("logging temperature τ  (right = more deterministic)")
ax.set_ylabel("abs. error of V($\\pi_1$) estimate")
ax.set_title("B · Support stress test")
ax.legend(frameon=False, fontsize=7.6, loc="upper left")
ax2 = ax.twinx()
ax2.plot(taus, [r["pi0_tight_prob_in_stress"] for r in sweep], color=RED, lw=1, ls=":")
ax2.set_yscale("log")
ax2.set_ylabel("P(tight quote | stressed) in logs", color=RED, fontsize=7.6)
ax2.tick_params(axis="y", colors=RED, labelsize=7)
ax2.spines["right"].set_visible(True)

# C: sensitivity to hidden confounding
ax = axes[2]
g = [c["gamma"] for c in conf["curve"]]
ax.fill_between(g, [c["lo"] for c in conf["curve"]], [c["hi"] for c in conf["curve"]], color=NAVY, alpha=0.15,
                label="MSM bounds on V($\\pi_1$)")
ax.axhline(conf["V0_logged"], color="#333", lw=1, label="current policy V($\\pi_0$)")
ax.axhline(conf["V1_true"], color=GREEN, lw=1.6, ls="--", label="true V($\\pi_1$)")
ax.scatter([1], [conf["SNIPS_nominal"]], color=RED, zorder=5, s=22, label="nominal OPE (ignores U)")
ax.axvline(conf["breakdown_gamma"], color=AMBER, lw=1.2, ls=":")
ax.text(conf["breakdown_gamma"] * 0.96, 1.05, f"Γ* = {conf['breakdown_gamma']:.2f}", color=AMBER, fontsize=8, ha="right")
ax.axvline(conf["gamma_true"], color=GREEN, lw=1, ls=":")
ax.text(conf["gamma_true"] * 1.04, 1.05, f"true Γ = {conf['gamma_true']:.1f}", color=GREEN, fontsize=8, ha="left")
ax.set_xscale("log")
ax.set_xticks([1, 2, 3, 5, 7]); ax.set_xticklabels(["1", "2", "3", "5", "7"]); ax.minorticks_off()
ax.set_xlabel("confounding strength Γ")
ax.set_ylabel("policy value (ticks / decision)")
ax.set_ylim(-0.25, 1.15)
ax.set_title("C · Hidden informed flow: robust verdict?")
ax.legend(frameon=True, framealpha=0.92, edgecolor="none", fontsize=6.8, loc="lower left")

fig.tight_layout(w_pad=1.6)
fig.savefig(OUT / "prototype_results.png", dpi=300)
print("saved", OUT / "prototype_results.png")

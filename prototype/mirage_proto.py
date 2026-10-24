"""MIRAGE prototype: off-policy evaluation of a market maker's quoting policy.

A deliberately small, fully reproducible version of the Round 1 plan:
  * a toy quoting world with calm/stressed regimes and hidden informed flow U,
  * a production policy pi0 that logs (x, a, pi0(a|x), r),
  * a candidate policy pi1 ("always quote tight"),
  * naive before/after vs IPS / SNIPS / DM / DR vs the simulator's ground truth,
  * day-block bootstrap CIs,
  * a support stress test (pi0 temperature sweep),
  * marginal-sensitivity-model bounds when pi0 secretly reacts to U.

Simplifications vs the full plan (stated honestly): 3 spread actions instead of 9
(no skew), one market order per decision instead of a full order book, no
inventory carry-over. Run:  python mirage_proto.py
"""
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

SPREADS = np.array([1.0, 2.0, 3.0])          # half-spread in ticks, actions 0,1,2
SIGMA = {0: 0.5, 1: 2.0}                     # calm, stressed volatility
P_INFORMED = {0: 0.06, 1: 0.45}              # base share of informed flow
P_STRESSED = 0.4                             # long-run share of stressed days
DAY = 2000                                   # decisions per day
OUT = Path(__file__).parent / "results"


# ---------------------------------------------------------------- world
def contexts(rng, regimes):
    """Per-decision context x and hidden informed-flow flag U for a list of day regimes."""
    reg = np.repeat(regimes, DAY)
    sigma = np.where(reg == 1, SIGMA[1], SIGMA[0])
    vol_est = sigma * np.exp(0.25 * rng.standard_normal(reg.size))       # noisy realised vol
    imb = rng.standard_normal(reg.size)                                   # book imbalance
    p_inf = np.where(reg == 1, P_INFORMED[1], P_INFORMED[0]) * (1 + 0.6 * np.abs(imb))
    U = rng.random(reg.size) < np.clip(p_inf, 0, 0.95)                    # NOT logged
    day = np.repeat(np.arange(len(regimes)), DAY)
    return dict(reg=reg, sigma=sigma, vol_est=vol_est, imb=imb, U=U, day=day)


def markout(rng, ctx, a):
    """h-step markout PnL of the fill (0 if no fill) for chosen actions a."""
    s = SPREADS[a]
    n = a.size
    jump = np.abs(rng.standard_normal(n)) * 1.5 * ctx["sigma"]           # informed trader's edge
    noise_fill = rng.random(n) < np.exp(-0.8 * (s - 1.0))                # tighter quotes fill more
    noise_r = s + 0.5 * ctx["sigma"] * rng.standard_normal(n)                  # noise flow: earn spread
    inf_fill = jump > s                                                  # informed only trade when edge > spread
    inf_r = s - jump                                                     # informed flow: adverse selection
    return np.where(ctx["U"], np.where(inf_fill, inf_r, 0.0), np.where(noise_fill, noise_r, 0.0))


# ---------------------------------------------------------------- policies
def pi0_probs(ctx, tau):
    """Production algorithm: widen spread as volatility rises (softmax with temperature tau)."""
    t = 1 + 2 * np.clip(np.log(ctx["vol_est"] / 0.5) / np.log(4.0), 0, 1)  # target spread 1..3
    logits = -np.abs(SPREADS[None, :] - t[:, None]) / tau
    p = np.exp(logits - logits.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def pi0_true_probs(ctx, tau, beta):
    """What the desk actually did: a human override widens to 3 ticks when flow is toxic (U)."""
    t = 1 + 2 * np.clip(np.log(ctx["vol_est"] / 0.5) / np.log(4.0), 0, 1)
    logits = -np.abs(SPREADS[None, :] - t[:, None]) / tau
    logits[:, 2] += beta * ctx["U"]
    p = np.exp(logits - logits.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def pi1_probs(ctx):
    """Candidate: always quote the tightest spread."""
    p = np.zeros((ctx["reg"].size, 3))
    p[:, 0] = 1.0
    return p


def sample(rng, p):
    return (rng.random(p.shape[0])[:, None] > np.cumsum(p, 1)).sum(1)


def features(ctx):
    return np.column_stack([np.log(ctx["vol_est"]), ctx["imb"], np.abs(ctx["imb"])])


# ---------------------------------------------------------------- estimators
def fit_q(X, a, r, rng, folds=2):
    """Cross-fitted per-action reward model q(x, a); returns out-of-fold predictions for all actions."""
    q = np.zeros((X.shape[0], 3))
    fold = rng.integers(0, folds, X.shape[0])
    for f in range(folds):
        tr, te = fold != f, fold == f
        for k in range(3):
            m = tr & (a == k)
            model = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, random_state=0)
            model.fit(X[m], r[m])
            q[te, k] = model.predict(X[te])
    return q


def estimates(a, r, p0, p1, q):
    idx = np.arange(a.size)
    w = p1[idx, a] / p0[idx, a]
    q_pi1 = (q * p1).sum(1)
    return {
        "IPS": float(np.mean(w * r)),
        "SNIPS": float(np.sum(w * r) / np.sum(w)),
        "DM": float(np.mean(q_pi1)),
        "DR": float(np.mean(q_pi1 + w * (r - q[idx, a]))),
        "ESS": float(w.sum() ** 2 / (w ** 2).sum()),
    }


def msm_bounds(r, w_nom, gamma):
    """Sharp bounds on the self-normalised policy value when true inverse propensities lie in
    [1 + (w-1)/gamma, 1 + (w-1)*gamma] (marginal sensitivity model; Kallus & Zhou 2018).
    Only samples with w_nom > 0 (actions pi1 would take) enter."""
    m = w_nom > 0
    r, w = r[m], w_nom[m]
    lo_w, hi_w = 1 + (w - 1) / gamma, 1 + (w - 1) * gamma

    def best(sign):
        order = np.argsort(-sign * r)                 # put weight on high (or low) rewards first
        rs, lo, hi = r[order], lo_w[order], hi_w[order]
        # first k samples at hi weight, rest at lo weight; scan all k
        num = np.concatenate([[0], np.cumsum(hi * rs)]) + (np.sum(lo * rs) - np.concatenate([[0], np.cumsum(lo * rs)]))
        den = np.concatenate([[0], np.cumsum(hi)]) + (np.sum(lo) - np.concatenate([[0], np.cumsum(lo)]))
        vals = num / den
        return vals.max() if sign > 0 else vals.min()

    return float(best(-1)), float(best(+1))


# ---------------------------------------------------------------- experiments
def replay_truth(rng, ctx, policy, reps=10, **kw):
    """Ground truth on the SAME logged contexts (x and hidden U): replay the policy with fresh noise."""
    p = policy(ctx, **kw)
    return float(np.mean([markout(rng, ctx, sample(rng, p)).mean() for _ in range(reps)]))


def truth(rng, policy, n_days=400, **kw):
    regimes = (rng.random(n_days) < P_STRESSED).astype(int)
    ctx = contexts(rng, regimes)
    p = policy(ctx, **kw)
    return float(markout(rng, ctx, sample(rng, p)).mean())


def calm_week_trap(seed=7, tau=0.5, n_log_days=40):
    rng = np.random.default_rng(seed)
    regimes = (rng.random(n_log_days) < P_STRESSED).astype(int)
    ctx = contexts(rng, regimes)
    p0 = pi0_probs(ctx, tau)
    a = sample(rng, p0)
    r = markout(rng, ctx, a)
    p1 = pi1_probs(ctx)
    X = features(ctx)
    q = fit_q(X, a, r, rng)
    est = estimates(a, r, p0, p1, q)

    # naive: launch pi1 in a calm week, compare with the logged period's average PnL
    wk = contexts(rng, np.zeros(5, dtype=int))
    naive_after = float(markout(rng, wk, sample(rng, pi1_probs(wk))).mean())
    v0_logged = float(r.mean())

    # day-block bootstrap for DR uplift
    idx = np.arange(a.size)
    w = p1[idx, a] / p0[idx, a]
    dr_terms = (q * p1).sum(1) + w * (r - q[idx, a])
    day_dr = np.bincount(ctx["day"], dr_terms) / DAY
    day_r = np.bincount(ctx["day"], r) / DAY
    boot = []
    for _ in range(2000):
        d = rng.integers(0, n_log_days, n_log_days)
        boot.append(day_dr[d].mean() - day_r[d].mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])

    trng = np.random.default_rng(seed + 1000)
    v1_true = replay_truth(trng, ctx, lambda c: pi1_probs(c))
    v0_true = replay_truth(trng, ctx, pi0_probs, tau=tau)
    return {
        "V0_logged": v0_logged,
        "V0_true": v0_true,
        "V1_true": v1_true,
        "uplift_true": v1_true - v0_true,
        "uplift_naive_before_after": naive_after - v0_logged,
        "uplift_IPS": est["IPS"] - v0_logged,
        "uplift_SNIPS": est["SNIPS"] - v0_logged,
        "uplift_DM": est["DM"] - v0_logged,
        "uplift_DR": est["DR"] - v0_logged,
        "uplift_DR_CI95": [float(lo), float(hi)],
        "ESS": est["ESS"],
        "n_logged": int(a.size),
    }


def support_sweep(taus=(1.0, 0.6, 0.4, 0.25, 0.15, 0.1), reps=8, n_log_days=20):
    """As pi0 becomes near-deterministic, logs stop containing tight quotes in stressed markets."""
    rows = []
    for tau in taus:
        errs = {"IPS": [], "SNIPS": [], "DM": [], "DR": []}
        ess, cover = [], []
        for rep in range(reps):
            rng = np.random.default_rng(1000 * rep + int(tau * 100))
            regimes = (rng.random(n_log_days) < P_STRESSED).astype(int)
            ctx = contexts(rng, regimes)
            p0 = pi0_probs(ctx, tau)
            a = sample(rng, p0)
            r = markout(rng, ctx, a)
            q = fit_q(features(ctx), a, r, rng)
            est = estimates(a, r, p0, pi1_probs(ctx), q)
            v1 = replay_truth(rng, ctx, lambda c: pi1_probs(c), reps=4)
            for k in errs:
                errs[k].append(abs(est[k] - v1))
            ess.append(est["ESS"] / a.size)
            stressed = ctx["reg"] == 1
            cover.append(float(np.mean(p0[stressed, 0])))
        rows.append({"tau": tau, "ESS_frac": float(np.mean(ess)),
                     "pi0_tight_prob_in_stress": float(np.mean(cover)),
                     **{f"abserr_{k}": float(np.median(v)) for k, v in errs.items()}})
    return {"rows": rows}


def confounding(beta=1.5, tau=0.5, seed=11, n_log_days=40):
    rng = np.random.default_rng(seed)
    regimes = (rng.random(n_log_days) < P_STRESSED).astype(int)
    ctx = contexts(rng, regimes)
    p_true = pi0_true_probs(ctx, tau, beta)
    p_nom = pi0_probs(ctx, tau)                      # what the log says the propensity was
    a = sample(rng, p_true)
    r = markout(rng, ctx, a)
    idx = np.arange(a.size)
    p1 = pi1_probs(ctx)
    w_nom = p1[idx, a] / p_nom[idx, a]
    w_true = p1[idx, a] / p_true[idx, a]
    snips_nom = float(np.sum(w_nom * r) / np.sum(w_nom))
    snips_oracle = float(np.sum(w_true * r) / np.sum(w_true))
    v0_logged = float(r.mean())

    v1_true = replay_truth(np.random.default_rng(seed + 1000), ctx, lambda c: pi1_probs(c))
    gammas = np.round(np.exp(np.linspace(0, 2.0, 41)), 3)
    bounds = [msm_bounds(r, w_nom, g) for g in gammas]
    # breakdown: smallest gamma at which "pi1 beats the logged policy" is no longer guaranteed
    nominal_says_better = snips_nom > v0_logged
    breakdown = next((float(g) for g, b in zip(gammas, bounds) if b[0] <= v0_logged), None)
    g_true = float(np.exp(beta))
    lo_t, hi_t = msm_bounds(r, w_nom, g_true)
    return {
        "beta": beta, "gamma_true": g_true,
        "V0_logged": v0_logged, "V1_true": v1_true,
        "SNIPS_nominal": snips_nom, "SNIPS_oracle_true_propensity": snips_oracle,
        "nominal_says_pi1_better": bool(nominal_says_better),
        "breakdown_gamma": breakdown,
        "bounds_at_gamma_true": [lo_t, hi_t], "truth_covered_at_gamma_true": bool(lo_t <= v1_true <= hi_t),
        "curve": [{"gamma": float(g), "lo": b[0], "hi": b[1]} for g, b in zip(gammas, bounds)],
    }


def multi_seed(seeds=range(1, 11)):
    """Robustness of the headline claims across independent worlds."""
    trap = [calm_week_trap(seed=s) for s in seeds]
    conf = [confounding(seed=s) for s in seeds]
    return {
        "n_seeds": len(trap),
        "naive_wrong_sign": sum(t["uplift_naive_before_after"] > 0 for t in trap),
        "DR_right_sign": sum(t["uplift_DR"] < 0 for t in trap),
        "DR_CI_covers_truth": sum(t["uplift_DR_CI95"][0] <= t["uplift_true"] <= t["uplift_DR_CI95"][1] for t in trap),
        "mean_abs_err": {k: float(np.mean([abs(t[f"uplift_{k}"] - t["uplift_true"]) for t in trap]))
                         for k in ["naive_before_after", "IPS", "SNIPS", "DM", "DR"]},
        "confounded_nominal_says_deploy": sum(c["nominal_says_pi1_better"] for c in conf),
        "bounds_cover_truth_at_true_gamma": sum(c["truth_covered_at_gamma_true"] for c in conf),
        "breakdown_gamma_range": [min(c["breakdown_gamma"] for c in conf), max(c["breakdown_gamma"] for c in conf)],
    }


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    res = {"calm_week_trap": calm_week_trap(), "support_sweep": support_sweep(), "confounding": confounding(),
           "multi_seed": multi_seed()}
    print(json.dumps(res["multi_seed"], indent=2))
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res["calm_week_trap"].items()}, indent=2))
    for row in res["support_sweep"]["rows"]:
        print(row)
    c = res["confounding"]
    print({k: v for k, v in c.items() if k != "curve"})

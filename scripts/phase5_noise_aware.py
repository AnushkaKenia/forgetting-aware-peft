"""Phase 5 v2: noise-aware adaptation/retention analysis (CPU only, no training).

Replaces the min-max composite of the original Phase 5 notebook, which stretched
differences smaller than measurement noise to a full 0-1 range (e.g. LoRA's
~2-point BoolQ "improvement" counted as a complete win while AdaLoRA's ~0
perplexity change scored as the worst retention).

Inputs : results/final_analysis/three_seed_per_seed_results.csv  (9 runs)
Outputs: results/phase5_v2/*.csv, plots/*.png, REPORT.md

Statistics note: n = 3 seeds, so every interval uses Student's t with 2 d.f.
(critical value 4.30). Resampling 3 points adds nothing, so no bootstrap here.
Seeds are paired across methods (same seed => same training sample and eval
subset), so method-vs-method comparisons use paired t-tests.
"""
from pathlib import Path
import itertools

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/final_analysis/three_seed_per_seed_results.csv"
OUT = ROOT / "results/phase5_v2"
PLOTS = OUT / "plots"
OUT.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

METHODS = ["lora", "adalora", "dora"]
ACC_PROBES = ["boolq_forgetting", "piqa_forgetting", "arc_forgetting", "mmlu_forgetting"]
PROBES = ACC_PROBES + ["wikitext_ppl_increase"]
ADAPT = "samsum_rl_gain"
N_PROBE = 500            # examples per accuracy probe
WIKI_BASE_PPL = 17.2027  # identical base-model perplexity in every seed
ALPHA = 0.05
TCRIT = stats.t.ppf(1 - ALPHA / 2, df=2)

df = pd.read_csv(INPUT)
df["method"] = df["method"].str.lower()
assert len(df) == 9 and set(df.method) == set(METHODS) and df.seed.nunique() == 3
wide = {m: df[df.method == m].sort_values("seed").set_index("seed") for m in METHODS}


def mean_ci(x):
    x = np.asarray(x, float)
    m, sd = x.mean(), x.std(ddof=1)
    half = TCRIT * sd / np.sqrt(len(x))
    return m, sd, m - half, m + half


# ---------------------------------------------------------------- 1. per-probe table
rows = []
for m in METHODS:
    for p in [ADAPT] + PROBES:
        mean, sd, lo, hi = mean_ci(wide[m][p])
        if p == ADAPT:
            verdict = ""
        elif lo > 0:
            verdict = "degradation"
        elif hi < 0:
            verdict = "improvement"
        else:
            verdict = "no detectable change"
        rows.append(dict(method=m, metric=p, mean=mean, sd=sd, ci95_low=lo, ci95_high=hi, verdict=verdict))
probe_table = pd.DataFrame(rows)
probe_table.to_csv(OUT / "probe_noise_table.csv", index=False)

# ---------------------------------------------------------------- 2. binomial noise floor
# Unpaired upper bound on the SE of (pre - post) accuracy at n=500 (true paired SE is smaller).
nf = []
for p0 in (0.5, 0.64, 0.68, 0.72, 0.45):
    se = np.sqrt(2 * p0 * (1 - p0) / N_PROBE)
    nf.append(dict(accuracy=p0, n=N_PROBE, se_gap_upper_bound=se, ci95_halfwidth=1.96 * se))
pd.DataFrame(nf).to_csv(OUT / "binomial_noise_floor.csv", index=False)

# ---------------------------------------------------------------- 3. paired comparisons
pw = []
for a, b in itertools.combinations(METHODS, 2):
    for p in [ADAPT] + PROBES:
        d = wide[a][p].values - wide[b][p].values
        t, pv = stats.ttest_1samp(d, 0.0) if d.std(ddof=1) > 0 else (np.nan, 1.0)
        half = TCRIT * d.std(ddof=1) / np.sqrt(len(d))
        pw.append(dict(a=a, b=b, metric=p, mean_diff_a_minus_b=d.mean(), ci95_low=d.mean() - half,
                       ci95_high=d.mean() + half, t=t, p=pv))
pairwise = pd.DataFrame(pw)
# Holm correction across all 18 tests
order = pairwise.p.argsort().values
m_tests = len(pairwise)
adj = np.empty(m_tests)
running = 0.0
for rank, i in enumerate(order):
    running = max(running, (m_tests - rank) * pairwise.p.iloc[i])
    adj[i] = min(1.0, running)
pairwise["p_holm"] = adj
pairwise.to_csv(OUT / "pairwise_paired_tests.csv", index=False)


# ---------------------------------------------------------------- 4. significance-aware Pareto
# Axes: adaptation (ROUGE-L gain, higher better) and WikiText PPL increase (lower better), the
# only retention probe with a detectable effect. A dominates B only if it is significantly better
# on one axis and not significantly worse on the other (unadjusted paired t-test, alpha=.05).
def sig_better(a, b, p, higher_is_better):
    row = pairwise[((pairwise.a == a) & (pairwise.b == b) | (pairwise.a == b) & (pairwise.b == a)) & (pairwise.metric == p)].iloc[0]
    if row.p >= ALPHA:
        return 0
    diff_a_minus_b = row.mean_diff_a_minus_b if row.a == a else -row.mean_diff_a_minus_b
    good = diff_a_minus_b > 0 if higher_is_better else diff_a_minus_b < 0
    return 1 if good else -1


pareto = []
for b in METHODS:
    dominated_by = []
    for a in METHODS:
        if a == b:
            continue
        s_ad = sig_better(a, b, ADAPT, True)
        s_rt = sig_better(a, b, "wikitext_ppl_increase", False)
        if s_ad >= 0 and s_rt >= 0 and (s_ad > 0 or s_rt > 0):
            dominated_by.append(a)
    pareto.append(dict(method=b, dominated_by=",".join(dominated_by), pareto_efficient=not dominated_by))
pareto = pd.DataFrame(pareto)
pareto.to_csv(OUT / "pareto_significance_aware.csv", index=False)

# ---------------------------------------------------------------- 5. fixed-reference composite
# loss_p = clip(max(mean_gap, 0) / ref_p, 0, 1); gaps whose 95% CI contains 0 count as 0.
# ref_p is a fixed "this much forgetting is a full loss" threshold chosen independently of the
# methods compared (min-max scaling over 3 methods is what manufactured the old ranking).
def composite(ref_scale):
    refs = {p: 0.05 * ref_scale for p in ACC_PROBES}
    refs["wikitext_ppl_increase"] = 0.10 * WIKI_BASE_PPL * ref_scale
    out = {}
    gain = {m: wide[m][ADAPT].mean() for m in METHODS}
    best = max(gain.values())
    for m in METHODS:
        losses = []
        for p in PROBES:
            row = probe_table[(probe_table.method == m) & (probe_table.metric == p)].iloc[0]
            gap = row["mean"] if row.verdict == "degradation" else 0.0
            losses.append(min(1.0, max(gap, 0.0) / refs[p]))
        out[m] = dict(adaptation=gain[m] / best, retention=1 - float(np.mean(losses)))
    return out


sweep_rows, win_rows = [], []
for scale in (0.5, 1.0, 2.0):
    comp = composite(scale)
    for w in np.linspace(0, 1, 101):
        scores = {m: (1 - w) * comp[m]["adaptation"] + w * comp[m]["retention"] for m in METHODS}
        top = max(scores.values())
        for m in METHODS:
            sweep_rows.append(dict(ref_scale=scale, retention_weight=w, method=m, score=scores[m],
                                   winner=bool(np.isclose(scores[m], top))))
    for m in METHODS:
        win_rows.append(dict(ref_scale=scale, method=m, adaptation=comp[m]["adaptation"], retention=comp[m]["retention"]))
sweep = pd.DataFrame(sweep_rows)
sweep.to_csv(OUT / "weight_sensitivity.csv", index=False)
comp_df = pd.DataFrame(win_rows)
comp_df.to_csv(OUT / "composite_components.csv", index=False)
winner_ranges = (sweep[sweep.winner].groupby(["ref_scale", "method"]).retention_weight.agg(["min", "max", "count"]).reset_index())
winner_ranges.to_csv(OUT / "winner_weight_ranges.csv", index=False)

# ---------------------------------------------------------------- 6. plots
colors = {"lora": "#1f77b4", "adalora": "#ff7f0e", "dora": "#2ca02c"}

fig, ax = plt.subplots(figsize=(7, 5))
for m in METHODS:
    a = probe_table[(probe_table.method == m) & (probe_table.metric == ADAPT)].iloc[0]
    w = probe_table[(probe_table.method == m) & (probe_table.metric == "wikitext_ppl_increase")].iloc[0]
    ax.errorbar(w["mean"], a["mean"], xerr=[[w["mean"] - w.ci95_low], [w.ci95_high - w["mean"]]],
                yerr=[[a["mean"] - a.ci95_low], [a.ci95_high - a["mean"]]], fmt="o", color=colors[m],
                capsize=4, label=f"{m.upper()}  (Pareto-efficient: {bool(pareto.set_index('method').loc[m, 'pareto_efficient'])})")
ax.set(xlabel="WikiText-2 perplexity increase (lower is better)", ylabel="SAMSum ROUGE-L gain (higher is better)",
       title="Adaptation vs retention, mean with 95% t-CI (n=3 seeds)")
ax.grid(alpha=.25)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(PLOTS / "tradeoff_with_ci.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ypos = {p: i for i, p in enumerate(ACC_PROBES)}
for j, m in enumerate(METHODS):
    for p in ACC_PROBES:
        r = probe_table[(probe_table.method == m) & (probe_table.metric == p)].iloc[0]
        y = ypos[p] + (j - 1) * 0.2
        ax.errorbar(r["mean"], y, xerr=[[r["mean"] - r.ci95_low], [r.ci95_high - r["mean"]]], fmt="o",
                    color=colors[m], capsize=3, label=m.upper() if p == ACC_PROBES[0] else None)
ax.axvline(0, color="k", lw=.8)
ax.axvspan(-1.96 * np.sqrt(2 * .25 / N_PROBE), 1.96 * np.sqrt(2 * .25 / N_PROBE), color="gray", alpha=.15,
           label="single-run noise floor (n=500, upper bound)")
ax.set_yticks(list(ypos.values()), [p.replace("_forgetting", "") for p in ACC_PROBES])
ax.set(xlabel="Accuracy gap, pre - post (positive = forgetting)", title="Accuracy probes: no detectable forgetting for any method")
ax.legend(fontsize=8)
ax.grid(alpha=.25)
fig.tight_layout()
fig.savefig(PLOTS / "accuracy_probe_forest.png", dpi=200)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, scale in zip(axes, (0.5, 1.0, 2.0)):
    for m in METHODS:
        d = sweep[(sweep.ref_scale == scale) & (sweep.method == m)]
        ax.plot(d.retention_weight, d.score, label=m.upper(), color=colors[m])
    ax.set(title=f"reference scale x{scale}", xlabel="Retention weight")
    ax.grid(alpha=.25)
axes[0].set_ylabel("Composite score")
axes[0].legend()
fig.tight_layout()
fig.savefig(PLOTS / "weight_sensitivity_v2.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 7. report
def f(x, n=4):
    return f"{x:.{n}f}"


lines = ["# Phase 5 v2: noise-aware analysis (3 seeds)", "",
         "All intervals are 95% Student-t (2 d.f.). Positive gap = degradation.", "",
         "## Per-probe verdicts", "", "| method | metric | mean | sd | 95% CI | verdict |", "|---|---|---|---|---|---|"]
for _, r in probe_table.iterrows():
    lines.append(f"| {r.method} | {r.metric} | {f(r['mean'])} | {f(r.sd)} | [{f(r.ci95_low)}, {f(r.ci95_high)}] | {r.verdict} |")
lines += ["", "## Significant paired differences (unadjusted p<.05; Holm-adjusted p shown)", "",
          "| a - b | metric | mean diff | 95% CI | p | p_holm |", "|---|---|---|---|---|---|"]
for _, r in pairwise[pairwise.p < ALPHA].iterrows():
    lines.append(f"| {r.a} - {r.b} | {r.metric} | {f(r.mean_diff_a_minus_b)} | [{f(r.ci95_low)}, {f(r.ci95_high)}] | {f(r.p)} | {f(r.p_holm)} |")
lines += ["", "## Significance-aware Pareto (adaptation vs WikiText PPL increase)", "",
          "| method | efficient | dominated by |", "|---|---|---|"]
for _, r in pareto.iterrows():
    lines.append(f"| {r.method} | {r.pareto_efficient} | {r.dominated_by or '-'} |")
lines += ["", "## Composite winners by retention weight (fixed-reference scaling)", "",
          "| ref scale | method | weight range where it wins | #grid points |", "|---|---|---|---|"]
for _, r in winner_ranges.iterrows():
    lines.append(f"| x{r.ref_scale} | {r.method} | {f(r['min'], 2)}-{f(r['max'], 2)} | {int(r['count'])} |")
(OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

#!/usr/bin/env python
"""Early-stopping table and matched-adaptation comparison (reads saved evals only).

Gain at a step = SAMSum ROUGE-L minus the untrained model's, on all 200 held-out examples where an
extension file exists (scripts/extend_samsum_early.py), else on the first 100 (flagged n=100).
Forgetting = WikiText PPL increase and KL(base || tuned).

  python scripts/matched_adaptation.py --tag v3  ->  results/matched_adaptation/
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from fapeft.store import read_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"lora": "#1f77b4", "adalora": "#ff7f0e", "dora": "#2ca02c"}
GRID = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
TCRIT = stats.t.ppf(0.975, 2)


def collect(root: Path) -> pd.DataFrame:
    base_rl = read_json(root / "base/evals/full_step_0000__samsum.json")["rougeL"]
    base_ppl = read_json(root / "base/evals/curve_step_0000__wiki.json")["ppl"]
    rows = []
    for run in sorted(root.glob("seed*_*")):
        seed, method = re.fullmatch(r"seed(\d+)_(\w+)", run.name).groups()
        ev = run / "evals"
        rows.append(dict(seed=int(seed), method=method, step=0, gain=0.0, ppl_inc=0.0, kl=0.0, n=200))
        for f in sorted(ev.glob("curve_step_*__samsum.json")):
            step = int(re.search(r"step_(\d+)", f.name).group(1))
            curve, wiki = read_json(f), read_json(ev / f.name.replace("samsum", "wiki"))
            if wiki is None:
                continue
            full, ext = ev / f"full_step_{step:04d}__samsum.json", ev / f"ext_step_{step:04d}__samsum.json"
            if full.exists():
                rl, n = read_json(full)["rougeL"], 200
            elif ext.exists():
                rl, n = (np.sum(curve["rougeL_per_example"]) + np.sum(read_json(ext)["rougeL_per_example"])) / 200, 200
            else:
                rl, n = curve["rougeL"], 100
            rows.append(dict(seed=int(seed), method=method, step=step, gain=rl - base_rl,
                             ppl_inc=wiki["ppl"] - base_ppl, kl=wiki["kl_base_to_tuned"], n=n))
    return pd.DataFrame(rows).sort_values(["method", "seed", "step"]).reset_index(drop=True)


def at_gain(run: pd.DataFrame, g: float, col: str):
    """Value of `col` where the run's gain first reaches g (linear interpolation); NaN if never."""
    r = run.sort_values("step")
    for (_, a), (_, b) in zip(r.iloc[:-1].iterrows(), r.iloc[1:].iterrows()):
        if a.gain < g <= b.gain:
            w = (g - a.gain) / (b.gain - a.gain)
            return a[col] + w * (b[col] - a[col]), b.step
    return np.nan, np.nan


def ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return x.mean() if len(x) else np.nan, np.nan, len(x)
    return x.mean(), TCRIT * x.std(ddof=1) / np.sqrt(len(x)) if len(x) == 3 else stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)), len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    a = ap.parse_args()
    df = collect(Path(a.runs_dir) / a.tag)
    out = ROOT / "results/matched_adaptation"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "curves_200.csv", index=False)

    # A2: early-stopping table
    rows = []
    for m in ("lora", "dora"):
        d = df[df.method == m]
        final = d[d.step == 250].set_index("seed")
        for step, g in d.groupby("step"):
            g = g.set_index("seed")
            rows.append(dict(method=m, step=step, n_examples=int(g.n.min()), gain=g.gain.mean(), gain_sd=g.gain.std(),
                             ppl_inc=g.ppl_inc.mean(), ppl_inc_sd=g.ppl_inc.std(), kl=g.kl.mean(),
                             pct_final_gain=100 * (g.gain / final.gain).mean(), pct_final_ppl=100 * (g.ppl_inc / final.ppl_inc).mean()))
    early = pd.DataFrame(rows)
    early.to_csv(out / "early_stopping.csv", index=False)

    # A1: matched adaptation
    mrows = []
    for gl in GRID:
        for m in ("lora", "dora", "adalora"):
            vals = [at_gain(g, gl, "ppl_inc") for _, g in df[df.method == m].groupby("seed")]
            kls = [at_gain(g, gl, "kl")[0] for _, g in df[df.method == m].groupby("seed")]
            mean, half, n = ci([v[0] for v in vals])
            kmean, _, _ = ci(kls)
            mrows.append(dict(gain=gl, method=m, n_seeds_reached=n, ppl_inc=mean, ppl_inc_ci_half=half, kl=kmean,
                              median_step=np.nanmedian([v[1] for v in vals]) if n else np.nan))
    matched = pd.DataFrame(mrows)
    matched.to_csv(out / "matched_gain.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for m, g in df.groupby("method"):
        mean = g.groupby("step")[["gain", "ppl_inc"]].mean()
        ax.plot(mean.gain, mean.ppl_inc, marker="o", color=COLORS[m], label=m.upper())
    ax.set(xlabel="SAMSum ROUGE-L gain", ylabel="WikiText PPL increase", title="Forgetting at matched adaptation (mean of 3 seeds)")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "matched_adaptation.png", dpi=180)
    pd.set_option("display.width", 200)
    print(early.round(3).to_string(index=False))
    print(matched.round(3).to_string(index=False))
    print(df.groupby("method").n.value_counts().to_string())


if __name__ == "__main__":
    main()

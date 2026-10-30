#!/usr/bin/env python
"""Learning-rate frontier (seed 42): gain vs forgetting trajectories for each method and learning rate.

  python scripts/lr_frontier.py --runs v3:2e-4 v4_lr1e-4:1e-4 v4_lr5e-4:5e-4  ->  results/lr_frontier/
Gain uses the full 200 SAMSum examples at the final step and the first 100 at intermediate steps.
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from fapeft.store import read_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ACC = ("boolq", "piqa", "arc", "mmlu")


def collect(tag, lr, seed=42):
    root = ROOT / "runs" / tag
    base_rl = read_json(root / "base/evals/full_step_0000__samsum.json")["rougeL"]
    base_ppl = read_json(root / "base/evals/curve_step_0000__wiki.json")["ppl"]
    base_acc = {p: read_json(root / f"base/evals/full_step_0000__{p}.json")["accuracy"] for p in ACC}
    pts, finals = [], []
    for run in sorted(root.glob(f"seed{seed}_*")):
        method = run.name.split("_", 1)[1]
        ev = run / "evals"
        pts.append(dict(lr=lr, method=method, step=0, gain=0.0, ppl_inc=0.0, kl=0.0))
        for f in sorted(ev.glob("curve_step_*__samsum.json")):
            step = int(re.search(r"step_(\d+)", f.name).group(1))
            wiki = read_json(ev / f.name.replace("samsum", "wiki"))
            full = ev / f"full_step_{step:04d}__samsum.json"
            rl = read_json(full)["rougeL"] if full.exists() else read_json(f)["rougeL"]
            pts.append(dict(lr=lr, method=method, step=step, gain=rl - base_rl, ppl_inc=wiki["ppl"] - base_ppl, kl=wiki["kl_base_to_tuned"]))
        last = pts[-1]
        row = dict(lr=lr, method=method, gain=last["gain"], ppl_inc=last["ppl_inc"], kl=last["kl"])
        for p in ACC:
            row[f"{p}_gap"] = base_acc[p] - read_json(ev / f"full_step_0250__{p}.json")["accuracy"]
        finals.append(row)
    return pts, finals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="tag:lr")
    a = ap.parse_args()
    pts, finals = [], []
    for spec in a.runs:
        tag, lr = spec.split(":")
        p, f = collect(tag, float(lr))
        pts += p
        finals += f
    pts, finals = pd.DataFrame(pts), pd.DataFrame(finals).sort_values(["method", "lr"])
    out = ROOT / "results/lr_frontier"
    out.mkdir(parents=True, exist_ok=True)
    pts.to_csv(out / "trajectories.csv", index=False)
    finals.to_csv(out / "final.csv", index=False)

    colors = {"lora": "#1f77b4", "adalora": "#ff7f0e", "dora": "#2ca02c"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharex=True, sharey=True)
    for ax, (m, g) in zip(axes, pts.groupby("method")):
        for lr, r in g.groupby("lr"):
            r = r.sort_values("step")
            ax.plot(r.gain, r.ppl_inc, marker="o", ms=3, label=f"lr {lr:g}")
        ax.set(title=m.upper(), xlabel="SAMSum ROUGE-L gain")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("WikiText PPL increase")
    axes[0].legend()
    fig.suptitle("Gain vs forgetting along training, by learning rate (seed 42)")
    fig.tight_layout()
    fig.savefig(out / "lr_frontier.png", dpi=170)

    # best retention among all snapshots reaching a given gain, per method (pooled over learning rates)
    rows = []
    for gl in (0.05, 0.07, 0.09, 0.10, 0.11):
        for m, g in pts.groupby("method"):
            ok = g[g.gain >= gl]
            if len(ok):
                best = ok.loc[ok.ppl_inc.idxmin()]
                rows.append(dict(min_gain=gl, method=m, best_ppl_inc=best.ppl_inc, lr=best.lr, step=int(best.step), gain=best.gain))
            else:
                rows.append(dict(min_gain=gl, method=m, best_ppl_inc=float("nan")))
    best = pd.DataFrame(rows)
    best.to_csv(out / "best_retention_at_gain.csv", index=False)
    pd.set_option("display.width", 200)
    print(finals.round(4).to_string(index=False))
    print(best.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

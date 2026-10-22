"""Collect per-step evaluations into a table and learning-curve plots."""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .store import read_json

COLORS = {"lora": "#1f77b4", "adalora": "#ff7f0e", "dora": "#2ca02c"}


def collect(root: Path) -> pd.DataFrame:
    base = {
        "rougeL": read_json(root / "base/evals/curve_step_0000__samsum.json")["rougeL"],
        "ppl": read_json(root / "base/evals/curve_step_0000__wiki.json")["ppl"],
    }
    rows = []
    for run in sorted(root.glob("seed*_*")):
        seed, method = re.fullmatch(r"seed(\d+)_(\w+)", run.name).groups()
        rows.append(dict(seed=int(seed), method=method, step=0, rougeL=base["rougeL"], ppl=base["ppl"], kl=0.0))
        for f in sorted((run / "evals").glob("curve_step_*__samsum.json")):
            step = int(re.search(r"step_(\d+)", f.name).group(1))
            wiki = read_json(f.with_name(f.name.replace("samsum", "wiki")))
            if wiki is None:
                continue
            rows.append(dict(seed=int(seed), method=method, step=step, rougeL=read_json(f)["rougeL"],
                             ppl=wiki["ppl"], kl=wiki["kl_base_to_tuned"]))
    df = pd.DataFrame(rows)
    df["rougeL_gain"] = df.rougeL - base["rougeL"]
    df["ppl_increase"] = df.ppl - base["ppl"]
    return df.sort_values(["method", "seed", "step"]).reset_index(drop=True)


def plot(df: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    panels = [("rougeL_gain", "SAMSum ROUGE-L gain"), ("ppl_increase", "WikiText PPL increase"),
              ("kl", "KL(base || tuned) on WikiText")]
    for ax, (col, title) in zip(axes[:3], panels):
        for m, g in df.groupby("method"):
            for _, run in g.groupby("seed"):
                ax.plot(run.step, run[col], color=COLORS.get(m), alpha=.35, lw=1)
            mean = g.groupby("step")[col].mean()
            ax.plot(mean.index, mean.values, color=COLORS.get(m), lw=2.2, marker="o", label=m.upper())
        ax.set(title=title, xlabel="optimizer step")
        ax.grid(alpha=.25)
    axes[0].legend()
    ax = axes[3]
    for m, g in df.groupby("method"):
        mean = g.groupby("step")[["rougeL_gain", "ppl_increase"]].mean()
        ax.plot(mean.rougeL_gain, mean.ppl_increase, color=COLORS.get(m), marker="o", lw=2, label=m.upper())
    ax.set(title="Forgetting at matched adaptation", xlabel="SAMSum ROUGE-L gain", ylabel="WikiText PPL increase")
    ax.grid(alpha=.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)

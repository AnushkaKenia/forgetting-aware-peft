"""Per-(seed, method) final results from a run directory, in the layout phase5_noise_aware.py reads.

Gaps are base minus tuned accuracy (positive = forgetting); samsum gain and wikitext increase are
tuned minus base, all from the unified protocol (full step-250 probes, 200 SAMSum examples).
"""
import argparse
import re
from pathlib import Path

import pandas as pd

from fapeft.config import Settings
from fapeft.store import read_json

ROOT = Path(__file__).resolve().parents[1]
ACC = ("boolq", "piqa", "arc", "mmlu")


def build(runs: Path, final_step: int) -> tuple[pd.DataFrame, float]:
    ev = lambda d, name: read_json(d / "evals" / name)
    base = runs / "base"
    base_acc = {p: ev(base, f"full_step_0000__{p}.json")["accuracy"] for p in ACC}
    base_rl = ev(base, "full_step_0000__samsum.json")["rougeL"]
    base_ppl = ev(base, "curve_step_0000__wiki.json")["ppl"]
    rows = []
    for run in sorted(runs.glob("seed*_*")):
        seed, method = re.fullmatch(r"seed(\d+)_(\w+)", run.name).groups()
        tag = f"full_step_{final_step:04d}__"
        row = dict(seed=int(seed), method=method,
                   samsum_rl_gain=ev(run, tag + "samsum.json")["rougeL"] - base_rl)
        for p in ACC:
            row[f"{p}_forgetting"] = base_acc[p] - ev(run, tag + f"{p}.json")["accuracy"]
        row["wikitext_ppl_increase"] = ev(run, f"curve_step_{final_step:04d}__wiki.json")["ppl"] - base_ppl
        rows.append(row)
    return pd.DataFrame(rows), base_ppl


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    df, base_ppl = build(ROOT / "runs" / a.tag, Settings().total_steps)
    out = a.out or ROOT / "results" / f"per_seed_{a.tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.round(4).to_string(index=False))
    print(f"base wikitext ppl {base_ppl:.4f}\nwrote {out}")

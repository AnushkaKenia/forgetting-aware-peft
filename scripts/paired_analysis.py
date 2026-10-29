#!/usr/bin/env python
"""Per-example paired analysis of the accuracy probes (CPU, reads saved evals only).

For each (method, probe): per-example change in correctness and in gold log-probability relative to the
untrained model, averaged over seeds, with a bootstrap over EXAMPLES (eval-sampling uncertainty) next to the
SEED spread (training noise). Also the paired SE vs the unpaired binomial SE, the minimum detectable effect
(80% power, 5% level: 2.8 x SE), answer churn, and for BoolQ the Yes-rate before/after (format-bias check).

  python scripts/paired_analysis.py --tag v3     ->  results/paired/{paired_probes,boolq_bias}.csv, REPORT.md
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fapeft.config import Settings  # noqa: E402
from fapeft.data import build_eval_sets, load_sources  # noqa: E402
from fapeft.store import read_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROBES = ["boolq", "piqa", "arc", "mmlu"]
B = 10_000
rng = np.random.default_rng(0)


def boot_ci(per_example, n_boot=B):
    n = len(per_example)
    idx = rng.integers(0, n, (n_boot, n))
    means = per_example[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    a = ap.parse_args()
    root = Path(a.runs_dir) / a.tag
    out = ROOT / "results/paired"
    out.mkdir(parents=True, exist_ok=True)
    ld = lambda d, n: read_json(d / "evals" / n)

    runs = {}
    for r in sorted(root.glob("seed*_*")):
        seed, method = re.fullmatch(r"seed(\d+)_(\w+)", r.name).groups()
        runs.setdefault(method, {})[int(seed)] = r

    rows = []
    for p in PROBES:
        base = ld(root / "base", f"full_step_0000__{p}.json")
        bc, bg = np.array(base["correct"], float), np.array(base["gold_logprob"], float)
        for method, by_seed in runs.items():
            tc = np.array([ld(d, f"full_step_0250__{p}.json")["correct"] for d in by_seed.values()], float)
            tg = np.array([ld(d, f"full_step_0250__{p}.json")["gold_logprob"] for d in by_seed.values()], float)
            d_acc = (bc[None] - tc)            # positive = forgetting, per seed and example
            d_gold = (bg[None] - tg)           # positive = gold answer became less likely
            ex_acc, ex_gold = d_acc.mean(0), d_gold.mean(0)
            lo_a, hi_a = boot_ci(ex_acc)
            lo_g, hi_g = boot_ci(ex_gold)
            seed_means = d_acc.mean(1)
            p0 = float(bc.mean())
            flips = ((bc[None] == 1) & (tc == 0)).sum(1) + ((bc[None] == 0) & (tc == 1)).sum(1)
            rows.append(dict(probe=p, method=method, base_acc=p0, gap_acc=ex_acc.mean(),
                             gap_ci_examples_low=lo_a, gap_ci_examples_high=hi_a,
                             gap_seed_sd=seed_means.std(ddof=1), se_paired=ex_acc.std(ddof=1) / np.sqrt(len(ex_acc)),
                             se_unpaired_binomial=np.sqrt(2 * p0 * (1 - p0) / len(bc)),
                             mde80_paired=2.8 * ex_acc.std(ddof=1) / np.sqrt(len(ex_acc)),
                             churn_per_seed=float(flips.mean()) / len(bc),
                             gap_gold_logprob=ex_gold.mean(), gold_ci_low=lo_g, gold_ci_high=hi_g,
                             gold_z=ex_gold.mean() / (ex_gold.std(ddof=1) / np.sqrt(len(ex_gold))),
                             acc_z=ex_acc.mean() / max(ex_acc.std(ddof=1) / np.sqrt(len(ex_acc)), 1e-12)))
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "paired_probes.csv", index=False)

    # BoolQ Yes-rate before/after: needs the gold labels
    s = Settings()
    evalsets = build_eval_sets(load_sources(), s)
    y = np.array([bool(x["answer"]) for x in evalsets["boolq"]])
    base_c = np.array(ld(root / "base", "full_step_0000__boolq.json")["correct"], bool)
    pred_yes = lambda c: np.where(y, c, ~c)    # correct on a Yes example => predicted Yes; wrong on a No example => predicted Yes
    brow = [dict(method="base", seed=0, yes_rate=pred_yes(base_c).mean(), acc_on_yes=base_c[y].mean(), acc_on_no=base_c[~y].mean())]
    for method, by_seed in runs.items():
        for seed, d in by_seed.items():
            c = np.array(ld(d, "full_step_0250__boolq.json")["correct"], bool)
            brow.append(dict(method=method, seed=seed, yes_rate=pred_yes(c).mean(), acc_on_yes=c[y].mean(), acc_on_no=c[~y].mean()))
    bias = pd.DataFrame(brow)
    bias.to_csv(out / "boolq_bias.csv", index=False)
    print(f"BoolQ gold Yes-rate {y.mean():.3f}")
    print(bias.round(3).to_string(index=False))
    pd.set_option("display.width", 220)
    print(tab.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Update geometry of every adapter snapshot (CPU only).

For each targeted weight, the effective update is dW = W_merged - W_base, computed by merging the
adapter, so LoRA, DoRA (magnitude re-scaling included) and AdaLoRA are measured the same way.
Per snapshot we record: relative Frobenius norm ||dW||/||W||, stable rank of dW, and a simplified
intruder-dimension count (Shuttleworth et al. 2024): top-K left singular vectors of W+dW whose
maximum cosine similarity to all left singular vectors of W is below EPS.

  python scripts/delta_w_analysis.py --tag v3
Output: <runs>/<tag>/delta_w.csv (per module) and results/delta_w/summary.csv (per snapshot).
"""
import argparse
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from fapeft.config import Settings  # noqa: E402

STEPS = [5, 25, 50, 100, 250]
K, EPS = 10, 0.5
ROOT = Path(__file__).resolve().parents[1]


def load_base(name):
    return AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    s = Settings()
    root = Path(a.runs_dir) / a.tag
    out_csv = root / "delta_w.csv"
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    rows = done.to_dict("records")
    seen = {(r["seed"], r["method"], r["step"]) for r in rows}

    base = load_base(s.base_model)
    targets = [n for n, p in base.named_parameters()
               if p.ndim == 2 and any(f".{t}." in n for t in s.target_modules)]
    w0 = {n: dict(base.named_parameters())[n].detach().clone() for n in targets}
    u0 = {n: torch.linalg.svd(w0[n], full_matrices=False)[0] for n in targets}
    del base

    for run in sorted(root.glob("seed*_*")):
        seed, method = re.fullmatch(r"seed(\d+)_(\w+)", run.name).groups()
        for step in STEPS:
            if (int(seed), method, step) in seen:
                continue
            print(f"[{run.name}] step {step}", flush=True)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The following rank_pattern keys did not match")
                m = PeftModel.from_pretrained(load_base(s.base_model), str(run / "snapshots" / f"step_{step:04d}"))
            merged = dict(m.merge_and_unload().named_parameters())
            for n in targets:
                w1 = merged[n].detach()
                dw = w1 - w0[n]
                sv = torch.linalg.svdvals(dw)
                fro = dw.norm().item()
                u1 = torch.linalg.svd(w1, full_matrices=False)[0][:, :K]
                maxcos = (u1.T @ u0[n]).abs().max(dim=1).values
                mod = re.search(r"layers\.(\d+)\..*\.(\w+_proj)\.weight", n)
                rows.append(dict(seed=int(seed), method=method, step=step, layer=int(mod.group(1)), module=mod.group(2),
                                 rel_fro=fro / w0[n].norm().item(), stable_rank=(fro ** 2 / sv[0].item() ** 2) if fro > 0 else 0.0,
                                 intruders=int((maxcos < EPS).sum())))
            del m, merged
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    summ = (df.groupby(["seed", "method", "step"])
              .agg(rel_fro_mean=("rel_fro", "mean"), stable_rank_mean=("stable_rank", "mean"), intruders_total=("intruders", "sum"))
              .reset_index())
    (ROOT / "results/delta_w").mkdir(parents=True, exist_ok=True)
    summ.to_csv(ROOT / "results/delta_w/summary.csv", index=False)
    print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

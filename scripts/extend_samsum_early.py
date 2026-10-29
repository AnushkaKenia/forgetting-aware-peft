#!/usr/bin/env python
"""Evaluates SAMSum examples 100-199 at curve snapshots, so early-step ROUGE-L uses all 200
held-out examples (curve points only used the first 100). Resumable: one JSON per snapshot
(`ext_step_NNNN__samsum.json`, ignored by aggregate_curves.py).

  caffeinate -dimsu python scripts/extend_samsum_early.py --tag v3
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from fapeft import evals  # noqa: E402
from fapeft.config import Settings, pick_device  # noqa: E402
from fapeft.pipeline import Context, release  # noqa: E402
from fapeft.store import cached_json  # noqa: E402

STEPS = {"lora": [5, 10, 15, 25, 35, 50, 100], "dora": [5, 10, 15, 25, 35, 50, 100],
         "adalora": [50, 100, 150, 200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--runs-dir", default=str(Path(__file__).resolve().parents[1] / "runs"))
    a = ap.parse_args()
    s, root = Settings(), Path(a.runs_dir) / a.tag
    ctx = Context(s, root, pick_device())
    tail = ctx.evalsets["samsum_eval"].select(range(s.curve_samsum_n, s.samsum_eval_n))
    for run in sorted(root.glob("seed*_*")):
        method = run.name.split("_", 1)[1]
        for step in STEPS[method]:
            out = run / "evals" / f"ext_step_{step:04d}__samsum.json"
            if out.exists():
                continue
            print(f"[{run.name}] step {step}", flush=True)
            model = ctx.load_snapshot(run / "snapshots" / f"step_{step:04d}")
            try:
                cached_json(out, lambda: evals.eval_samsum(
                    model, ctx.tok, ctx.device, tail, len(tail), s.samsum_batch, s.max_len, s.samsum_max_new_tokens))
            finally:
                del model
                release(ctx.device)
    print("done")


if __name__ == "__main__":
    main()

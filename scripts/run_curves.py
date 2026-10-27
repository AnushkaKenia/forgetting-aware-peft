#!/usr/bin/env python
"""Resumable learning-curve pipeline. Safe to kill and re-run at any time.

  python scripts/run_curves.py --tag v2 --seeds 42 --methods lora adalora dora
  python scripts/run_curves.py --tag smoke --smoke --seeds 42 --methods lora
  python scripts/run_curves.py --tag v2 --seeds 42 --stages eval      # evaluate only
  python scripts/run_curves.py --tag v2 --seeds 42 --stages early eval  # back-fill early snapshots

On a Mac keep it awake and plugged in:  caffeinate -dimsu python scripts/run_curves.py ...
On Colab pass --runs-dir /content/drive/MyDrive/fapeft_runs so progress lives on Drive.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from fapeft.config import METHODS, Settings  # noqa: E402
from fapeft.pipeline import SMOKE, parse_overrides, run_pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="run-set name; results go to <runs-dir>/<tag>/")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--runs-dir", default=str(Path(__file__).resolve().parents[1] / "runs"))
    ap.add_argument("--stages", nargs="+", default=["train", "eval"], choices=["train", "early", "eval"])
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--smoke", action="store_true", help="tiny settings to check the pipeline end to end")
    ap.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="override Settings fields")
    a = ap.parse_args()

    s = Settings()
    if a.smoke:
        s = s.with_overrides(**SMOKE)
    s = s.with_overrides(**parse_overrides(a.set))
    run_pipeline(s, a.seeds, a.methods, a.tag, Path(a.runs_dir), tuple(a.stages),
                 torch.device(a.device) if a.device else None)


if __name__ == "__main__":
    main()

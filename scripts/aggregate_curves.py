#!/usr/bin/env python
"""Build curves.csv and curves.png from runs/<tag>/ (works on partial results)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fapeft.report import collect, plot  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--runs-dir", default=str(Path(__file__).resolve().parents[1] / "runs"))
a = ap.parse_args()
root = Path(a.runs_dir) / a.tag
df = collect(root)
df.to_csv(root / "curves.csv", index=False)
plot(df, root / "curves.png")
print(df.round(4).to_string(index=False))
print(f"\nwrote {root / 'curves.csv'} and {root / 'curves.png'}")

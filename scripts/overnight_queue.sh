#!/bin/bash
# Overnight queue: finish Phase A evals, then the learning-rate sweep (B1), seed 42. Re-run to resume.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
echo "=== $(date) extend early evals"; $PY scripts/extend_samsum_early.py --tag v3 >> runs/v3/extend.log 2>&1
echo "=== $(date) matched adaptation"; $PY scripts/matched_adaptation.py --tag v3 > results/matched_adaptation/stdout.txt 2>&1
for lr in 5e-4 1e-4 5e-5; do
  tag="v4_lr$lr"
  mkdir -p runs/$tag && [ -d runs/$tag/base ] || cp -R runs/v3/base runs/$tag/base
  echo "=== $(date) lr sweep $tag"
  $PY scripts/run_curves.py --tag $tag --seeds 42 --methods lora adalora dora --set lr=$lr >> runs/$tag/pipeline.log 2>&1
done
echo "=== $(date) queue finished"

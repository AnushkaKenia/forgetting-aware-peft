# Forgetting-aware PEFT

How much do LoRA, AdaLoRA and DoRA forget when fine-tuned on one task, and can forgetting be
reduced without losing what the adapter learns? Base model Qwen2.5-0.5B, fine-tuned on SAMSum
(dialogue summarisation), forgetting measured on BoolQ, PIQA, ARC-Easy, MMLU and WikiText-2.

## Layout

| Path | Contents |
|---|---|
| `0*_*.ipynb` | Original Colab notebooks, Phases 1-5 (kept as the historical record) |
| `results/` | Results from those notebooks; `results/phase5_v2/` is the noise-aware re-analysis |
| `src/fapeft/` | Resumable training + evaluation pipeline (this is what new experiments use) |
| `scripts/run_curves.py` | Learning-curve runner: train, snapshot every 50 steps, evaluate each snapshot |
| `scripts/aggregate_curves.py` | Builds `curves.csv` / `curves.png` from a run directory |
| `scripts/phase5_noise_aware.py` | CPU-only statistics on the 9 existing runs |
| `data/indices/` | Fixed evaluation subsets, so every model sees identical examples |
| `tests/` | Crash/resume tests (tiny model, CPU, ~10 s) |

## Setup and running

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q                                       # resume-correctness tests
.venv/bin/python scripts/run_curves.py --tag smoke --smoke --methods lora   # a few minutes end to end
caffeinate -dimsu .venv/bin/python scripts/run_curves.py --tag v2 --seeds 42   # real run, laptop
.venv/bin/python scripts/aggregate_curves.py --tag v2               # works on partial results
```

Device is picked automatically (CUDA, then Apple MPS, then CPU). On Colab, install the same way and
pass `--runs-dir /content/drive/MyDrive/fapeft_runs` so progress survives a disconnect. Do not split one
curve across machines: evaluate every point of a curve on the same device.

## Save points (kill it anywhere, re-run the same command)

* **Trainer checkpoint every 25 steps** (`runs/<tag>/seedS_method/ckpt`): weights, optimizer, RNG state.
* **Adapter snapshot every 50 steps** (`.../snapshots/step_NNNN`), kept for the learning curve.
* **One JSON per (step, probe)** (`.../evals`), written atomically; existing files are skipped.
* **`done.json`** marks a finished run; finished runs are skipped.
* AdaLoRA's pruning state is not in Trainer checkpoints, so `AdaLoraStateCallback` saves it and
  `FullRankCheckpointTrainer` keeps checkpoints full-size. `tests/test_resume.py` kills AdaLoRA before and
  after its final prune and checks the resumed run matches an uninterrupted one.

A crash loses at most 25 optimizer steps (roughly 3 minutes on an M4, extrapolated from smoke timings).

## Evaluation protocol (one, for every seed)

SAMSum ROUGE-1/2/L with stemming, greedy decoding, 96 new tokens, 200 held-out examples (first 100 at
intermediate points). BoolQ/PIQA/ARC/MMLU: 500 fixed examples each, mean continuation log-likelihood.
WikiText-2: perplexity on the first 50k tokens, plus per-token KL(base || tuned). Per-example outcomes are
saved so paired significance tests are possible.

**Why this exists:** the original 3-seed table mixed two protocols. Seed 42 (Phases 1-3) used 64 generation
tokens, unstemmed ROUGE and a different BoolQ prompt; seeds 123 and 2026 (Phase 4) used 96 tokens, stemmed
ROUGE and another BoolQ prompt. The new pipeline reruns every seed under the single protocol above.
Untrained-model numbers on MPS match the earlier CUDA runs (BoolQ .620, PIQA .676, ARC .716, MMLU .456,
WikiText PPL 17.20 vs 17.20).

## Phase 5 v2 findings (`results/phase5_v2/REPORT.md`)

The original composite min-max-scaled differences smaller than measurement noise to a full 0-1 range, which
ranked AdaLoRA as worst on retention although it changes nothing measurable. With noise-aware statistics
(n = 3 seeds, t-intervals, paired tests, Holm correction):

* WikiText perplexity is the only probe with a detectable effect: LoRA +0.88, DoRA +0.97, AdaLoRA about 0.
* No method shows detectable forgetting on BoolQ, ARC or MMLU; LoRA's PIQA gain is real on this fixed
  eval set but small (about 1.7 points).
* AdaLoRA is not dominated: it forgets significantly less and learns significantly less. LoRA and DoRA are
  statistically tied. The seed intervals cover training noise on a fixed eval set; example-sampling noise
  (about +/-3 points at n = 500) is additional.

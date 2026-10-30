# Roadmap: from experiment to publishable study

Target: a rigorous empirical paper (TMLR-style, arXiv first). Working claim to test:
**"Forgetting differences between PEFT methods are largely explained by how much each method learns and
by hyperparameters, and common forgetting metrics are too noisy to show it."**
Everything below is ordered by cost: free first (existing data), then laptop, then free Colab.

## Status (updated 2026-10-30)

| Item | Status |
|---|---|
| 3 seeds x 3 methods, unified protocol (`runs/v3`) | **Done**, results in `results/curves_3seed`, `results/phase5_v3` |
| A1 matched adaptation, A2 early stopping, A3 delta-W, A4/A5 paired + noise floor | **Done**, `results/matched_adaptation`, `delta_w`, `paired` (findings in README) |
| B1 learning-rate sweep, seed 42 | lr 1e-4, 5e-4 **done** (with 2e-4 baseline), `results/lr_frontier`. **lr 5e-5 not run.** |
| B2 rank/alpha sweep, B3 mitigation baselines, B4 seeds 4-5, B5 | Not started |
| C0-C5 (Colab, second model/task, generation probes) | Not started |
| D method + writing | Not started |

**Findings so far** (details in README): forgetting is mostly a function of how much the model adapts, not of the
method or learning rate; AdaLoRA's retention advantage disappears at matched gain; early stopping at steps 5-25 is a
strong baseline; BoolQ "improvement" is a Yes-bias shift; ARC/MMLU gold log-probability drops while accuracy does not.

## Handing tasks to a teammate

Good tasks to hand off, independent of each other (each writes only to its own `runs/<tag>` folder and its own
`results/<name>` folder, so they do not conflict):

| Task | Where to run | Rough time | Command / note |
|---|---|---|---|
| B1: lr 5e-5, seed 42 | laptop or Colab | ~4 h (laptop) | see "Running a sweep" below |
| B2: LoRA rank/alpha sweep (r = 4, 16, 32, alpha = 2r) | laptop or Colab | ~1 h per run | `--set lora_r=16 lora_alpha=32`, tags `v5_r16` etc., LoRA only |
| B4: seeds 4 and 5 for the 2e-4 setup | laptop or Colab | ~3.5 h per seed (laptop) | `--seeds 4 5`, new tag, copy base evals |
| C0: Colab notebook + one benchmark run | Colab | a day | clone repo, `pip install -e .`, `--runs-dir /content/drive/MyDrive/fapeft_runs` |
| New analysis code (B3 KL/replay options in `train.py`, task abstraction) | any | days | needs the tests in `tests/` to keep passing |

### Running a sweep
```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]" && .venv/bin/python -m pytest -q
mkdir -p runs/v4_lr5e-5 && cp -R results/base_evals runs/v4_lr5e-5/base          # reuse the untrained-model evals (laptop/MPS numbers)
caffeinate -dimsu .venv/bin/python scripts/run_curves.py --tag v4_lr5e-5 --seeds 42 --methods lora adalora dora --set lr=5e-5
```
Re-running the same command resumes after a crash or disconnect (at most 25 steps are lost).

### Rules so results stay comparable
- `runs/` is git-ignored (large). Share finished runs through Drive or a zip; commit only the small `results/` files.
- **Do not mix devices within one comparison.** A run's `base/` evaluations must come from the same device as its
  snapshots. Colab's untrained-model numbers must match the laptop's (WikiText PPL 17.20, BoolQ .620, PIQA .676,
  ARC .716, MMLU .456) before cross-device results are used; if not, evaluate every compared run on one device.
- Keep the evaluation protocol unchanged (200 SAMSum examples, 500 per accuracy probe, fixed indices in `data/indices`).
- Use a new tag for every run set and one branch per task; open a PR into the working branch.
- The untrained-model evaluations are committed in `results/base_evals` (80 KB, from the laptop). Copy them to
  `runs/<tag>/base` to skip the 7-minute base evaluation. On a different device (e.g. Colab) skip the copy: the pipeline
  evaluates the base model automatically when `base/` is missing, so the comparison stays on one device.

Time estimates use measured laptop (M4 Air, MPS) numbers: train LoRA ~36 min, AdaLoRA ~40, DoRA ~70-75,
plus ~20-30 min evaluation per run (full pipeline, 11 curve points). Colab T4 speeds are **not measured**;
benchmark one run before planning around them.

## Phase A: no new training (laptop, hours of CPU/MPS, a day of work)

| # | Task | Output | Decision it informs |
|---|---|---|---|
| A1 | **Matched-adaptation analysis.** Interpolate each run's (ROUGE-L gain, PPL increase, KL) curve; compare methods at equal gain. Observed so far: LoRA at step 5 (+0.061 gain, +0.001 PPL) vs AdaLoRA at step 250 (+0.051, +0.001). Re-evaluate early snapshots (steps 5-100) on 200+ SAMSum examples because 100-example ROUGE has seed sd up to 0.02. | `results/matched_adaptation/` | Whether the central claim holds. **Gate for everything else.** |
| A2 | **Early-stopping baseline table** from existing curves: LoRA/DoRA stopped at steps 5, 10, 15, 25, 35, 50 vs final. | table + plot | Strongest simple baseline any method must beat. |
| A3 | **Update-geometry analysis** from saved snapshots: ‖ΔW‖ per module, singular spectrum, intruder dimensions (Shuttleworth et al.), correlation with KL and PPL increase. | `results/delta_w/` | Mechanism: why AdaLoRA drifts so little. |
| A4 | **Per-example paired analysis**: use saved per-example correctness and gold log-prob; bootstrap over examples and seeds (or mixed effects). Check whether the BoolQ/PIQA "improvements" are a Yes/No or format bias shift. | `results/paired/` | Stop calling it "no forgetting" if it is a calibration effect. |
| A5 | **Noise-floor study**: how many examples a probe needs to detect a 1-point shift; compare accuracy vs gold log-prob vs KL sensitivity. | figure | Core of the measurement-paper angle. |

Gate after A: if A1 confirms the claim, proceed with Phases B and C as the measurement/dynamics paper.
If not, pivot to the mechanism/method angle (A3 plus Phase B3).

## Phase B: new experiments on the laptop (0.5B model, resumable)

| # | Task | Runs | Laptop time | Notes |
|---|---|---|---|---|
| B1 | **Learning-rate sweep**, lr in {5e-5, 1e-4, 5e-4} (2e-4 done), 3 methods, seed 42 first | 9 | ~11-12 h | Run each lr as its own tag via `--set lr=...`; copy `base/` into each tag. Compare **frontiers** (gain vs forgetting), not points. Biggest design gap reviewers will cite. |
| B2 | **Rank/alpha sweep**, LoRA only, r in {4, 16, 32}, alpha=2r | 3 | ~3.5 h | Controls the intruder-dimension story. |
| B3 | **Mitigation baselines on LoRA**: KL-to-base penalty (beta 0.01, 0.1, via `disable_adapter`), replay of generic text, KL+replay, self-generated replay | ~6 | ~10 h | Needs new code (extra forward pass, about 1.5-2x step cost). These are baselines from published work, not our contribution. |
| B4 | **Seeds 4 and 5** for the headline config, 3 methods | 6 | ~7 h | Makes intervals less wide (t-critical 4.30 at n=3 vs 2.78 at n=5). |
| B5 | Multi-seed for any lr/rank point that turns out to matter | variable | variable | Only after B1/B2 results. |

Total Phase B: about 30-35 h of laptop time, spread over several overnights.
Run order: B1 (seed 42) -> look at frontiers -> B2 -> B3 -> B4.

## Phase C: free Colab (T4, 16 GB, ~12 h sessions, can disconnect)

Use `--runs-dir /content/drive/MyDrive/fapeft_runs`; the pipeline resumes from the last checkpoint.
Keep every curve and its base evaluation on the same device; run a base-model evaluation on Colab and
check it matches the laptop numbers (PPL 17.20 on MPS) before trusting cross-device comparisons.

| # | Task | Notes |
|---|---|---|
| C0 | Colab notebook that clones the repo, installs, mounts Drive, runs `run_curves.py`; benchmark one LoRA run for time and fp16 behaviour (T4 has no bf16) | Do this first; everything below depends on it. |
| C1 | **Second model size**: Qwen2.5-1.5B, LoRA/AdaLoRA/DoRA, seed 42 first, then more seeds | Fits in memory; expect several hours per run. |
| C2 | **Second model family**: Llama-3.2-1B or Gemma (gated: needs a Hugging Face login and license acceptance) | Shows the result is not Qwen-specific. |
| C3 | **Second task**: GSM8K (math) training subset, same pipeline | Needs a task module (data, prompt, scorer). |
| C4 | **Generation-based forgetting probes** (e.g. GSM8K accuracy for the SAMSum-tuned model, IFEval or a small instruction-following set) | Log-likelihood multiple choice is weakly sensitive. Run on laptop too for 0.5B. |
| C5 | Base vs instruct variants of one model | Matches setups in recent replay papers. |

## Phase D: method (only if justified) and writing

1. Decide after A1 and B3. If early stopping or plain KL already matches everything, the paper is the
   measurement/dynamics study and the method is not needed.
2. If a method is worth trying: forgetting-aware AdaLoRA importance (use KL-to-base sensitivity in the rank
   allocator). Must beat early stopping, KL penalty, KL+replay and LaLoRA at matched adaptation.
3. Writing: related work (Biderman 2025, Shuttleworth 2025, KL+replay 2025, LaLoRA, CLoRA, self-generated
   replay 2026), protocol appendix, release code and per-example data.

## Code work this needs

- `scripts/matched_adaptation.py`, `scripts/delta_w_analysis.py`, `scripts/paired_analysis.py` (Phase A).
- A seed-and-lr-aware aggregation (current `aggregate_curves.py` assumes one config per tag).
- KL-penalty and replay options in `train.py` (Phase B3).
- Task and model abstraction for C2-C4 (currently SAMSum and Qwen-specific).
- Colab notebook (C0).

## Milestones

1. Phase A finished and gate decided: about 1-2 days.
2. B1 plus C0/C1 started in parallel (laptop and Colab): about 1 week.
3. B2-B4 and C2-C4: 2-3 more weeks.
4. Draft: after the frontiers and second model exist.

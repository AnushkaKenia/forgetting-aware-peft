# Roadmap: from experiment to publishable study

Target: a rigorous empirical paper (TMLR-style, arXiv first). Working claim to test:
**"Forgetting differences between PEFT methods are largely explained by how much each method learns and
by hyperparameters, and common forgetting metrics are too noisy to show it."**
Everything below is ordered by cost: free first (existing data), then laptop, then free Colab.

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

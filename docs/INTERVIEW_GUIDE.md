# Interview guide: Forgetting-aware PEFT

How to use this: read section 1 until you can say it in 60 seconds, learn the design-decision table (section 5)
and the numbers (section 7), then rehearse the Q&A (section 10). Every number here comes from the repo
(README, `results/`, `src/fapeft/`). Where something is a limitation, say so; being honest about limits is
what interviewers reward.

---

## 1. The 60-second pitch

> "I studied how much three parameter-efficient fine-tuning methods (LoRA, AdaLoRA, DoRA) forget when you
> fine-tune Qwen2.5-0.5B on one task, SAMSum dialogue summarisation. My starting question was which method
> forgets least. The answer I ended up with is different: **forgetting is mostly a function of how far the
> weights move, not of which method you use.** AdaLoRA looks like it 'forgets less', but it simply learns less.
> At equal task gain, LoRA stopped at step 5 retains exactly as much as AdaLoRA at step 250. I also found that
> the usual forgetting probes are too coarse to see this: accuracy on 500 examples hides drift that
> log-probability and KL divergence show clearly. The work runs on a laptop with a resumable pipeline, and I'm
> turning it into a paper."

Three takeaways to land: (1) matched-adaptation reframing, (2) measurement noise and better metrics,
(3) engineering for reproducibility (resumable pipeline, unified protocol).

---

## 2. Background you must be able to explain

| Concept | One-liner |
|---|---|
| Catastrophic forgetting | Fine-tuning on task B degrades what the model could do before (task A / general ability). |
| PEFT | Freeze the base weights, train a small number of extra parameters. Cheaper, and often assumed to forget less. |
| LoRA | Learn a low-rank update: W' = W + (alpha/r)·B·A, with A (r×k), B (d×r). Here r=8, alpha=16, dropout .05, on all 7 linear projections (q,k,v,o,gate,up,down). 4.4M trainable params of 498M (0.88%). |
| AdaLoRA | Parametrises the update as an SVD-like P·Λ·Q and **prunes singular values by importance** during training, so the rank budget is reallocated across layers. Here init rank 12 → target 8, with warm-up (tinit 50), pruning every 10 steps, final phase (tfinal 50). 6.6M params before pruning. |
| DoRA | Splits the weight into **magnitude and direction**; LoRA updates the direction, a separate learned vector handles magnitude. 4.7M params. About 2x slower to train here (63 min vs 34 for LoRA). |
| Perplexity (PPL) | exp(mean negative log-likelihood) on held-out text. Used on WikiText-2 as a proxy for general language modelling. Base: 17.20. |
| KL(base ‖ tuned) | Average per-token divergence between the base model's next-token distribution and the tuned model's, on WikiText. A continuous, label-free measure of how far behaviour has moved. |
| ROUGE-L | Overlap-based summarisation metric (longest common subsequence F1). Task-gain measure. |
| Log-likelihood multiple choice | For BoolQ/PIQA/ARC/MMLU, score each candidate answer by its mean log-probability under the model and pick the highest. No generation parsing needed. |
| Paired vs unpaired | Comparing the same 500 examples before/after (paired) removes example-difficulty variance, so the standard error is far smaller than comparing two independent samples. |

---

## 3. The research question and how it evolved

1. **Original question:** which PEFT method has the best adaptation-versus-retention trade-off?
2. **Notebook phases 1–5 (Colab, T4):** LoRA baseline → three-method comparison → multi-capability forgetting
   probes → multi-seed robustness → a composite "forgetting-aware trade-off" score.
3. **Problems found while re-auditing my own results** (this is the interesting part):
   - The composite score min-max scaled tiny differences to a 0–1 range, which **ranked AdaLoRA worst on
     retention even though it changes nothing measurable**. Noise got amplified into a ranking.
   - The 3 seeds were **evaluated under two different protocols** (64 vs 96 generation tokens, stemmed vs
     unstemmed ROUGE, different BoolQ prompts). Not comparable.
   - A run that was resumed after a crash on Apple GPU **silently diverged** from an uninterrupted run (RNG state).
4. **Fixes:** noise-aware statistics (t-intervals, paired tests, Holm correction), one unified evaluation
   protocol rerun for all seeds, a resumable pipeline with the MPS RNG fix.
5. **New hypothesis, tested in Phase A:** *forgetting differences between methods are largely explained by how
   much each method adapts, and standard metrics are too noisy to show it.*

Be ready to say: "I discovered my own earlier analysis was misleading, and rebuilt it." That is a strength.

---

## 4. System design: the pipeline (`src/fapeft/`)

| File | Responsibility |
|---|---|
| `config.py` | Frozen `Settings` dataclass (all hyper-parameters); `build_peft()` constructs LoRA/DoRA/AdaLoRA configs. `total_steps = ceil(2000 / (2×8)) × 2 = 250`. |
| `data.py` | Datasets, **fixed evaluation index files** (`data/indices/`) so every model sees identical examples; per-seed training subsets; target-only loss tokenisation. |
| `train.py` | Resumable training with callbacks (below). |
| `evals.py` | One implementation of every probe, used identically for the base model and every snapshot. |
| `pipeline.py` | Run queue: for each (seed, method) train → snapshot → evaluate. Each stage skips work that already exists. |
| `store.py` | Atomic JSON read/write (`cached_json`). |
| `report.py` + `scripts/` | Aggregation, statistics, and Phase A analyses. |
| `tests/` | Crash/resume tests on a tiny model on CPU (~10 s). |

### Save points ("kill it anywhere, re-run the same command")
- Trainer checkpoint every **25 steps** (weights, optimizer, RNG): a crash loses at most 25 steps (~3 min).
- Adapter **snapshots** at steps 5, 10, 15, 25, 35, 50, 100, 150, 200, 250 (early ones dense because that is where the action is).
- **One JSON per (step, probe)**, written atomically (write temp file then `os.replace`); a rerun skips existing files.
- `done.json` marks a finished run.
- Snapshot callback runs **before** the checkpoint for the same step, so a checkpoint can never exist without its snapshot.

### Non-obvious engineering problems solved (great interview stories)
1. **AdaLoRA resume.** Its pruning state (importance moving averages, rank pattern) is not in a normal Trainer
   checkpoint. A resume would silently restart the importance estimates, or crash after the final prune.
   → `AdaLoraStateCallback` saves/restores it; `FullRankCheckpointTrainer` keeps checkpoints un-pruned (PEFT
   prunes on save, but the live model keeps full-size masked tensors). Verified by a test that kills AdaLoRA
   before and after its final prune and checks the result matches an uninterrupted run.
2. **MPS RNG bug.** HF Trainer checkpoints Python/NumPy/CPU/CUDA RNG but not Apple MPS RNG, so dropout masks
   changed after a resume. I found it by **replaying** a resumed run: LoRA (which had resumed) differed from its
   replay by 0.0024 max weight difference (weights up to 0.039), while AdaLoRA and DoRA (never resumed) matched
   exactly (0.0). CPU tests did not catch it because the bug is device-specific. → `MpsRngCallback`.
   Lesson: equivalence tests must run on the device you actually use.
3. **MPS memory growth.** The MPS allocator caches blocks per tensor shape; with variable-length batches the
   cache grew past RAM (17 GB wired, swap, step time 5 s → 80 s). → release the cache after every micro-batch.
4. **Early-step back-fill.** I realised the early steps matter after the main runs finished, so
   `train_prefix` replays the first N steps (same LR schedule spanning the full run, then stop early) and
   verifies the replay against the original snapshot at step 50 (stored in `early/verify.json`).
5. **Single-device rule.** Never mix devices within one curve; untrained-model numbers on MPS matched the earlier
   CUDA numbers (PPL 17.20 = 17.20, BoolQ .620, PIQA .676, ARC .716, MMLU .456), which validated the laptop setup.

---

## 5. Design decisions and why (the table interviewers dig into)

| Decision | Choice | Why | Trade-off / what you would say if pushed |
|---|---|---|---|
| Base model | Qwen2.5-0.5B (base, not instruct) | Fits a laptop; fast iteration; base model avoids instruction-tuning confounds | Small model; results may not transfer to larger/instruct models → planned 1.5B and a second family |
| Fine-tuning task | SAMSum summarisation, 2,000 examples, 2 epochs, 250 optimizer steps | Generation task with clear metric (ROUGE) far from the probes' format | Single task; planned GSM8K as second |
| Effective batch | 2 × 8 grad-accum = 16 | Memory on a laptop | – |
| Learning rate | 2e-4 (main), then sweep 1e-4, 5e-4 | Common LoRA default; sweep added because reviewers would say the comparison is lr-confounded | Sweep is seed 42 only; 5e-5 not run |
| Rank / alpha | r=8, alpha=16, all 7 modules | Standard; same for all methods for fairness | Rank sweep (B2) not yet done; AdaLoRA budget (12→8) is not strictly equal to LoRA's |
| Loss | Target-only (prompt tokens masked with -100) | Learn to summarise, not to repeat dialogues | – |
| Probes | BoolQ, PIQA, ARC-Easy, MMLU, WikiText-2 | Cover reading comprehension, commonsense, science, knowledge, language modelling; all disjoint from SAMSum | Multiple-choice by log-likelihood is weakly sensitive; generation probes (GSM8K, IFEval) are planned |
| Scoring | Mean continuation log-prob, no generation parsing | Removes format-parsing noise; same code for all models | Prompt format effects (see BoolQ yes-bias) |
| Eval sizes | 500 per accuracy probe, 200 SAMSum, 50k WikiText tokens | Laptop time budget | Unpaired binomial noise about ±3 points; mitigated by pairing |
| Fixed eval indices | Saved to `data/indices/` | Every model state sees identical examples so paired tests are valid | – |
| Extra metrics | Gold-answer log-prob and KL(base‖tuned) | Continuous, much more sensitive than accuracy | KL measured on WikiText only |
| KL without a second model | `model.disable_adapter()` gives the base distribution | Saves memory, guarantees identical weights/tokenizer | – |
| Statistics | n=3 seeds, Student-t intervals (2 d.f.; t-crit 4.30), paired t-tests, **Holm correction** | Honest about small n; controls false positives across many tests | Low power; hence B4 plan for seeds 4–5 (t-crit 2.78 at n=5) |
| Composite score | **Dropped** as the headline | Min-max scaling amplified noise (see section 3) | Replaced by Pareto analysis that only declares dominance when significant |
| Protocol | One protocol for every seed; seeds rerun | Original table mixed protocols | Costs ~3.5 h per seed on a laptop |
| Reproducibility | Resumable checkpoints + atomic evals + tests | Laptop gets closed, Colab disconnects | Adds code complexity (AdaLoRA state) |
| Hardware | M4 MacBook Air (MPS), caffeinate; Colab later | Free and available | Do not mix devices within a comparison |

---

## 6. Experiments, in order

| # | What | Why | Result |
|---|---|---|---|
| Phases 1–5 (Colab notebooks) | Original LoRA / PEFT comparison / forgetting probes / multi-seed / trade-off score | Initial study | Superseded by noise-aware re-analysis (kept as historical record) |
| Phase 5 v2 | Re-analysis of existing runs with t-intervals, paired tests, Holm | Fix the noise-amplifying composite | Only WikiText PPL shows a detectable effect; AdaLoRA "forgets less because it learns less" |
| Resumable pipeline | Build + tests | Run long experiments on a laptop safely | MPS RNG and AdaLoRA bugs found/fixed |
| Seed-42 learning curves | Evaluate adapters during training, not only at the end | Forgetting is a trajectory | LoRA reaches ~90% of its ROUGE-L gain by step 50 but only ~37% of its final PPL increase |
| 3 seeds × 3 methods, unified protocol | 9 runs, all under one protocol | Defensible statistics | Table in section 7 |
| A1 Matched adaptation | Compare methods at equal task gain (200 SAMSum examples) | Central claim | LoRA at step 5 ≈ AdaLoRA at step 250 in both gain and PPL |
| A2 Early stopping | LoRA/DoRA stopped at steps 5–50 | Strongest simple baseline | Step 15 keeps 73% of the final gain with 12% of the PPL increase |
| A3 ΔW geometry | Merge adapter, measure ‖ΔW‖/‖W‖, stable rank, intruder dimensions | Mechanism | Relative norm correlates 0.95 with PPL, 0.97 with KL |
| A4/A5 Paired + noise floor | Per-example analysis, bootstrap, minimum detectable effect | How trustworthy are the probes | SE 0.3–1.2 pts paired vs ~3 pts unpaired; BoolQ is a yes-bias shift |
| B1 Learning-rate sweep (seed 42) | lr 1e-4, 2e-4, 5e-4 × 3 methods | Rule out lr confound | Gain-vs-forgetting trajectories overlap across lr; higher lr adds forgetting without more gain |

---

## 7. Key numbers to memorise

**Setup:** 250 steps; LoRA 4.40M, AdaLoRA 6.60M (pre-prune), DoRA 4.70M trainable params; train time 34 / 35 / 63 min per run (M4 Air).

**Final results, mean of 3 seeds (step 250):**

| Method | ROUGE-L gain | WikiText PPL increase (base 17.20) | KL to base |
|---|---|---|---|
| LoRA | +0.102 | +0.95 | 0.052 |
| DoRA | +0.100 | +1.00 | 0.055 |
| AdaLoRA | +0.050 | +0.00 | 0.001 |

**Statistics:** AdaLoRA vs LoRA/DoRA on PPL is Holm-significant (p ≈ 0.02). LoRA vs DoRA is not (Holm p ≈ 0.10).

**Findings:**
- ΔW relative norm: LoRA .0111, DoRA .0115, AdaLoRA .0026. Correlation with PPL .95, KL .97.
- LoRA step 5 (norm .0026) matches AdaLoRA step 250 (norm .0026): gain +0.055 vs +0.050, PPL +0.001 both.
- ARC/MMLU: gold log-prob drops ~0.03 nats (z 3–4) for LoRA/DoRA with no accuracy change.
- BoolQ: base says "Yes" 80% (gold is 62% Yes); after LoRA/DoRA 82–89%; balanced accuracy flat (~.55).
- LR sweep (seed 42, final gain / PPL increase): LoRA .093/.56 (1e-4), .096/1.01 (2e-4), .103/2.13 (5e-4); AdaLoRA .016/.003, .052/.011, .083/.20.
- Intruder dimensions (simplified top-10, cos<0.5): 0 everywhere. Updates are only ~1% of weight norm.
- Resumed vs clean LoRA: ~0.004 ROUGE-L and ~0.006 PPL apart (a rough measure of dropout noise).

---

## 8. Honest limitations (say these before the interviewer does)

1. **One small model, one task.** Generalisation is unproven. Planned: Qwen2.5-1.5B, another family, GSM8K.
2. **Three seeds** (t-critical 4.30). Wide intervals, low power. LoRA vs DoRA unresolved.
3. **LR sweep is single-seed**; rank/alpha sweep not done; lr 5e-5 not run.
4. **Probes are log-likelihood multiple choice.** Weakly sensitive; generation-based forgetting probes are future work.
5. **Intruder-dimension measure is simplified**, not the exact Shuttleworth et al. definition. Zero intruders does not refute their finding.
6. **100 examples at intermediate curve points** make ROUGE jagged (sd up to 0.02 early). Mitigated by extending early steps to 200.
7. **AdaLoRA's late start** (no learning until about step 50 because of the budget schedule) is a property of the schedule I chose; a different tinit would shift its curve.
8. **Mitigation baselines** (KL penalty, replay) are not yet implemented, so I cannot yet claim anything about methods that actively reduce forgetting.
9. **Single device** (MPS). Validated against CUDA for base-model numbers only.

---

## 9. Related work to name-drop (verify before citing in an interview)

- LoRA (Hu et al.), AdaLoRA (Zhang et al.), DoRA (Liu et al.).
- *LoRA Learns Less and Forgets Less* (Biderman et al., 2025): the closest prior work; they find the same
  learn-less-forget-less trade-off. Our contribution is matched-adaptation, update-norm explanation, and
  measurement-noise analysis at small scale.
- *LoRA vs Full Fine-tuning: An Illusion of Equivalence* (Shuttleworth et al.): intruder dimensions.
- KL-regularised fine-tuning and replay for mitigation; LaLoRA, CLoRA; self-generated replay (these are the
  baselines a method would have to beat).
- Continual learning classics: EWC (Kirkpatrick et al.), experience replay.

---

## 10. Likely questions and strong answers

**Q: What is the main finding?**
A: Forgetting tracks how much the weights move. AdaLoRA's retention advantage vanishes at matched task gain:
LoRA at step 5 and AdaLoRA at step 250 have the same ‖ΔW‖/‖W‖ (0.0026), the same gain (~+0.05) and the same
perplexity change (+0.001).

**Q: Why not just compare final checkpoints?**
A: A final checkpoint conflates "learned less" with "forgot less". Evaluating during training gives a gain-versus-forgetting
curve, and comparing methods at equal gain is the fair comparison.

**Q: Why is your sample so small, and does it matter?**
A: Three seeds is a limitation; I reported t-intervals with 2 degrees of freedom and Holm correction rather than
overclaiming, and flagged LoRA-vs-DoRA as unresolved. The main conclusions rest on large effects (AdaLoRA vs others, norm correlation) not on borderline p-values. Seeds 4–5 are queued.

**Q: How do you know accuracy is too coarse?**
A: With n=500, the unpaired standard error is about ±3 points, so 1-point shifts are invisible. Paired analysis shrinks it to 0.3–1.2 points, and gold-answer log-probability shows a z=3–4 drop on ARC/MMLU while accuracy is unchanged.

**Q: Explain the BoolQ result.**
A: The base model answers "Yes" 80% of the time vs 62% gold. After LoRA/DoRA 82–89%. Accuracy on yes-gold rises and on no-gold falls; balanced accuracy is flat. So "improvement" is a label-bias shift, not comprehension. Mean accuracy alone would have misled me.

**Q: Why KL(base‖tuned) and why on WikiText?**
A: It is label-free and continuous, so it detects small distribution shifts. WikiText is general text unrelated to the training task, so any shift there is unintended drift. It also correlates .97 with update norm, which supports the interpretation. Limitation: it measures one domain.

**Q: Why use `disable_adapter()` for the base distribution?**
A: Exactly the same weights and tokenizer, no second model in memory, and it works for all three adapter types.

**Q: What would you do with more compute?**
A: Replicate on 1.5B and a second family; generation-based probes; rank/alpha sweep; more seeds on the LR sweep; compare against KL-penalty and replay baselines at matched adaptation.

**Q: How would you improve on LoRA?**
A: Early stopping is already a strong baseline (step 15 keeps 73% of the gain with 12% of the PPL increase). A method must beat early stopping, KL penalty and KL+replay at matched adaptation before it earns a claim. A candidate idea is feeding KL-to-base sensitivity into AdaLoRA's importance score.

**Q: What bug are you proudest of finding?**
A: The MPS RNG resume divergence. Tests on CPU passed; I found it by replaying a resumed run against the original and seeing LoRA differ by 0.0024 while AdaLoRA/DoRA matched exactly. It taught me that equivalence tests need to run on the target device.

**Q: Why did you drop your composite score?**
A: Min-max scaling stretches tiny differences over a full range, so noise looked like signal and AdaLoRA was ranked worst on a dimension where it was indistinguishable from the base model. I replaced it with noise-aware tests and a Pareto analysis that only calls dominance when it is significant.

**Q: Why AdaLoRA 12→8 rank and others fixed 8?**
A: AdaLoRA's rank budget prunes toward a target of 8 so the final average rank matches LoRA's, but its trainable parameters before pruning are higher (6.6M). That is not perfectly equal, which is another reason I compare at matched gain and update norm, not matched parameters.

**Q: Is the paper novel given Biderman et al.?**
A: Partly overlapping on the learn-less-forget-less idea. What is new here is the matched-adaptation comparison across LoRA/AdaLoRA/DoRA, the update-norm explanation validated per snapshot, the BoolQ bias and log-prob diagnostics, and a measurement-noise analysis. I would position it as a rigorous empirical/measurement study (TMLR-style), not a new method.

**Q: How did you make results reproducible?**
A: Fixed evaluation indices, a single frozen `Settings` dataclass saved into each run's `done.json`, seeds for model, data and sampling, atomic evaluation files, crash-resume tests, and a rule that all points in one curve come from one device.

---

## 11. Status and next steps (as of 2026-11-01)

- Done: 3-seed unified results, Phase A analyses, LR sweep at 1e-4 and 5e-4 (seed 42), roadmap pushed for a teammate.
- Next: lr 5e-5; rank/alpha sweep; mitigation baselines (KL, replay); seeds 4–5; Colab benchmark; second model/task; generation probes.
- Publication plan: arXiv first, then TMLR (rolling, no deadline) or an ARR cycle for ACL Findings. Cannot dual-submit.

See `docs/ROADMAP.md` for the full plan and teammate hand-off instructions.

---

## 12. Quick glossary of the code, in case you are asked to walk through it

- `Settings` (config.py): all hyper-parameters in one frozen dataclass.
- `build_peft`: LoRA and DoRA share `LoraConfig` (DoRA via `use_dora=True`); AdaLoRA uses `AdaLoraConfig`.
- `tokenize_train`: builds "Dialogue:\n…\n\nSummary:" prompt, masks prompt tokens with -100.
- `choice_scores`: if all candidates are one token (Yes/No, A–D) a single forward pass scores all of them; otherwise score each separately.
- `eval_wiki_kl`: perplexity in 512-token blocks over the first 50k tokens, KL against the adapter-disabled model.
- `eval_samsum`: batched greedy decoding with left padding, ROUGE with stemming, per-example scores saved.
- `run_jobs`: loads the model only if some result file is missing.
- `train_run`: Trainer with callbacks (snapshot, progress, MPS cache, MPS RNG, AdaLoRA update/state), resumes from latest checkpoint.
- `train_prefix` / `backfill_early`: replay the first steps of a finished run to add early snapshots.

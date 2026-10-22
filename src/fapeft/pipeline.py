"""Run queue: for each (seed, method) train -> snapshot -> evaluate, resumable at every step.

Layout under <runs_dir>/<tag>/:
  base/evals/                  untrained-model evaluations (step 0 of every curve, PRE for the probes)
  seed{S}_{method}/ckpt        Trainer checkpoints (resume points)
  seed{S}_{method}/snapshots   adapters at every snapshot step
  seed{S}_{method}/evals       one JSON per (step, probe); re-runs skip whatever exists
  seed{S}_{method}/done.json   training complete marker
"""
import gc
import warnings
from dataclasses import fields
from pathlib import Path

import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import evals
from .config import Settings, build_peft, pick_device
from .data import INDEX_DIR, build_eval_sets, load_sources, make_collator, tokenize_train, train_subset, wiki_token_ids
from .store import cached_json
from .train import train_run

FULL_PROBES = ("boolq", "piqa", "arc", "mmlu")
_EVAL_SIZE_FIELDS = ("samsum_eval_n", "boolq_n", "piqa_n", "arc_n", "mmlu_n", "seed_for_eval_sampling")


def release(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def index_dir_for(s: Settings, root: Path) -> Path:
    """Canonical (committed) index files only for the default evaluation sizes."""
    default = Settings()
    same = all(getattr(s, f) == getattr(default, f) for f in _EVAL_SIZE_FIELDS)
    return INDEX_DIR if same else root / "indices"


class Context:
    """Everything shared across runs: tokenizer, data, evaluation subsets."""

    def __init__(self, s: Settings, root: Path, device: torch.device):
        self.s, self.root, self.device = s, root, device
        self.tok = AutoTokenizer.from_pretrained(s.base_model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.src = load_sources()
        self.index_dir = index_dir_for(s, root)
        self.evalsets = build_eval_sets(self.src, s, self.index_dir)
        self.wiki_ids = wiki_token_ids(self.src, self.tok, s)

    def load_base(self):
        m = AutoModelForCausalLM.from_pretrained(self.s.base_model, dtype=torch.float32)
        return m.to(self.device).eval()

    def load_snapshot(self, path: Path):
        with warnings.catch_warnings():
            # PEFT warns that AdaLoRA's rank_pattern keys are not LoRA-style module names; AdaLoRA
            # applies the pattern itself when loading (verified by tests/test_resume.py).
            warnings.filterwarnings("ignore", message="The following rank_pattern keys did not match")
            return PeftModel.from_pretrained(self.load_base(), str(path)).to(self.device).eval()


def _curve_jobs(ctx: Context, step: int):
    s = ctx.s
    return {
        f"curve_step_{step:04d}__samsum.json": lambda m: evals.eval_samsum(
            m, ctx.tok, ctx.device, ctx.evalsets["samsum_eval"], s.curve_samsum_n, s.samsum_batch, s.max_len, s.samsum_max_new_tokens),
        f"curve_step_{step:04d}__wiki.json": lambda m: evals.eval_wiki_kl(m, ctx.device, ctx.wiki_ids, s.wiki_block),
    }


def _full_jobs(ctx: Context, step: int):
    s = ctx.s
    fns = {"boolq": evals.eval_boolq, "piqa": evals.eval_piqa, "arc": evals.eval_arc, "mmlu": evals.eval_mmlu}
    jobs = {f"full_step_{step:04d}__{k}.json": (lambda m, f=f, k=k: f(m, ctx.tok, ctx.device, ctx.evalsets[k], s.max_len))
            for k, f in fns.items()}
    # the SAMSum final evaluation uses the full held-out set, not the curve prefix
    jobs[f"full_step_{step:04d}__samsum.json"] = lambda m: evals.eval_samsum(
        m, ctx.tok, ctx.device, ctx.evalsets["samsum_eval"], s.samsum_eval_n, s.samsum_batch, s.max_len, s.samsum_max_new_tokens)
    return jobs


def run_jobs(ctx: Context, evals_dir: Path, jobs: dict, load_model):
    """Loads the model only if some job's result is missing."""
    pending = [name for name in jobs if not (evals_dir / name).exists()]
    if not pending:
        return
    model = load_model()
    try:
        for name, fn in jobs.items():
            cached_json(evals_dir / name, lambda fn=fn: fn(model))
    finally:
        del model
        release(ctx.device)


def evaluate_base(ctx: Context):
    d = ctx.root / "base" / "evals"
    print("[base] evaluating untrained model")
    run_jobs(ctx, d, {**_curve_jobs(ctx, 0), **_full_jobs(ctx, 0)}, ctx.load_base)


def run_one(ctx: Context, seed: int, method: str, stages):
    s, name = ctx.s, f"seed{seed}_{method}"
    run_dir = ctx.root / name
    print(f"\n=== {name} ===")
    if "train" in stages:
        transformers.set_seed(seed)
        model = build_peft(AutoModelForCausalLM.from_pretrained(s.base_model, dtype=torch.float32), method, s)
        train_ds = tokenize_train(train_subset(ctx.src, s, seed, ctx.index_dir), ctx.tok, s.max_len)
        train_run(model, method, train_ds, make_collator(ctx.tok), s, run_dir, seed, use_cpu=(ctx.device.type == "cpu"))
        del model
        release(ctx.device)
    if "eval" in stages:
        for step in s.snapshot_steps:
            snap = run_dir / "snapshots" / f"step_{step:04d}"
            if not snap.exists():
                print(f"  snapshot step {step} missing (training not finished?)")
                continue
            jobs = _curve_jobs(ctx, step)
            if step == s.total_steps:
                jobs.update(_full_jobs(ctx, step))
            print(f"  [{name}] evaluating step {step}")
            run_jobs(ctx, run_dir / "evals", jobs, lambda snap=snap: ctx.load_snapshot(snap))


def run_pipeline(s: Settings, seeds, methods, tag: str, runs_dir: Path, stages=("train", "eval"), device=None):
    s.validate()
    device = device or pick_device()
    root = Path(runs_dir) / tag
    root.mkdir(parents=True, exist_ok=True)
    print(f"device={device} tag={tag} seeds={list(seeds)} methods={list(methods)} total_steps={s.total_steps}")
    ctx = Context(s, root, device)
    if "eval" in stages:
        evaluate_base(ctx)
    for seed in seeds:
        for method in methods:
            run_one(ctx, seed, method, stages)
    print("\nAll queued work complete.")


def parse_overrides(pairs):
    types = {f.name: f.type for f in fields(Settings)}
    out = {}
    for pair in pairs or []:
        k, v = pair.split("=", 1)
        if k not in types or types[k] not in (int, float, str):
            raise ValueError(f"cannot override {k!r}")
        out[k] = types[k](v)
    return out


SMOKE = dict(train_n=64, epochs=1, grad_accum=2, save_every=4, snapshot_every=8, adalora_tinit=4,
             adalora_tfinal=6, adalora_deltaT=2, samsum_eval_n=16, curve_samsum_n=8, samsum_batch=8,
             samsum_max_new_tokens=32, boolq_n=20, piqa_n=20, arc_n=20, mmlu_n=20, wiki_tokens=2048)

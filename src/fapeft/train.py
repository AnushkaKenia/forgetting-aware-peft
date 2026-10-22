"""Resumable PEFT training.

Save points:
  * runs/.../ckpt/checkpoint-N     full Trainer state every `save_every` steps (resume point)
  * runs/.../snapshots/step_NNNN   adapter kept forever every `snapshot_every` steps
  * runs/.../done.json             written only after the last step; marks the run complete
Killing the process at any moment loses at most `save_every` optimizer steps.

AdaLoRA keeps its pruning state (importance averages, rank pattern) outside the normal
Trainer checkpoint, so AdaLoraStateCallback persists and restores it. Without that, a
resume would silently restart the importance estimates, and a resume after the final
pruning step would crash on a missing rank pattern.
"""
import os
import shutil
import time
from pathlib import Path

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

from .config import Settings
from .store import read_json, write_json


class SimulatedCrash(RuntimeError):
    """Raised by tests to emulate a runtime disconnect right after a checkpoint."""


def latest_checkpoint(ckpt_dir: Path):
    found = []
    for p in Path(ckpt_dir).glob("checkpoint-*"):
        try:
            found.append((int(p.name.split("-")[-1]), p))
        except ValueError:
            pass
    return max(found)[1] if found else None


class AdaLoraUpdateCallback(TrainerCallback):
    """PEFT requires update_and_allocate() after optimizer.step() and before zero_grad()."""

    def __init__(self, peft_model):
        self.model = peft_model
        self.calls = []

    def on_optimizer_step(self, args, state, control, **kwargs):
        self.calls.append(state.global_step)
        self.model.base_model.update_and_allocate(state.global_step)


class AdaLoraStateCallback(TrainerCallback):
    FILE = "adalora_allocator.pt"

    def __init__(self, peft_model, ckpt_dir: Path):
        self.model = peft_model
        self.ckpt_dir = Path(ckpt_dir)

    def _alloc(self):
        return self.model.base_model.rankallocator

    def on_save(self, args, state, control, **kwargs):
        a = self._alloc()
        cfg = self.model.base_model.peft_config[self.model.base_model.trainable_adapter_name]
        payload = {"exp_avg_ipt": a.exp_avg_ipt, "exp_avg_unc": a.exp_avg_unc, "rank_pattern": cfg.rank_pattern}
        folder = self.ckpt_dir / f"checkpoint-{state.global_step}"
        tmp = folder / (self.FILE + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, folder / self.FILE)

    def on_train_begin(self, args, state, control, **kwargs):
        if state.global_step == 0:
            return
        path = self.ckpt_dir / f"checkpoint-{state.global_step}" / self.FILE
        if not path.exists():
            raise FileNotFoundError(f"resuming AdaLoRA at step {state.global_step} but {path} is missing")
        payload = torch.load(path, map_location=next(self.model.parameters()).device)
        a = self._alloc()
        a.exp_avg_ipt, a.exp_avg_unc = payload["exp_avg_ipt"], payload["exp_avg_unc"]
        # `ipt` is overwritten from fresh gradients before use, but update_ipt() re-zeroes every
        # moving average whose key is missing from it, so it must contain the restored keys.
        a.ipt = {n: torch.zeros_like(t) for n, t in a.exp_avg_ipt.items()}
        cfg = self.model.base_model.peft_config[self.model.base_model.trainable_adapter_name]
        cfg.rank_pattern = payload["rank_pattern"]


class SnapshotCallback(TrainerCallback):
    """Keeps an adapter at every snapshot step. Runs before the Trainer saves its checkpoint
    for the same step, so a crash can never leave a checkpoint whose snapshot is missing."""

    def __init__(self, peft_model, snap_dir: Path, steps):
        self.model, self.snap_dir, self.steps = peft_model, Path(snap_dir), set(steps)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step not in self.steps:
            return
        final = self.snap_dir / f"step_{state.global_step:04d}"
        if final.exists():
            return
        tmp = self.snap_dir / f".tmp_step_{state.global_step:04d}"
        shutil.rmtree(tmp, ignore_errors=True)
        self.model.save_pretrained(tmp)
        os.replace(tmp, final)


class ProgressCallback(TrainerCallback):
    """Tracks cumulative training seconds across resumes and writes the loss log."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.prior = (read_json(self.run_dir / "progress.json") or {}).get("train_seconds", 0.0)
        self.t0 = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()

    def _seconds(self):
        return self.prior + (time.time() - self.t0)

    def on_save(self, args, state, control, **kwargs):
        write_json(self.run_dir / "progress.json", {"step": state.global_step, "train_seconds": self._seconds()})
        write_json(self.run_dir / "loss_log.json", state.log_history)

    def on_train_end(self, args, state, control, **kwargs):
        write_json(self.run_dir / "loss_log.json", state.log_history)


class MpsCacheCallback(TrainerCallback):
    """Apple's MPS allocator caches blocks per tensor shape. With variable-length batches the
    cache grows past physical RAM (observed: ~17 GB wired, 10 GB swap, steps slowing 5 s -> 80 s),
    so release unused blocks after every micro-batch."""

    def on_substep_end(self, args, state, control, **kwargs):
        if torch.backends.mps.is_available() and not args.use_cpu:
            torch.mps.empty_cache()

    on_step_end = on_substep_end


class CrashAfterCheckpoint(TrainerCallback):
    def __init__(self, step: int):
        self.step = step

    def on_save(self, args, state, control, **kwargs):
        if state.global_step == self.step:
            raise SimulatedCrash(f"simulated disconnect after checkpoint {self.step}")


class FullRankCheckpointTrainer(Trainer):
    """PEFT prunes an AdaLoRA adapter to its rank pattern whenever it is saved, but the live
    model keeps full-size, masked tensors (and so does the optimizer state). A pruned
    checkpoint therefore cannot be loaded back into a freshly built model to resume. Saving
    the Trainer checkpoint with rank_pattern cleared keeps it full-size; the pattern itself
    is persisted by AdaLoraStateCallback. Snapshots (SnapshotCallback) are still saved pruned."""

    def _save(self, output_dir=None, state_dict=None):
        base = getattr(self.model, "base_model", None)
        cfg = base.peft_config[base.trainable_adapter_name] if hasattr(base, "rankallocator") else None
        kept = getattr(cfg, "rank_pattern", None)
        if cfg is not None:
            cfg.rank_pattern = None
        try:
            super()._save(output_dir, state_dict)
        finally:
            if cfg is not None:
                cfg.rank_pattern = kept


def train_run(peft_model, method: str, train_ds, collator, s: Settings, run_dir: Path, seed: int,
              use_cpu: bool = False, extra_callbacks=()):
    """Trains (or resumes) one (seed, method) run. Returns the run's done.json record."""
    run_dir = Path(run_dir)
    done_path = run_dir / "done.json"
    if read_json(done_path) is not None:
        print(f"  run already complete: {run_dir.name}")
        return read_json(done_path)

    ckpt_dir, snap_dir = run_dir / "ckpt", run_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    s.validate()

    args = TrainingArguments(
        output_dir=str(ckpt_dir), num_train_epochs=s.epochs,
        per_device_train_batch_size=s.batch, gradient_accumulation_steps=s.grad_accum,
        learning_rate=s.lr, logging_steps=10, save_strategy="steps", save_steps=s.save_every,
        save_total_limit=s.keep_checkpoints, fp16=torch.cuda.is_available() and not use_cpu,
        report_to="none", seed=seed, data_seed=seed, remove_unused_columns=False,
        dataloader_num_workers=0, use_cpu=use_cpu,
    )
    callbacks = [SnapshotCallback(peft_model, snap_dir, s.snapshot_steps), ProgressCallback(run_dir), MpsCacheCallback()]
    if method == "adalora":
        callbacks += [AdaLoraUpdateCallback(peft_model), AdaLoraStateCallback(peft_model, ckpt_dir)]
    callbacks += list(extra_callbacks)

    trainer = FullRankCheckpointTrainer(model=peft_model, args=args, train_dataset=train_ds, data_collator=collator, callbacks=callbacks)
    resume = latest_checkpoint(ckpt_dir)
    print(f"  training {run_dir.name}: {s.total_steps} steps, resume from {resume.name if resume else 'scratch'}")
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    assert trainer.state.global_step == s.total_steps, "training ended early"

    n_train = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in peft_model.parameters())
    record = {"method": method, "seed": seed, "steps": trainer.state.global_step,
              "trainable_params": n_train, "total_params": n_total,
              "train_seconds": (read_json(run_dir / "progress.json") or {}).get("train_seconds"),
              "settings": {k: v for k, v in s.__dict__.items()}}
    if method == "adalora":
        cfg = peft_model.base_model.peft_config[peft_model.base_model.trainable_adapter_name]
        record["adalora_rank_pattern"] = cfg.rank_pattern
    write_json(done_path, record)
    return record

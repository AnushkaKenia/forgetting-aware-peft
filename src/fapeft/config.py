"""Experiment settings and PEFT configurations.

Training hyper-parameters match Phases 2-4 (2,000 SAMSum examples, 2 epochs, effective
batch 16, lr 2e-4, rank 8 / alpha 16 / dropout .05, same seven target modules).
"""
import math
from dataclasses import dataclass, field, replace
from typing import List

import torch
from peft import AdaLoraConfig, LoraConfig, get_peft_model

METHODS = ("lora", "adalora", "dora")


@dataclass(frozen=True)
class Settings:
    base_model: str = "Qwen/Qwen2.5-0.5B"
    # training
    train_n: int = 2000
    epochs: int = 2
    batch: int = 2
    grad_accum: int = 8
    lr: float = 2e-4
    max_len: int = 512
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    adalora_init_r: int = 12
    adalora_target_r: int = 8
    adalora_tinit: int = 50
    adalora_tfinal: int = 50
    adalora_deltaT: int = 10
    # save points
    save_every: int = 25       # full Trainer checkpoint (resume point)
    snapshot_every: int = 50   # adapter kept forever, evaluated for the learning curve
    keep_checkpoints: int = 2
    # evaluation
    seed_for_eval_sampling: int = 42
    samsum_eval_n: int = 200       # held-out set used for the final (step-250) evaluation
    curve_samsum_n: int = 100      # prefix of the above used at every curve point
    samsum_batch: int = 16
    samsum_max_new_tokens: int = 96
    boolq_n: int = 500
    piqa_n: int = 500
    arc_n: int = 500
    mmlu_n: int = 500
    wiki_tokens: int = 50_000
    wiki_block: int = 512

    @property
    def total_steps(self) -> int:
        return math.ceil(self.train_n / (self.batch * self.grad_accum)) * self.epochs

    @property
    def snapshot_steps(self) -> List[int]:
        return list(range(self.snapshot_every, self.total_steps + 1, self.snapshot_every))

    def validate(self) -> None:
        assert self.snapshot_every % self.save_every == 0, "snapshots must coincide with resume checkpoints"
        assert self.curve_samsum_n <= self.samsum_eval_n
        assert self.adalora_tinit < self.total_steps - self.adalora_tfinal, "AdaLoRA budget schedule invalid"

    def with_overrides(self, **kw) -> "Settings":
        return replace(self, **kw)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_peft(base_model, method: str, s: Settings):
    if method in ("lora", "dora"):
        cfg = LoraConfig(r=s.lora_r, lora_alpha=s.lora_alpha, lora_dropout=s.lora_dropout,
                         target_modules=s.target_modules, bias="none", task_type="CAUSAL_LM",
                         use_dora=(method == "dora"))
    elif method == "adalora":
        cfg = AdaLoraConfig(init_r=s.adalora_init_r, target_r=s.adalora_target_r,
                            lora_alpha=s.lora_alpha, lora_dropout=s.lora_dropout,
                            target_modules=s.target_modules, bias="none", task_type="CAUSAL_LM",
                            tinit=s.adalora_tinit, tfinal=s.adalora_tfinal,
                            deltaT=s.adalora_deltaT, total_step=s.total_steps)
    else:
        raise ValueError(f"unknown method {method!r}")
    return get_peft_model(base_model, cfg)

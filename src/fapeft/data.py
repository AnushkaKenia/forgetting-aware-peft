"""Datasets, fixed evaluation subsets, and SAMSum tokenisation.

Roles (never violated): SAMSum is the only fine-tuning set. BoolQ, PIQA, ARC-Easy,
MMLU and WikiText-2 are evaluation-only. Evaluation subsets are sampled once with a
fixed seed and persisted under data/indices/ so every model state sees identical
examples; training samples are drawn per run seed.
"""
import random
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq

from .config import Settings
from .store import read_json, write_json

INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "indices"


def load_sources():
    return {
        "samsum": load_dataset("knkarthick/samsum"),
        "boolq": load_dataset("google/boolq"),
        "piqa": load_dataset("lighteval/piqa"),
        "arc": load_dataset("allenai/ai2_arc", "ARC-Easy"),
        "mmlu": load_dataset("cais/mmlu", "all"),
        "wiki": load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1"),
    }


def fixed_indices(name: str, n_total: int, k: int, seed: int, index_dir: Path = INDEX_DIR):
    path = index_dir / f"{name}.json"
    saved = read_json(path)
    if saved is not None:
        assert len(saved) == min(k, n_total) and max(saved) < n_total, f"stale index file {path}"
        return saved
    idx = sorted(random.Random(seed).sample(range(n_total), min(k, n_total)))
    write_json(path, idx)
    return idx


def balanced_mmlu_indices(ds, k: int, seed: int, index_dir: Path = INDEX_DIR):
    path = index_dir / "mmlu.json"
    saved = read_json(path)
    if saved is not None:
        return saved
    by_subject = {}
    for i, subject in enumerate(ds["subject"]):
        by_subject.setdefault(subject, []).append(i)
    rng = random.Random(seed)
    for subject in by_subject:
        rng.shuffle(by_subject[subject])
    chosen, pos = [], {s: 0 for s in by_subject}
    while len(chosen) < min(k, len(ds)):
        added = False
        for s in sorted(by_subject):
            if pos[s] < len(by_subject[s]) and len(chosen) < k:
                chosen.append(by_subject[s][pos[s]])
                pos[s] += 1
                added = True
        if not added:
            break
    chosen = sorted(chosen)
    write_json(path, chosen)
    return chosen


def build_eval_sets(src, s: Settings, index_dir: Path = INDEX_DIR):
    seed = s.seed_for_eval_sampling
    return {
        "samsum_eval": src["samsum"]["test"].select(
            fixed_indices("samsum_eval", len(src["samsum"]["test"]), s.samsum_eval_n, seed, index_dir)),
        "boolq": src["boolq"]["validation"].select(
            fixed_indices("boolq", len(src["boolq"]["validation"]), s.boolq_n, seed, index_dir)),
        "piqa": src["piqa"]["validation"].select(
            fixed_indices("piqa", len(src["piqa"]["validation"]), s.piqa_n, seed, index_dir)),
        "arc": src["arc"]["validation"].select(
            fixed_indices("arc", len(src["arc"]["validation"]), s.arc_n, seed, index_dir)),
        "mmlu": src["mmlu"]["validation"].select(balanced_mmlu_indices(src["mmlu"]["validation"], s.mmlu_n, seed, index_dir)),
    }


def wiki_token_ids(src, tok, s: Settings) -> torch.Tensor:
    text = "\n\n".join(x for x in src["wiki"]["test"]["text"] if x.strip())
    return torch.tensor(tok(text, add_special_tokens=False).input_ids[: s.wiki_tokens], dtype=torch.long)


def train_subset(src, s: Settings, seed: int, index_dir: Path = INDEX_DIR):
    train = src["samsum"]["train"]
    return train.select(fixed_indices(f"samsum_train_n{s.train_n}_seed{seed}", len(train), s.train_n, seed, index_dir))


def tokenize_train(ds, tok, max_len: int):
    """Target-only loss: prompt and padding positions are -100, only summary tokens count."""
    def encode(ex):
        prompt = tok(f"Dialogue:\n{ex['dialogue']}\n\nSummary:", add_special_tokens=False).input_ids
        target = tok(" " + ex["summary"] + tok.eos_token, add_special_tokens=False).input_ids
        prompt = prompt[: max(1, max_len - len(target))]
        ids = (prompt + target)[:max_len]
        labels = ([-100] * len(prompt) + target)[: len(ids)]
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}
    return ds.map(encode, remove_columns=ds.column_names)


def make_collator(tok):
    return DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100, return_tensors="pt")

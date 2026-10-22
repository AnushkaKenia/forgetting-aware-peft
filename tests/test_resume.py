"""Crash/resume equivalence on a tiny random Qwen2 model (CPU, seconds per test).

An interrupted-then-resumed run must produce the same final adapter (and, for AdaLoRA,
the same rank pattern) as an uninterrupted run. This is the guarantee the Colab/laptop
save-point design rests on.
"""
import pytest
import torch
from datasets import Dataset
from transformers import Qwen2Config, Qwen2ForCausalLM, default_data_collator
from peft import PeftModel

from fapeft.config import Settings, build_peft
from fapeft.store import read_json
from fapeft.train import CrashAfterCheckpoint, SimulatedCrash, train_run

SEQ = 12


def tiny_settings(**kw):
    base = dict(train_n=32, epochs=2, batch=2, grad_accum=2, lr=1e-3, lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                adalora_init_r=6, adalora_target_r=4, adalora_tinit=4, adalora_tfinal=8, adalora_deltaT=2,
                save_every=4, snapshot_every=8, curve_samsum_n=1, samsum_eval_n=1)
    base.update(kw)
    return Settings(**base)  # 32/(2*2)*2 = 16 optimizer steps


def make_data(n=32, vocab=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (n, SEQ), generator=g)
    return Dataset.from_dict({"input_ids": ids.tolist(), "attention_mask": [[1] * SEQ] * n, "labels": ids.tolist()})


def build(method, s, seed=7):
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    return build_peft(Qwen2ForCausalLM(cfg), method, s)


def adapter_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k}


def run(method, run_dir, s, crash_at=None):
    model = build(method, s)
    extra = [CrashAfterCheckpoint(crash_at)] if crash_at else []
    return model, lambda: train_run(model, method, make_data(), default_data_collator, s, run_dir, seed=1,
                                    use_cpu=True, extra_callbacks=extra)


@pytest.mark.parametrize("method", ["lora", "dora", "adalora"])
@pytest.mark.parametrize("crash_at", [4, 8, 12])  # 8: just before AdaLoRA's final prune, 12: after it
def test_resumed_run_matches_uninterrupted(tmp_path, method, crash_at):
    s = tiny_settings()
    ref_model, go = run(method, tmp_path / "ref", s)
    ref_record = go()

    crashed_model, go = run(method, tmp_path / "crash", s, crash_at=crash_at)
    with pytest.raises(SimulatedCrash):
        go()
    assert read_json(tmp_path / "crash" / "done.json") is None, "crashed run must not be marked done"

    resumed_model, go = run(method, tmp_path / "crash", s)  # fresh process, same run dir
    resumed_record = go()

    a, b = adapter_state(ref_model), adapter_state(resumed_model)
    assert a.keys() == b.keys()
    worst = max((a[k] - b[k]).abs().max().item() for k in a)
    assert worst < 1e-5, f"resumed weights diverge from uninterrupted run (max abs diff {worst})"
    if method == "adalora":
        assert resumed_record["adalora_rank_pattern"] == ref_record["adalora_rank_pattern"]
        assert ref_record["adalora_rank_pattern"], "AdaLoRA never pruned: test is not exercising the mechanism"


def test_snapshots_exist_and_reload(tmp_path):
    s = tiny_settings()
    for method in ("lora", "adalora"):
        model, go = run(method, tmp_path / method, s)
        go()
        snaps = sorted(p.name for p in (tmp_path / method / "snapshots").iterdir())
        assert snaps == ["step_0008", "step_0016"], snaps
        cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
        torch.manual_seed(7)
        reloaded = PeftModel.from_pretrained(Qwen2ForCausalLM(cfg), str(tmp_path / method / "snapshots" / "step_0016"))
        x = torch.randint(0, 64, (1, SEQ))
        model.eval(); reloaded.eval()
        torch.manual_seed(7)
        assert torch.allclose(model(x).logits, reloaded.to("cpu")(x).logits, atol=1e-4)


def test_completed_run_is_skipped(tmp_path):
    s = tiny_settings()
    model, go = run("lora", tmp_path / "r", s)
    first = go()
    model2, go2 = run("lora", tmp_path / "r", s)
    assert go2() == first

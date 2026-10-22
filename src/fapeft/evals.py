"""Evaluation probes. One implementation, used identically for the base model and every
adapter snapshot, so measurements cannot drift apart between model states.

Accuracy probes score candidates by mean continuation log-likelihood (no generation
parsing). Per-example outcomes are returned so paired significance tests are possible.
Continuous measures (gold log-prob, KL to the base model) detect drift that accuracy on
500 examples is too coarse to see.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from rouge_score import rouge_scorer


@torch.inference_mode()
def _score(model, tok, device, prompt, cont, max_len):
    p = tok(prompt, add_special_tokens=False).input_ids
    c = tok(cont, add_special_tokens=False).input_ids
    p = p[-max(1, max_len - len(c)):]
    x = torch.tensor([p + c], device=device)
    logits = model(input_ids=x).logits[:, len(p) - 1: len(p) - 1 + len(c), :].float()
    lp = F.log_softmax(logits, dim=-1)
    y = torch.tensor([c], device=device)
    return lp.gather(-1, y.unsqueeze(-1)).squeeze(-1).mean().item()


@torch.inference_mode()
def choice_scores(model, tok, device, prompt, conts, max_len):
    """Mean continuation log-prob of every candidate. When all candidates are one token
    (Yes/No, A-D) one forward pass gives exactly the same numbers as scoring each alone."""
    cids = [tok(c, add_special_tokens=False).input_ids for c in conts]
    if all(len(c) == 1 for c in cids):
        p = tok(prompt, add_special_tokens=False).input_ids[-max(1, max_len - 1):]
        logits = model(input_ids=torch.tensor([p], device=device)).logits[0, -1].float()
        lp = F.log_softmax(logits, dim=-1)
        return [lp[c[0]].item() for c in cids]
    return [_score(model, tok, device, prompt, c, max_len) for c in conts]


def _summarise(correct, gold_lp):
    return {"accuracy": float(np.mean(correct)), "n": len(correct),
            "gold_logprob_mean": float(np.mean(gold_lp)),
            "correct": [int(c) for c in correct], "gold_logprob": [float(g) for g in gold_lp]}


@torch.inference_mode()
def eval_boolq(model, tok, device, ds, max_len):
    correct, gold = [], []
    for x in ds:
        prompt = f"Passage: {x['passage']}\nQuestion: {x['question']}\nAnswer:"
        yes, no = choice_scores(model, tok, device, prompt, [" Yes", " No"], max_len)
        correct.append((yes > no) == bool(x["answer"]))
        gold.append(yes if x["answer"] else no)
    return _summarise(correct, gold)


@torch.inference_mode()
def eval_piqa(model, tok, device, ds, max_len):
    correct, gold = [], []
    for x in ds:
        prompt = f"Goal: {x['goal']}\nSolution:"
        s1, s2 = choice_scores(model, tok, device, prompt, [" " + x["sol1"], " " + x["sol2"]], max_len)
        pred = int(s2 > s1)
        correct.append(pred == int(x["label"]))
        gold.append((s1, s2)[int(x["label"])])
    return _summarise(correct, gold)


@torch.inference_mode()
def eval_arc(model, tok, device, ds, max_len):
    correct, gold = [], []
    for x in ds:
        labs, texts = list(x["choices"]["label"]), list(x["choices"]["text"])
        prompt = ("Question: " + x["question"] + "\nChoices:\n"
                  + "\n".join(f"{a}. {b}" for a, b in zip(labs, texts)) + "\nAnswer:")
        scores = choice_scores(model, tok, device, prompt, [" " + a for a in labs], max_len)
        gi = labs.index(str(x["answerKey"]))
        correct.append(int(np.argmax(scores)) == gi)
        gold.append(scores[gi])
    return _summarise(correct, gold)


@torch.inference_mode()
def eval_mmlu(model, tok, device, ds, max_len):
    labs = ["A", "B", "C", "D"]
    correct, gold = [], []
    for x in ds:
        prompt = (f"Subject: {x['subject'].replace('_', ' ')}\nQuestion: {x['question']}\nChoices:\n"
                  + "\n".join(f"{labs[i]}. {c}" for i, c in enumerate(x["choices"])) + "\nAnswer:")
        scores = choice_scores(model, tok, device, prompt, [" " + a for a in labs], max_len)
        correct.append(int(np.argmax(scores)) == int(x["answer"]))
        gold.append(scores[int(x["answer"])])
    return _summarise(correct, gold)


@torch.inference_mode()
def eval_wiki_kl(model, device, ids, block):
    """WikiText-2 perplexity on a fixed token prefix, plus mean per-token KL(base || tuned).
    The base distribution comes from the same model with the adapter switched off, so no
    second copy of the model is needed. KL is omitted for a plain (non-PEFT) model."""
    nll, kl, n = 0.0, 0.0, 0
    has_adapter = hasattr(model, "disable_adapter")
    for st in range(0, len(ids) - 1, block):
        chunk = ids[st: st + block]
        if len(chunk) < 2:
            continue
        x = chunk.unsqueeze(0).to(device)
        lp = F.log_softmax(model(x).logits[0, :-1].float(), dim=-1)
        y = x[0, 1:]
        nll -= lp.gather(-1, y[:, None]).sum().item()
        n += y.numel()
        if has_adapter:
            with model.disable_adapter():
                blp = F.log_softmax(model(x).logits[0, :-1].float(), dim=-1)
            kl += (blp.exp() * (blp - lp)).sum().item()
    out = {"ppl": math.exp(nll / n), "tokens": n}
    if has_adapter:
        out["kl_base_to_tuned"] = kl / n
    return out


@torch.inference_mode()
def eval_samsum(model, tok, device, ds, n, batch, max_len, max_new_tokens):
    """Greedy generation, batched with left padding; ROUGE with stemming, mean of per-example F1."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    old_side, tok.padding_side = tok.padding_side, "left"
    preds, refs = [], []
    try:
        for i in range(0, n, batch):
            rows = [ds[j] for j in range(i, min(n, i + batch))]
            prompts = [f"Dialogue:\n{r['dialogue']}\n\nSummary:" for r in rows]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            for o, r in zip(out, rows):
                preds.append(tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip())
                refs.append(r["summary"])
    finally:
        tok.padding_side = old_side
    per = [scorer.score(r, p) for r, p in zip(refs, preds)]
    res = {k: float(np.mean([s[k].fmeasure for s in per])) for k in ("rouge1", "rouge2", "rougeL")}
    res["n"] = len(preds)
    res["rougeL_per_example"] = [float(s["rougeL"].fmeasure) for s in per]
    res["sample_predictions"] = preds[:3]
    return res

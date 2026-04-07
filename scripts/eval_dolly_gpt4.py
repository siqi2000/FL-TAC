"""GPT-4 scoring for the Dolly scenario (paper-style evaluation).

This is a *starter* script. It mirrors the protocol used by FedIT
(Zhang et al., 2023): for each held-out Dolly example we let the trained
FL-TAC adapter generate an answer, then ask GPT-4 to grade the answer
against the reference response on a 0–1 scale.

Usage:
    export OPENAI_API_KEY=sk-...
    python scripts/eval_dolly_gpt4.py \
        --adapters_dir results/dolly_llama_fltac \
        --model_name huggyllama/llama-7b \
        --round 10 \
        --max_per_task 50 \
        --out results/dolly_llama_fltac/gpt4_scores.json

You will need:  pip install openai
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import Dict, List

import torch

from fltac.data import load_dolly, DOLLY_TASKS
from fltac.models import build_base_model, attach_lora, set_adapter_state


JUDGE_SYSTEM = (
    "You are a strict but fair grader. Compare the assistant's answer to the "
    "reference answer and rate it on a scale from 0 (completely wrong / "
    "irrelevant) to 1 (matches the reference in content and quality). "
    "Reply with a single floating-point number between 0 and 1."
)

JUDGE_USER_TEMPLATE = (
    "Question: {question}\n\n"
    "Reference answer: {reference}\n\n"
    "Assistant answer: {prediction}\n\n"
    "Score (0 to 1):"
)


def grade_with_gpt4(question: str, reference: str, prediction: str, model: str = "gpt-4") -> float:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",
             "content": JUDGE_USER_TEMPLATE.format(
                 question=question, reference=reference, prediction=prediction)},
        ],
        temperature=0.0,
        max_tokens=8,
    )
    text = resp.choices[0].message.content.strip()
    try:
        return max(0.0, min(1.0, float(text.split()[0])))
    except Exception:
        return 0.0


@torch.no_grad()
def generate(peft_model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(peft_model.device)
    out = peft_model.generate(**inputs, max_new_tokens=max_new_tokens,
                              do_sample=False, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters_dir", required=True,
                    help="dir containing adapters_round_<t>.pt")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--model_name", default="huggyllama/llama-7b")
    ap.add_argument("--max_per_task", type=int, default=50)
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base, tok, arch = build_base_model("dolly", args.model_name)
    peft_model = attach_lora(base, arch, rank=args.lora_rank)
    _, eval_ds = load_dolly(tok, max_len=512)

    snap = torch.load(Path(args.adapters_dir) / f"adapters_round_{args.round}.pt",
                      map_location="cpu", weights_only=False)
    cluster_states: Dict[int, dict] = snap["cluster_states"]
    labels: List[int] = snap["labels"]
    true_tasks: List[str] = snap["true_tasks"]

    # majority-vote cluster -> task assignment
    from collections import Counter
    cluster_to_task = {}
    for c in set(labels):
        votes = Counter(t for l, t in zip(labels, true_tasks) if l == c)
        cluster_to_task[c] = votes.most_common(1)[0][0]
    task_to_state = {cluster_to_task[c]: s for c, s in cluster_states.items()}

    scores: Dict[str, List[float]] = {t: [] for t in DOLLY_TASKS}
    for task in DOLLY_TASKS:
        if task not in task_to_state:
            print(f"[skip] no aggregated adapter for {task}")
            continue
        set_adapter_state(peft_model, task_to_state[task])
        ds = eval_ds[task].select(range(min(args.max_per_task, len(eval_ds[task]))))
        for i, ex in enumerate(ds):
            input_ids = ex["input_ids"]
            labels_t = ex["labels"]
            # split prompt vs answer using the -100 mask
            prompt_len = sum(1 for l in labels_t if l == -100)
            prompt_text = tok.decode(input_ids[:prompt_len], skip_special_tokens=True)
            ref_text = tok.decode([t for t in labels_t if t != -100], skip_special_tokens=True)
            pred = generate(peft_model, tok, prompt_text)
            score = grade_with_gpt4(prompt_text, ref_text, pred)
            scores[task].append(score)
            print(f"  [{task} {i+1}/{len(ds)}] {score:.2f}")
        print(f"== {task} avg = {sum(scores[task])/max(len(scores[task]),1):.3f} ==")

    summary = {t: (sum(s)/len(s) if s else None) for t, s in scores.items()}
    Path(args.out).write_text(json.dumps({"per_example": scores, "average": summary},
                                          indent=2))
    print(f"\nWrote {args.out}")
    for t, v in summary.items():
        print(f"  {t:25s} {v}")


if __name__ == "__main__":
    main()

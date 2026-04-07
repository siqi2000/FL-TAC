"""Model + LoRA wrapping for FL-TAC.

We use HuggingFace `peft` to attach LoRA adapters. The trick: each (client,
task) pair has its OWN adapter, but we don't actually need to live with N*K
adapters in memory simultaneously — at any given time, only one client trains
at a time, and that client only holds its own adapters as state-dicts.

Therefore each FL client keeps a python dict {task -> adapter_state_dict} of
LoRA params, and uses `set_peft_model_state_dict` to swap them in and out of
a single base+LoRA model when training that task.
"""
from __future__ import annotations
from typing import List

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from peft.utils.save_and_load import load_peft_weights  # noqa


# ---------------------------------------------------------------------------
# Target module presets per architecture
# ---------------------------------------------------------------------------
LORA_TARGETS = {
    "bert": ["query", "value"],
    "vit":  ["query", "value"],
    "llama": ["q_proj", "v_proj"],
}


def build_base_model(scenario: str, model_name: str, num_labels_map=None):
    """Return (model, tokenizer_or_processor, arch_key)."""
    if scenario == "glue":
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        # one head per task is impractical for shared backbone; we use binary
        # head and treat all GLUE tasks as 2-class (they all are in our subset).
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2)
        return model, tok, "bert"

    if scenario == "cifar":
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        proc = AutoImageProcessor.from_pretrained(model_name)
        # Use 100 classes (CIFAR-100); CIFAR-10 labels < 10 still valid.
        model = AutoModelForImageClassification.from_pretrained(
            model_name, num_labels=100, ignore_mismatched_sizes=True)
        return model, proc, "vit"

    if scenario == "dolly":
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None)
        return model, tok, "llama"

    raise ValueError(f"unknown scenario {scenario}")


def attach_lora(model, arch: str, rank: int, alpha: int = 16,
                dropout: float = 0.05):
    """Wrap model with one initial LoRA adapter named 'default'."""
    task_type = "CAUSAL_LM" if arch == "llama" else (
        "SEQ_CLS" if arch == "bert" else "FEATURE_EXTRACTION")
    cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=LORA_TARGETS[arch],
        bias="none", task_type=task_type,
    )
    return get_peft_model(model, cfg)


# ---------------------------------------------------------------------------
# State-dict helpers (the FL "adapters" we exchange)
# ---------------------------------------------------------------------------
def get_adapter_state(peft_model) -> dict:
    """Return CPU copy of LoRA params only (the v_{i,j} of the paper).

    We deliberately exclude classifier heads / `modules_to_save` so that
    only the low-rank adapters are transmitted between server and clients.
    """
    sd = get_peft_model_state_dict(peft_model)
    return {k: v.detach().cpu().clone() for k, v in sd.items() if "lora_" in k}


def set_adapter_state(peft_model, state: dict):
    """Load LoRA params into the model in-place, ignoring keys not in `state`.

    We bypass `set_peft_model_state_dict` because some peft versions error out
    when the state dict is missing classifier-head keys. We only need to copy
    the LoRA tensors into the corresponding model parameters.
    """
    name_to_param = dict(peft_model.named_parameters())
    with torch.no_grad():
        for k, v in state.items():
            # peft state-dict keys start with "base_model." which matches
            # the model's named_parameters() exactly.
            if k in name_to_param:
                name_to_param[k].data.copy_(v.to(name_to_param[k].device))


def zero_like_state(state: dict) -> dict:
    return {k: torch.zeros_like(v) for k, v in state.items()}


def average_states(states: List[dict], weights: List[float] | None = None) -> dict:
    if weights is None:
        weights = [1.0 / len(states)] * len(states)
    out = {k: torch.zeros_like(v) for k, v in states[0].items()}
    for s, w in zip(states, weights):
        for k in out:
            out[k] = out[k] + s[k] * w
    return out


def flatten_state(state: dict) -> torch.Tensor:
    return torch.cat([v.flatten().float() for _, v in sorted(state.items())])

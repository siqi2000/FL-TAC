"""Dataset loading + Dirichlet partition for FL-TAC.

Three scenarios:
  - GLUE  (5 tasks: sst2, mrpc, qqp, qnli, rte)             -> BERT
  - CIFAR (2 tasks: cifar10, cifar100)                       -> ViT
  - Dolly (8 tasks from databricks-dolly-15k category field) -> LLaMA

For each scenario we return:
    task_datasets : dict[task_name -> HF Dataset (train)]
    task_eval     : dict[task_name -> HF Dataset (eval)]
    client_indices: dict[task_name -> list[list[int]]]   # per-client sample idx

Dirichlet partition follows Hsu et al. 2020. We drop clients whose share is
below `min_frac` (default 0.01) for that task, matching the paper.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple

GLUE_TASKS = ["sst2", "mrpc", "qqp", "qnli", "rte"]
CIFAR_TASKS = ["cifar10", "cifar100"]
DOLLY_TASKS = [
    "brainstorming", "classification", "closed_qa", "creative_writing",
    "general_qa", "information_extraction", "open_qa", "summarization",
]


# ---------------------------------------------------------------------------
# Dirichlet partition
# ---------------------------------------------------------------------------
def dirichlet_partition(
    n_samples: int,
    n_clients: int,
    alpha: float = 0.5,
    min_frac: float = 0.01,
    seed: int = 0,
) -> List[List[int]]:
    """Split [0, n_samples) into n_clients lists using Dirichlet(alpha).

    Clients whose proportion < min_frac get no data for this task.
    """
    rng = np.random.default_rng(seed)
    proportions = rng.dirichlet([alpha] * n_clients)
    proportions = np.where(proportions < min_frac, 0.0, proportions)
    if proportions.sum() == 0:
        proportions = np.ones(n_clients) / n_clients
    proportions = proportions / proportions.sum()

    indices = rng.permutation(n_samples)
    cuts = (np.cumsum(proportions) * n_samples).astype(int)[:-1]
    parts = np.split(indices, cuts)
    return [p.tolist() for p in parts]


def partition_all(
    sizes: Dict[str, int],
    n_clients: int,
    alpha: float,
    seed: int,
) -> Dict[str, List[List[int]]]:
    """Partition every task independently with Dirichlet(alpha)."""
    out = {}
    for i, (task, n) in enumerate(sorted(sizes.items())):
        out[task] = dirichlet_partition(n, n_clients, alpha, seed=seed + i)
    return out


# ---------------------------------------------------------------------------
# GLUE  (BERT)
# ---------------------------------------------------------------------------
GLUE_TEXT_FIELDS = {
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
}

def load_glue(tokenizer, max_len: int = 128, cache_dir: str | None = None):
    from datasets import load_dataset

    train, evald = {}, {}
    for task in GLUE_TASKS:
        ds = load_dataset("glue", task, cache_dir=cache_dir)
        f1, f2 = GLUE_TEXT_FIELDS[task]

        def tok(batch, f1=f1, f2=f2):
            if f2 is None:
                enc = tokenizer(batch[f1], truncation=True, padding="max_length",
                                max_length=max_len)
            else:
                enc = tokenizer(batch[f1], batch[f2], truncation=True,
                                padding="max_length", max_length=max_len)
            enc["labels"] = batch["label"]
            return enc

        train[task] = ds["train"].map(tok, batched=True,
                                      remove_columns=ds["train"].column_names)
        eval_split = "validation_matched" if task == "mnli" else "validation"
        evald[task] = ds[eval_split].map(tok, batched=True,
                                         remove_columns=ds[eval_split].column_names)
        train[task].set_format("torch")
        evald[task].set_format("torch")
    return train, evald


# ---------------------------------------------------------------------------
# CIFAR  (ViT)
# ---------------------------------------------------------------------------
def load_cifar(image_processor, cache_dir: str | None = None):
    from datasets import load_dataset
    import torch

    train, evald = {}, {}
    name_map = {"cifar10": ("cifar10", "img", "label"),
                "cifar100": ("cifar100", "img", "fine_label")}

    for task, (hf_name, img_key, label_key) in name_map.items():
        ds = load_dataset(hf_name, cache_dir=cache_dir)

        def transform(batch, img_key=img_key, label_key=label_key):
            imgs = [im.convert("RGB") for im in batch[img_key]]
            enc = image_processor(imgs, return_tensors="pt")
            return {"pixel_values": enc["pixel_values"],
                    "labels": torch.tensor(batch[label_key])}

        ds["train"].set_transform(transform)
        ds["test"].set_transform(transform)
        train[task] = ds["train"]
        evald[task] = ds["test"]
    return train, evald


# ---------------------------------------------------------------------------
# Dolly-15k  (LLaMA, instruction-tuning)
# ---------------------------------------------------------------------------
DOLLY_CATEGORY_MAP = {
    "brainstorming": "brainstorming",
    "classification": "classification",
    "closed_qa": "closed_qa",
    "creative_writing": "creative_writing",
    "general_qa": "general_qa",
    "information_extraction": "information_extraction",
    "open_qa": "open_qa",
    "summarization": "summarization",
}

DOLLY_PROMPT = (
    "Below is an instruction that describes a task{ctx_hint}. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n{ctx_block}### Response:\n"
)

def _format_dolly(example):
    has_ctx = bool(example.get("context", "").strip())
    ctx_hint = ", paired with an input that provides further context" if has_ctx else ""
    ctx_block = f"### Input:\n{example['context']}\n\n" if has_ctx else ""
    prompt = DOLLY_PROMPT.format(ctx_hint=ctx_hint,
                                  instruction=example["instruction"],
                                  ctx_block=ctx_block)
    return prompt, example["response"]


def load_dolly(tokenizer, max_len: int = 512, eval_frac: float = 0.05,
               cache_dir: str | None = None, seed: int = 42):
    from datasets import load_dataset

    ds = load_dataset("databricks/databricks-dolly-15k", cache_dir=cache_dir)["train"]

    def tok(example):
        prompt, response = _format_dolly(example)
        full = prompt + response + tokenizer.eos_token
        enc = tokenizer(full, truncation=True, max_length=max_len,
                        padding="max_length")
        # mask prompt portion in labels
        prompt_ids = tokenizer(prompt, truncation=True, max_length=max_len)["input_ids"]
        labels = list(enc["input_ids"])
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100
        # mask padding
        for i, m in enumerate(enc["attention_mask"]):
            if m == 0:
                labels[i] = -100
        enc["labels"] = labels
        enc["category"] = example["category"]
        return enc

    ds = ds.map(tok, remove_columns=[c for c in ds.column_names if c != "category"])

    train, evald = {}, {}
    for task, cat in DOLLY_CATEGORY_MAP.items():
        sub = ds.filter(lambda x, c=cat: x["category"] == c)
        sub = sub.remove_columns(["category"])
        split = sub.train_test_split(test_size=eval_frac, seed=seed)
        split["train"].set_format("torch")
        split["test"].set_format("torch")
        train[task] = split["train"]
        evald[task] = split["test"]
    return train, evald


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
def task_sizes(train_dict) -> Dict[str, int]:
    return {k: len(v) for k, v in train_dict.items()}

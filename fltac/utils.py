"""Utilities: seeding, logging, evaluation."""
from __future__ import annotations
import json, os, random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class JsonlLogger:
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Overwrite per run -- experiments expect fresh logs each invocation
        self.f = open(self.path, "w", encoding="utf-8")

    def log(self, **record):
        self.f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


@torch.no_grad()
def evaluate_classification(peft_model, eval_ds, device, batch_size=64,
                            collate_fn=None, max_batches: int | None = None):
    from torch.utils.data import DataLoader
    loader = DataLoader(eval_ds, batch_size=batch_size, collate_fn=collate_fn)
    peft_model.eval()
    correct, total = 0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        labels = batch["labels"]
        out = peft_model(**{k: v for k, v in batch.items() if k != "labels"})
        preds = out.logits.argmax(dim=-1)
        if preds.shape != labels.shape:
            # CIFAR-10 has 10 classes but head is 100 — labels still align (0..9)
            pass
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_lm_loss(peft_model, eval_ds, device, batch_size=4, collate_fn=None,
                     max_batches: int | None = None):
    from torch.utils.data import DataLoader
    loader = DataLoader(eval_ds, batch_size=batch_size, collate_fn=collate_fn)
    peft_model.eval()
    loss_sum, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        out = peft_model(**batch)
        loss_sum += float(out.loss.item())
        n += 1
    return loss_sum / max(n, 1)

"""FL-TAC client.

A client owns:
  - a list of (task_name, sample_indices) for the tasks it is assigned
  - one LoRA adapter state-dict per local task

In each communication round it receives the current adapter state for each
of its local tasks, runs `local_steps` of fine-tuning per task, and returns
the updated adapter states.

Implementation note: instead of instantiating a separate model per task we
share one peft-wrapped backbone and swap adapter state-dicts in/out via
`set_adapter_state` between tasks. This keeps memory low (one base model)
while still implementing the per-task adapter semantics.
"""
from __future__ import annotations
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Subset

from .models import get_adapter_state, set_adapter_state


class FLClient:
    def __init__(self, client_id: int, task_to_indices: Dict[str, List[int]]):
        self.id = client_id
        self.task_to_indices = task_to_indices
        # Filled by server before round 0:
        self.adapters: Dict[str, dict] = {}

    @property
    def local_tasks(self) -> List[str]:
        return list(self.task_to_indices.keys())

    def num_samples(self, task: str) -> int:
        return len(self.task_to_indices[task])

    def local_finetune(
        self,
        peft_model,
        train_datasets,
        task: str,
        lr: float,
        local_steps: int,
        batch_size: int,
        device: torch.device,
        collate_fn=None,
    ) -> Tuple[dict, float]:
        """Run `local_steps` of SGD on `task`, return updated adapter state."""
        idxs = self.task_to_indices[task]
        if len(idxs) == 0:
            return self.adapters[task], 0.0

        ds = Subset(train_datasets[task], idxs)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            collate_fn=collate_fn, drop_last=False)

        # Load this client's task adapter into the model
        set_adapter_state(peft_model, self.adapters[task])
        peft_model.train()

        params = [p for p in peft_model.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(params, lr=lr)

        loss_sum, n = 0.0, 0
        step = 0
        it = iter(loader)
        while step < local_steps:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            out = peft_model(**batch)
            loss = out.loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_sum += float(loss.item())
            n += 1
            step += 1

        # Pull the new adapter state back
        new_state = get_adapter_state(peft_model)
        self.adapters[task] = new_state
        return new_state, loss_sum / max(n, 1)

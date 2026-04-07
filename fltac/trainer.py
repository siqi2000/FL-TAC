"""FL-TAC training loop (Algorithm 1).

Single-process simulation of N clients × K tasks. Supports:
  - GLUE / CIFAR / Dolly scenarios via the same loop
  - single-card or multi-card (multi-card via HF `device_map="auto"` for LLaMA;
    BERT/ViT are small enough to fit on one card)
  - FL-TAC method (per-task adapters + KMeans clustering)
  - 'fedit'    baseline (single shared adapter per client, plain FedAvg)
"""
from __future__ import annotations
import os, time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import default_collate

from . import data as data_mod
from .models import (build_base_model, attach_lora, get_adapter_state,
                     set_adapter_state, average_states)
from .client import FLClient
from .server import cluster_and_aggregate, assign_global_adapters, clustering_accuracy
from .utils import JsonlLogger, set_seed, evaluate_classification, evaluate_lm_loss


# ---------------------------------------------------------------------------
# Scenario glue
# ---------------------------------------------------------------------------
def load_scenario(cfg):
    sc = cfg["scenario"]
    if sc == "glue":
        model, tok, arch = build_base_model("glue", cfg["model_name"])
        train, evald = data_mod.load_glue(tok, max_len=cfg.get("max_len", 128),
                                          cache_dir=cfg.get("cache_dir"))
        collate = default_collate
        eval_fn = evaluate_classification
        return model, tok, arch, train, evald, collate, eval_fn

    if sc == "cifar":
        model, proc, arch = build_base_model("cifar", cfg["model_name"])
        train, evald = data_mod.load_cifar(proc, cache_dir=cfg.get("cache_dir"))
        def collate(batch):
            return {"pixel_values": torch.stack([b["pixel_values"] for b in batch]),
                    "labels": torch.tensor([int(b["labels"]) for b in batch])}
        eval_fn = evaluate_classification
        return model, proc, arch, train, evald, collate, eval_fn

    if sc == "dolly":
        model, tok, arch = build_base_model("dolly", cfg["model_name"])
        train, evald = data_mod.load_dolly(tok, max_len=cfg.get("max_len", 512),
                                           cache_dir=cfg.get("cache_dir"))
        collate = default_collate
        eval_fn = evaluate_lm_loss
        return model, tok, arch, train, evald, collate, eval_fn

    raise ValueError(f"unknown scenario {sc}")


# ---------------------------------------------------------------------------
# Main FL loop
# ---------------------------------------------------------------------------
def run(cfg: dict):
    set_seed(cfg.get("seed", 42))
    # Some torch+cuDNN+driver combos throw CUDNN_STATUS_NOT_INITIALIZED on the
    # first conv2d of ViT. Disabling cuDNN is a safe (slightly slower) fallback.
    if cfg.get("disable_cudnn", False) or cfg["scenario"] == "cifar":
        torch.backends.cudnn.enabled = False
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(out_dir / "metrics.jsonl")
    cluster_logger = JsonlLogger(out_dir / "cluster_assignments.jsonl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Build base model + LoRA wrapper (one shared model, swap adapters)
    base, tok_or_proc, arch, train_ds, eval_ds, collate, eval_fn = load_scenario(cfg)
    peft_model = attach_lora(base, arch, rank=cfg["lora_rank"],
                             alpha=cfg.get("lora_alpha", 16),
                             dropout=cfg.get("lora_dropout", 0.05))
    if arch != "llama":
        peft_model.to(device)
    peft_model.print_trainable_parameters()

    tasks = sorted(train_ds.keys())
    n_tasks = len(tasks)
    n_clients = cfg["n_clients"]
    method = cfg.get("method", "fltac")  # 'fltac' or 'fedit'

    # 2. Dirichlet partition each task across clients
    sizes = data_mod.task_sizes(train_ds)
    partitions = data_mod.partition_all(sizes, n_clients,
                                        alpha=cfg.get("dirichlet_alpha", 0.5),
                                        seed=cfg.get("seed", 42))

    # Build clients
    clients: List[FLClient] = []
    for cid in range(n_clients):
        t2i = {}
        for t in tasks:
            idxs = partitions[t][cid]
            if len(idxs) > 0:
                t2i[t] = idxs
        clients.append(FLClient(cid, t2i))

    # 3. Initialize adapters
    init_state = get_adapter_state(peft_model)
    if method == "fltac":
        for c in clients:
            c.adapters = {t: {k: v.clone() for k, v in init_state.items()}
                          for t in c.local_tasks}
    else:  # fedit: one shared adapter per client
        for c in clients:
            c.adapters = {"_shared": {k: v.clone() for k, v in init_state.items()}}

    # 4. Communication rounds
    n_rounds = cfg["rounds"]
    local_steps = cfg["local_steps"]
    bs = cfg["batch_size"]
    lr = cfg["lr"]
    eval_every = cfg.get("eval_every", 1)
    max_eval_batches = cfg.get("max_eval_batches", None)

    for rnd in range(1, n_rounds + 1):
        t0 = time.time()
        # client selection (full participation by default)
        sel = list(range(n_clients))
        submissions = []  # (client_id, task, adapter_state, n_samples)

        for cid in sel:
            client = clients[cid]
            if method == "fltac":
                for t in client.local_tasks:
                    state, loss = client.local_finetune(
                        peft_model, train_ds, t, lr, local_steps, bs, device,
                        collate_fn=collate)
                    submissions.append((cid, t, state, client.num_samples(t)))
                    logger.log(round=rnd, phase="local", client=cid, task=t,
                               loss=loss, n=client.num_samples(t))
            else:  # fedit baseline: one shared adapter, all tasks mixed
                # We temporarily store the shared adapter under each task slot
                # so local_finetune (which expects self.adapters[task]) works,
                # then collect the final state.
                shared = client.adapters["_shared"]
                steps_per_task = max(1, local_steps // max(1, len(client.local_tasks)))
                cur = shared
                for t in client.local_tasks:
                    client.adapters[t] = cur
                    cur, loss = client.local_finetune(
                        peft_model, train_ds, t, lr, steps_per_task, bs,
                        device, collate_fn=collate)
                    logger.log(round=rnd, phase="local", client=cid, task=t,
                               loss=loss, n=client.num_samples(t))
                client.adapters = {"_shared": cur}
                submissions.append((cid, "_shared", cur,
                                    sum(len(v) for v in client.task_to_indices.values())))

        # 5. Server aggregation
        if method == "fltac":
            cluster_states, labels, true_tasks = cluster_and_aggregate(
                submissions, n_clusters=n_tasks, seed=cfg.get("seed", 42))
            cacc = clustering_accuracy(labels, true_tasks)
            cluster_logger.log(round=rnd, labels=labels, true_tasks=true_tasks,
                               clustering_accuracy=cacc)
            new_for = assign_global_adapters(submissions, cluster_states, labels)
            for cid in sel:
                for t in clients[cid].local_tasks:
                    if (cid, t) in new_for:
                        clients[cid].adapters[t] = new_for[(cid, t)]
            # Save per-cluster aggregated adapters for offline UMAP plots
            torch.save({"round": rnd, "cluster_states": cluster_states,
                        "labels": labels, "true_tasks": true_tasks},
                       out_dir / f"adapters_round_{rnd}.pt")
        else:
            # plain FedAvg over the shared adapter
            states = [s[2] for s in submissions]
            sizes_w = [s[3] for s in submissions]
            tot = sum(sizes_w) or 1
            weights = [w / tot for w in sizes_w]
            new_state = average_states(states, weights)
            for c in clients:
                c.adapters["_shared"] = new_state

        # 6. Evaluation (on global aggregated adapter for each task)
        if rnd % eval_every == 0:
            for t in tasks:
                if method == "fltac":
                    # use any client's copy of task t (they're all equal post-aggreg)
                    holder = next((c for c in clients if t in c.adapters), None)
                    if holder is None:
                        continue
                    set_adapter_state(peft_model, holder.adapters[t])
                else:
                    set_adapter_state(peft_model, clients[0].adapters["_shared"])
                metric = eval_fn(peft_model, eval_ds[t], device,
                                 collate_fn=collate,
                                 max_batches=max_eval_batches)
                logger.log(round=rnd, phase="eval", task=t, metric=float(metric))

        dur = time.time() - t0
        logger.log(round=rnd, phase="round_done", seconds=dur)
        print(f"[round {rnd}/{n_rounds}] done in {dur:.1f}s")

    logger.close()
    cluster_logger.close()

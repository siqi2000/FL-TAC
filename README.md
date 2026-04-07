# FL-TAC: Enhanced Fine-Tuning in Federated Learning via Low-Rank, Task-Specific Adapter Clustering

[![Paper](https://img.shields.io/badge/arXiv-2404.15384-b31b1b.svg)](https://arxiv.org/abs/2404.15384)
[![Conference](https://img.shields.io/badge/ICLR-2024-blue)](https://iclr.cc/)

Official implementation of **"FL-TAC: Enhanced Fine-Tuning in Federated Learning via Low-Rank, Task-Specific Adapter Clustering"** (ICLR 2024).

> Siqi Ping*, Yuzhu Mao*, Yang Liu, Xiao-Ping Zhang, Wenbo Ding
> *Equal contribution.* Tsinghua-Berkeley Shenzhen Institute.

---

## Overview

Fine-tuning large pre-trained models in a Federated Learning (FL) setting is bottlenecked by communication cost. LoRA reduces this cost by transmitting only low-rank adapters, but aggressively shrinking the rank hurts performance — especially when each client must handle **multiple downstream tasks** with diverse data distributions.

**FL-TAC** addresses this by giving each client **one low-rank adapter per local task** instead of a single shared adapter, then clustering similar adapters on the server side and aggregating within each cluster. This recovers strong per-task performance at a *lower* communication budget than the standard FedIT baseline.

<p align="center">
  <img src="assets/framework.png" alt="FL-TAC framework" width="720"/>
  <br><em>Figure 1. (a) Each client trains one LoRA adapter per local task and uploads them to the server.
  (b) The server runs K-means on received adapters and performs FedAvg within each cluster, producing N task-specific global adapters.</em>
</p>

### Key idea

For each communication round t:

1. **Local fine-tuning** — Each selected client `i` updates one low-rank adapter `v_{i,j}` per local task `j ∈ N_i`, while keeping the pre-trained backbone `θ` frozen.
2. **Server clustering** — The server collects all submitted adapters from all clients across all tasks, flattens them into vectors, and runs **K-means with K = N** (total number of tasks in the system).
3. **Cluster-wise FedAvg** — Adapters within the same cluster are averaged (weighted by sample count) into a new global task-specific adapter `v^t_n`.

The result: each downstream task ends up with its own dedicated global adapter, even though no task labels are transmitted to the server.

---

## Experiments

We evaluate on three scenarios spanning text generation, text classification, and image classification.

| Scenario | Base Model | Datasets | # Tasks |
|---|---|---|---|
| Instruction tuning | LLaMA-7B | Databricks-Dolly-15k | 8 |
| Text classification | BERT-base | GLUE: SST-2, MRPC, QQP, QNLI, RTE | 5 |
| Image classification | ViT-base (ImageNet-21k) | CIFAR-10, CIFAR-100 | 2 |

All scenarios use **10 clients** with a **Dirichlet(α = 0.5)** partition (Hsu et al., 2020) of each task across clients.

### Baselines per scenario

| Scenario | Baselines |
|---|---|
| **Dolly + LLaMA-7B** | LLaMA-7B (no fine-tune) · LLaMA-7B-LoRA (centralized, rank 1) · **FedIT** (FL, LoRA rank 16, single shared adapter) |
| **GLUE + BERT** | BERT (centralized full FT) · BERT-LoRA (centralized) · **FedIT** (FL, single shared adapter) |
| **CIFAR + ViT** | ViT (centralized full FT) · ViT-LoRA (centralized) · **FedIT** (FL, single shared adapter) |

### Main results (from the paper)

**Databricks-Dolly-15k** (GPT-4 scoring):

| Method | Avg. score | Trainable params |
|---|---|---|
| LLaMA-7B (no FT) | 0.770 | 6.6 B |
| LLaMA-7B-LoRA (CL) | **0.905** | 4.26 M |
| FedIT (FL) | 0.671 | 8.52 M |
| **FL-TAC (ours)** | **0.705** | **4.26 M** |

**GLUE (BERT) and Images (ViT)**:

| Method | QNLI | QQP | SST-2 | RTE | MRPC | CIFAR-10 | CIFAR-100 |
|---|---|---|---|---|---|---|---|
| FedIT | 0.670 | 0.510 | 0.652 | 0.524 | 0.520 | 0.935 | 0.854 |
| **FL-TAC** | **0.787** | **0.807** | **0.874** | **0.532** | **0.673** | **0.946** | **0.884** |

FL-TAC also uses **fewer** trainable params than FedIT in both BERT (192 K vs 614 K) and ViT (36.8 K vs 294.9 K) settings.

---

## Repository structure

```
FL-TAC/
├── configs/                    # YAML configs, one per scenario
│   ├── glue_bert.yaml
│   ├── cifar_vit.yaml
│   └── dolly_llama.yaml
├── fltac/
│   ├── data.py                 # Dolly / GLUE / CIFAR + Dirichlet partition
│   ├── models.py               # base model + PEFT/LoRA wrapping
│   ├── client.py               # FL client (multi-task local fine-tuning)
│   ├── server.py               # K-means clustering + per-cluster FedAvg
│   ├── trainer.py              # FL training loop (Algorithm 1)
│   └── utils.py                # logging, eval, seeding
├── scripts/
│   ├── run_glue.sh
│   ├── run_cifar.sh
│   └── run_dolly.sh
├── main.py                     # entry point
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/<your-username>/FL-TAC.git
cd FL-TAC
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.1, transformers 4.40, peft 0.10, CUDA 12.4.

---

## Datasets

All datasets are pulled automatically from the HuggingFace Hub on first run; no manual preparation is needed. They will be cached under `./hf_cache/` (override with `cache_dir` in the config).

| Dataset | HuggingFace ID |
|---|---|
| Databricks-Dolly-15k | [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k) |
| GLUE | [`glue`](https://huggingface.co/datasets/glue) (configs: `sst2`, `mrpc`, `qqp`, `qnli`, `rte`) |
| CIFAR-10 | [`cifar10`](https://huggingface.co/datasets/cifar10) |
| CIFAR-100 | [`cifar100`](https://huggingface.co/datasets/cifar100) |

For LLaMA-7B you'll also need access to either `huggyllama/llama-7b` or `meta-llama/Llama-2-7b-hf` on the Hub.

---

## Reproducing the paper

```bash
# GLUE + BERT  (single 8 GB GPU is enough)
bash scripts/run_glue.sh

# CIFAR-10/100 + ViT  (single 8 GB GPU is enough)
bash scripts/run_cifar.sh

# Databricks-Dolly-15k + LLaMA-7B
#   single card  (>= 24 GB):
CUDA_VISIBLE_DEVICES=0 bash scripts/run_dolly.sh
#   four cards (recommended for speed):
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_dolly.sh
```

For LLaMA-7B we use HuggingFace's `device_map="auto"` to shard the backbone across all visible GPUs, so the **same code runs on single-card and multi-card** without changes.

### Running the FedIT baseline

Override the method on the command line:

```bash
python main.py --config configs/glue_bert.yaml --override method=fedit \
  output_dir=results/glue_bert_fedit
```

### Hyper-parameters

Default hyper-parameters per scenario (see the YAML files for the full list):

| Scenario | LoRA rank | LR | Batch | Local steps | Rounds |
|---|---|---|---|---|---|
| GLUE / BERT | 4 | 5e-4 | 32 | 50 | 20 |
| CIFAR / ViT | 4 | 1e-3 | 64 | 50 | 20 |
| Dolly / LLaMA-7B | 8 | 3e-4 | 4 | 30 | 10 |

All scenarios: 10 clients, Dirichlet α = 0.5, seed = 42.

---

## Hardware requirements

| Scenario | Min GPU | Recommended |
|---|---|---|
| GLUE + BERT | 8 GB | any modern GPU |
| CIFAR + ViT | 8 GB | any modern GPU |
| Dolly + LLaMA-7B | 24 GB (single) | A100 40 GB / 4 × RTX 4090 |

---

## Output format

Each run writes to `<output_dir>/`:

- `metrics.jsonl` — one JSON record per local-step / eval / round-end:
  ```json
  {"round": 3, "phase": "eval", "task": "sst2", "metric": 0.81}
  {"round": 3, "phase": "local", "client": 2, "task": "sst2", "loss": 0.42, "n": 124}
  ```
- `cluster_assignments.jsonl` — K-means cluster id and matched task labels per round, plus clustering accuracy.
- `adapters_round_<t>.pt` — aggregated cluster adapters at round `t`, suitable for offline UMAP visualization (Fig. 4 in the paper).

---

## Citation

```bibtex
@inproceedings{ping2024fltac,
  title     = {FL-TAC: Enhanced Fine-Tuning in Federated Learning via Low-Rank, Task-Specific Adapter Clustering},
  author    = {Ping, Siqi and Mao, Yuzhu and Liu, Yang and Zhang, Xiao-Ping and Ding, Wenbo},
  booktitle = {ICLR Workshop},
  year      = {2024}
}
```

---

## Acknowledgements

This work was supported by the Shenzhen Key Laboratory of Ubiquitous Data Enabling (ZDSYS20220527171406015), the National Key R&D Program of China under Grant No. 2022ZD0160504, and Meituan.

The federated instruction-tuning baseline is adapted from [FedIT (Zhang et al., 2023)](https://github.com/JayZhang42/FederatedGPT-Shepherd).

# FL-TAC: Enhanced Fine-Tuning in Federated Learning via Low-Rank, Task-Specific Adapter Clustering

<p align="center"><b>📖 <a href="README.md">English</a> · <a href="README_zh.md">中文</a></b></p>

<p align="center">
  <a href="https://arxiv.org/abs/2404.15384"><img src="https://img.shields.io/badge/arXiv-2404.15384-b31b1b.svg" alt="arXiv"/></a>
  <a href="https://llmagents.github.io/"><img src="https://img.shields.io/badge/ICLR%202024-LLM%20Agents%20Workshop-blue" alt="Venue"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"/></a>
</p>

Official implementation of **"FL-TAC: Enhanced Fine-Tuning in Federated Learning via Low-Rank, Task-Specific Adapter Clustering"**, accepted at the **ICLR 2024 Workshop on Large Language Model (LLM) Agents**.

📄 **Paper:** [arXiv:2404.15384](https://arxiv.org/abs/2404.15384)

### Authors

**Siqi Ping**<sup>1,†</sup>, **Yuzhu Mao**<sup>1,†</sup>, Yang Liu<sup>2,3</sup>, Xiao-Ping Zhang<sup>1,5</sup>, Wenbo Ding<sup>1,3,4,*</sup>

<sup>1</sup> Tsinghua-Berkeley Shenzhen Institute, Tsinghua Shenzhen International Graduate School, Tsinghua University
<sup>2</sup> Institute for AI Industry Research (AIR), Tsinghua University
<sup>3</sup> Shanghai AI Lab, Shanghai, China
<sup>4</sup> RISC-V International Open Source Laboratory, Shenzhen, China
<sup>5</sup> Department of Electrical, Computer and Biomedical Engineering, Ryerson University

<sup>†</sup> *Equal contribution.* &nbsp;&nbsp; <sup>*</sup> *Corresponding author* (`ding.wenbo@sz.tsinghua.edu.cn`).

---

## Table of Contents

- [Overview](#overview)
- [Experiments](#experiments)
  - [Baselines per scenario](#baselines-per-scenario)
  - [Main results](#main-results-from-the-paper)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Data partition](#data-partition)
- [Reproducing the paper](#reproducing-the-paper)
  - [Scenario 1 — GLUE text classification (BERT)](#scenario-1--glue-text-classification-bert)
  - [Scenario 2 — CIFAR image classification (ViT)](#scenario-2--cifar-image-classification-vit)
  - [Scenario 3 — Databricks-Dolly-15k instruction tuning (LLaMA-7B)](#scenario-3--databricks-dolly-15k-instruction-tuning-llama-7b)
  - [Default hyper-parameters](#default-hyper-parameters)
- [Evaluation](#evaluation)
- [Hardware requirements](#hardware-requirements)
- [Output format](#output-format)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Overview

Fine-tuning large pre-trained models in a Federated Learning (FL) setting is bottlenecked by communication cost. LoRA reduces this cost by transmitting only low-rank adapters, but aggressively shrinking the rank hurts performance — especially when each client must handle **multiple downstream tasks** with diverse data distributions.

**FL-TAC** addresses this by giving each client **one low-rank adapter per local task** instead of a single shared adapter, then clustering similar adapters on the server side and aggregating within each cluster. This recovers strong per-task performance at a *lower* communication budget than the standard FedIT baseline.

<p align="center">
  <img src="assets/framework.png" alt="FL-TAC framework" width="780"/>
  <br><em>Figure 1. (a) Each client trains one LoRA adapter per local task and uploads them to the server.
  (b) The server runs K-means on received adapters and performs FedAvg within each cluster, producing N task-specific global adapters.</em>
</p>

<p align="center">
  <img src="assets/approx_error_vs_rank.png" alt="Approximation error vs LoRA rank" width="520"/>
  <br><em>Figure 3. Toy simulation: a single shared adapter (red) needs much higher rank than per-task adapters (green) to reach the same approximation error — motivating the per-task design of FL-TAC.</em>
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

<p align="center">
  <img src="assets/data_distribution_and_radar.png" alt="Data distribution and Dolly radar" width="820"/>
  <br><em>Figure 2. (a) Per-client task data proportions under Dirichlet(α = 0.5).
  (b) Radar chart of FL-TAC vs LLaMA / LLaMA-LoRA / FedIT on the eight Databricks-Dolly-15k tasks.</em>
</p>

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

<p align="center">
  <img src="assets/umap_clustering.png" alt="UMAP clustering progression" width="820"/>
  <br><em>Figure 4. UMAP visualization of K-means clustering of received adapters at the server, from epoch 1 (left) to epoch 9 (right). Clusters become increasingly separated as training progresses, confirming that LoRA adapters carry strong task identity even without explicit task labels.</em>
</p>

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

| Dataset | HuggingFace Hub | Original source |
|---|---|---|
| Databricks-Dolly-15k | [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | [databrickslabs/dolly](https://github.com/databrickslabs/dolly) |
| GLUE (SST-2, MRPC, QQP, QNLI, RTE) | [`nyu-mll/glue`](https://huggingface.co/datasets/nyu-mll/glue) | [gluebenchmark.com](https://gluebenchmark.com/) |
| CIFAR-10 | [`uoft-cs/cifar10`](https://huggingface.co/datasets/uoft-cs/cifar10) | [cs.toronto.edu/~kriz/cifar](https://www.cs.toronto.edu/~kriz/cifar.html) |
| CIFAR-100 | [`uoft-cs/cifar100`](https://huggingface.co/datasets/uoft-cs/cifar100) | [cs.toronto.edu/~kriz/cifar](https://www.cs.toronto.edu/~kriz/cifar.html) |

For LLaMA-7B you'll also need [`huggyllama/llama-7b`](https://huggingface.co/huggyllama/llama-7b) (ungated) or [`meta-llama/Llama-2-7b-hf`](https://huggingface.co/meta-llama/Llama-2-7b-hf) (requires HuggingFace token).

> **For users in mainland China**: prepend `export HF_ENDPOINT=https://hf-mirror.com` to use the HuggingFace mirror.

---

## Data partition

All three scenarios use a **per-task Dirichlet(α) split** following Hsu et al., 2020 — the standard non-IID partition for FL benchmarking. For each task independently:

1. Sample a vector `p ~ Dir(α, ..., α)` of length `n_clients` (we use **n_clients = 10**, **α = 0.5**).
2. Zero out any entry below `min_frac = 0.01` and renormalise — this gives some clients zero data for that task, modelling realistic data sparsity.
3. Partition the task's training samples across the 10 clients in those proportions.

A different `seed` is used for each task (`seed + task_index`) so the per-task splits are independent. The implementation is in [`fltac/data.py`](fltac/data.py) (`dirichlet_partition` and `partition_all`).

You can inspect the exact split for any scenario **without downloading anything or loading a model** using the helper script:

```bash
# Default: 10 clients, alpha=0.5, seed=42
python scripts/inspect_partition.py --scenario glue
python scripts/inspect_partition.py --scenario cifar
python scripts/inspect_partition.py --scenario dolly

# Try a more skewed split, save to CSV
python scripts/inspect_partition.py --scenario glue --alpha 0.1 \
    --save partition.csv
```

Example output for `--scenario glue` (default config):

```
=== GLUE partition (n_clients=10, alpha=0.5, min_frac=0.01, seed=42) ===

client      sst2    mrpc     qqp    qnli     rte    total
---------------------------------------------------------
#0          9765       0       0    4435    1029    15229
#1         15540     263   12764   10121      43    38731
#2             0     536       0   11902     245    12683
#3          9174     131    8855    2208     151    20519
#4             0    2164   31032    5540     518    39254
#5          1669      58   13121    8412     379    23639
#6          5443      71  165334   43650       0   214498
#7          3777     175       0   12879      88    16919
#8         12940     270  132740    5596      36   151582
#9          9041       0       0       0       1     9042
---------------------------------------------------------
total      67349    3668  363846  104743    2490   542096
```

The runs that produced our results in this README all use **n_clients = 10, α = 0.5, seed = 42**. Pass `--n_clients`, `--alpha`, `--seed` to the training script (or this inspection script) to try different splits — the partition is fully reproducible from those four inputs.

---

## Reproducing the paper

All three scenarios use the same federated setup: **10 clients, Dirichlet(α = 0.5) data partition** on every task (Hsu et al., 2020), seed = 42, full client participation each round. See [Data partition](#data-partition) above for details.

### Scenario 1 — GLUE text classification (BERT)

| | |
|---|---|
| **Backbone** | `bert-base-uncased` |
| **Datasets** | GLUE: SST-2, MRPC, QQP, QNLI, RTE (5 tasks) |
| **Eval metric** | accuracy on the official GLUE validation split |
| **Hardware** | 1 × GPU with ≥ 8 GB |

```bash
# FL-TAC (ours)
python main.py --config configs/glue_bert.yaml

# FedIT baseline (single shared adapter)
python main.py --config configs/glue_bert.yaml --method fedit \
    --output_dir results/glue_bert_fedit
```

### Scenario 2 — CIFAR image classification (ViT)

| | |
|---|---|
| **Backbone** | `google/vit-base-patch16-224-in21k` |
| **Datasets** | CIFAR-10, CIFAR-100 (2 tasks) |
| **Eval metric** | top-1 accuracy on the official test split |
| **Hardware** | 1 × GPU with ≥ 8 GB |

```bash
# FL-TAC (ours)
python main.py --config configs/cifar_vit.yaml

# FedIT baseline
python main.py --config configs/cifar_vit.yaml --method fedit \
    --output_dir results/cifar_vit_fedit
```

### Scenario 3 — Databricks-Dolly-15k instruction tuning (LLaMA-7B)

| | |
|---|---|
| **Backbone** | `huggyllama/llama-7b` (ungated) — or `meta-llama/Llama-2-7b-hf` (gated) |
| **Dataset** | Databricks-Dolly-15k, split by its 8 task categories |
| **Code metric** | per-task language-modelling cross-entropy loss on a held-out 5 % split |
| **Paper metric** | **GPT-4 scoring** on generated answers (see [Evaluation](#evaluation) below) |
| **Hardware** | 1 × 24 GB GPU minimum, 2–4 × RTX 4090 / 1 × A100 recommended |

```bash
# FL-TAC (ours) — single card
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/dolly_llama.yaml \
    --model_name huggyllama/llama-7b

# FL-TAC (ours) — multi-card via device_map="auto"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config configs/dolly_llama.yaml \
    --model_name huggyllama/llama-7b

# FedIT baseline
CUDA_VISIBLE_DEVICES=0,1 python main.py --config configs/dolly_llama.yaml \
    --model_name huggyllama/llama-7b --method fedit \
    --output_dir results/dolly_llama_fedit
```

For LLaMA-7B we rely on HuggingFace's `device_map="auto"` to shard the backbone across all visible GPUs. **The same code runs on single-card and multi-card** without changes — just set `CUDA_VISIBLE_DEVICES`.

### Default hyper-parameters

Defined in the YAML configs; can be overridden on the command line via either named flags (`--rounds 30 --lr 1e-4`) or generic `--override key=value`.

| Scenario | LoRA rank | LR | Batch | Local steps | Rounds |
|---|---|---|---|---|---|
| GLUE / BERT | 4 | 5e-4 | 32 | 50 | 20 |
| CIFAR / ViT | 4 | 1e-3 | 64 | 50 | 20 |
| Dolly / LLaMA-7B | 8 | 3e-4 | 4 | 30 | 10 |

All scenarios: 10 clients, Dirichlet α = 0.5, seed = 42.

---

## Evaluation

| Scenario | What `metrics.jsonl` contains | How the paper scores it |
|---|---|---|
| **GLUE** | top-1 **accuracy** per task per eval round | same — accuracy on the GLUE validation split |
| **CIFAR** | top-1 **accuracy** per task per eval round | same — accuracy on the test split |
| **Dolly** | **language-modelling loss** (cross-entropy) per task per eval round | **GPT-4 scoring** on generated answers, using the protocol of [FedIT (Zhang et al., 2023)](https://github.com/JayZhang42/FederatedGPT-Shepherd) |

For GLUE and CIFAR our `metrics.jsonl` numbers are directly comparable to the paper.

For **Dolly**, the paper uses GPT-4 to score the model's generated answers against the reference responses on a 0–1 scale, then averages per category. Our `metrics.jsonl` reports LM loss instead, which is fast to compute and useful for sanity-checking convergence, but is **not** the same metric as the paper's. To exactly reproduce the paper's Dolly numbers you need to:

1. Load the final FL-TAC adapter for each task and `model.generate(...)` answers on the held-out split.
2. Send `(reference, prediction)` pairs to a GPT-4 model with a scoring prompt.
3. Average the returned scores per task.

A starter script is provided at [`scripts/eval_dolly_gpt4.py`](scripts/eval_dolly_gpt4.py) — fill in your `OPENAI_API_KEY` and run after the FL-TAC training has finished.

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
  booktitle = {ICLR 2024 Workshop on Large Language Model (LLM) Agents},
  year      = {2024},
  url       = {https://arxiv.org/abs/2404.15384}
}
```

---

## Acknowledgements

This work was supported by the Shenzhen Key Laboratory of Ubiquitous Data Enabling (ZDSYS20220527171406015), the National Key R&D Program of China under Grant No. 2022ZD0160504, and Meituan.

The federated instruction-tuning baseline is adapted from [FedIT (Zhang et al., 2023)](https://github.com/JayZhang42/FederatedGPT-Shepherd).

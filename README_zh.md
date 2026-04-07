# FL-TAC:基于低秩任务特定适配器聚类的联邦学习微调方法

<p align="center"><b>📖 <a href="README.md">English</a> · <a href="README_zh.md">中文</a></b></p>

<p align="center">
  <a href="https://arxiv.org/abs/2404.15384"><img src="https://img.shields.io/badge/arXiv-2404.15384-b31b1b.svg" alt="arXiv"/></a>
  <a href="https://llmagents.github.io/"><img src="https://img.shields.io/badge/ICLR%202024-LLM%20Agents%20Workshop-blue" alt="Venue"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"/></a>
</p>

论文 **"FL-TAC: Enhanced Fine-Tuning in Federated Learning via Low-Rank, Task-Specific Adapter Clustering"** 的官方代码实现。本工作发表于 **ICLR 2024 大语言模型智能体研讨会(LLM Agents Workshop)**。

📄 **论文链接:** [arXiv:2404.15384](https://arxiv.org/abs/2404.15384)

### 作者

**平思琪**<sup>1,†</sup>, **毛宇柱**<sup>1,†</sup>, 刘洋<sup>2,3</sup>, 张晓平<sup>1,5</sup>, 丁文伯<sup>1,3,4,*</sup>

<sup>1</sup> 清华大学深圳国际研究生院,清华-伯克利深圳学院
<sup>2</sup> 清华大学智能产业研究院(AIR)
<sup>3</sup> 上海人工智能实验室
<sup>4</sup> 深圳市 RISC-V 国际开源实验室
<sup>5</sup> 多伦多都会大学(原瑞尔森大学)电气、计算机与生物医学工程系

<sup>†</sup> *共同第一作者* &nbsp;&nbsp; <sup>*</sup> *通讯作者*(`ding.wenbo@sz.tsinghua.edu.cn`)

---

## 目录

- [简介](#简介)
- [实验](#实验)
  - [各场景对应的 Baseline](#各场景对应的-baseline)
  - [论文主要结果](#论文主要结果)
- [仓库结构](#仓库结构)
- [安装](#安装)
- [数据集](#数据集)
- [数据切分](#数据切分)
- [复现实验](#复现实验)
  - [场景 1 —— GLUE 文本分类(BERT)](#场景-1--glue-文本分类bert)
  - [场景 2 —— CIFAR 图像分类(ViT)](#场景-2--cifar-图像分类vit)
  - [场景 3 —— Databricks-Dolly-15k 指令微调(LLaMA-7B)](#场景-3--databricks-dolly-15k-指令微调llama-7b)
  - [修改超参的两种方式](#修改超参的两种方式)
  - [默认超参](#默认超参)
- [评估说明](#评估说明)
- [硬件需求](#硬件需求)
- [输出格式](#输出格式)
- [引用](#引用)
- [致谢](#致谢)

---

## 简介

在联邦学习(FL)框架下微调大规模预训练模型时,通信开销是主要瓶颈。LoRA 通过仅传输低秩适配器降低了通信代价,但当 rank 被压得过低时模型性能会显著下降——尤其是当**每个客户端持有多种下游任务**且数据分布异构时。

**FL-TAC** 的核心思想:让每个客户端**为本地的每一个任务训练一个独立的低秩 LoRA 适配器**,而不是只用一个共享适配器;服务端收到所有适配器后,通过 K-means 聚类将相似的(隐含同任务的)适配器分到一起,然后在每个簇内做加权 FedAvg 聚合。这样在**比 FedIT 基线更低的通信预算下**,获得了更强的多任务性能。

<p align="center">
  <img src="assets/framework.png" alt="FL-TAC 框架图" width="780"/>
  <br><em>图 1. (a) 每个客户端为每一个本地任务训练一个 LoRA 适配器并上传到服务端。
  (b) 服务端对收到的适配器做 K-means 聚类,并在每个簇内做 FedAvg 聚合,得到 N 个全局任务特定适配器。</em>
</p>

<p align="center">
  <img src="assets/approx_error_vs_rank.png" alt="近似误差 vs LoRA rank" width="520"/>
  <br><em>图 3. 玩具实验:单个共享适配器(红线)需要远高于"每任务一个适配器"(绿线)的 rank,才能达到相同的近似误差——这就是 FL-TAC 采用每任务适配器设计的动机。</em>
</p>

### 算法流程

每一轮通信 t:

1. **本地微调** —— 选中的客户端 `i` 对其本地的每个任务 `j ∈ N_i` 单独更新一个低秩适配器 `v_{i,j}`,预训练主干 `θ` 全程冻结。
2. **服务端聚类** —— 服务端把所有客户端、所有任务上传的适配器拉平成向量,统一跑一次 **K-means(K = 系统总任务数 N)**。
3. **簇内 FedAvg** —— 同一簇内的适配器按样本数加权平均,得到该簇对应任务的新全局适配器 `v^t_n`。

注意:服务端**不需要任何任务标签**,完全凭适配器自身的几何结构把同任务客户端识别出来。

---

## 实验

我们在三个场景上评估 FL-TAC,分别覆盖文本生成、文本分类、图像分类:

| 场景 | 主干模型 | 数据集 | 任务数 |
|---|---|---|---|
| 指令微调 | LLaMA-7B | Databricks-Dolly-15k | 8 |
| 文本分类 | BERT-base | GLUE: SST-2、MRPC、QQP、QNLI、RTE | 5 |
| 图像分类 | ViT-base(ImageNet-21k 预训练)| CIFAR-10、CIFAR-100 | 2 |

三个场景都使用 **10 个客户端**,每个任务的数据按 **Dirichlet(α = 0.5)** (Hsu et al., 2020)切分到客户端上。

<p align="center">
  <img src="assets/data_distribution_and_radar.png" alt="数据分布与 Dolly 雷达图" width="820"/>
  <br><em>图 2. (a) Dirichlet(α=0.5) 下各客户端的任务数据占比。
  (b) Databricks-Dolly-15k 八个任务上 FL-TAC 与 LLaMA / LLaMA-LoRA / FedIT 的雷达图对比。</em>
</p>

### 各场景对应的 Baseline

| 场景 | Baselines |
|---|---|
| **Dolly + LLaMA-7B** | LLaMA-7B(无微调) · LLaMA-7B-LoRA(集中式,rank 1) · **FedIT**(联邦,LoRA rank 16,单共享适配器) |
| **GLUE + BERT** | BERT(集中式全参微调)· BERT-LoRA(集中式)· **FedIT**(联邦,单共享适配器) |
| **CIFAR + ViT** | ViT(集中式全参微调)· ViT-LoRA(集中式)· **FedIT**(联邦,单共享适配器) |

### 论文主要结果

**Databricks-Dolly-15k**(由 GPT-4 评分):

| 方法 | 平均分 | 可训练参数量 |
|---|---|---|
| LLaMA-7B(无微调) | 0.770 | 6.6 B |
| LLaMA-7B-LoRA(集中式) | **0.905** | 4.26 M |
| FedIT(联邦) | 0.671 | 8.52 M |
| **FL-TAC(本文)** | **0.705** | **4.26 M** |

**GLUE(BERT)与图像分类(ViT)**:

| 方法 | QNLI | QQP | SST-2 | RTE | MRPC | CIFAR-10 | CIFAR-100 |
|---|---|---|---|---|---|---|---|
| FedIT | 0.670 | 0.510 | 0.652 | 0.524 | 0.520 | 0.935 | 0.854 |
| **FL-TAC** | **0.787** | **0.807** | **0.874** | **0.532** | **0.673** | **0.946** | **0.884** |

在 BERT(192 K vs 614 K)和 ViT(36.8 K vs 294.9 K)两个场景下,FL-TAC 的可训练参数量也都**少于** FedIT。

<p align="center">
  <img src="assets/umap_clustering.png" alt="UMAP 聚类演化" width="820"/>
  <br><em>图 4. 服务端 K-means 聚类结果的 UMAP 降维可视化,从 epoch 1(最左)到 epoch 9(最右)。可以清晰看到随着训练推进,不同任务的 adapter 在表示空间中越来越分得开,即使没有任何任务标签的监督。</em>
</p>

---

## 仓库结构

```
FL-TAC/
├── configs/                    # 每个场景一份 YAML 配置
│   ├── glue_bert.yaml
│   ├── cifar_vit.yaml
│   └── dolly_llama.yaml
├── fltac/
│   ├── data.py                 # Dolly / GLUE / CIFAR 加载 + Dirichlet 切分
│   ├── models.py               # 主干模型 + PEFT/LoRA 包装
│   ├── client.py               # 联邦客户端(多任务本地微调)
│   ├── server.py               # K-means 聚类 + 簇内 FedAvg
│   ├── trainer.py              # 联邦训练主循环(算法 1)
│   └── utils.py                # 日志、评估、随机种子
├── scripts/
│   ├── run_glue.sh
│   ├── run_cifar.sh
│   └── run_dolly.sh
├── main.py                     # 入口
└── requirements.txt
```

---

## 安装

```bash
git clone https://github.com/siqi2000/FL-TAC.git
cd FL-TAC
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

测试环境:Python 3.10、PyTorch 2.1、transformers 4.40、peft 0.10、CUDA 12.4。

---

## 数据集

所有数据集都会在第一次运行时自动从 HuggingFace Hub 下载,**不需要手动准备**,默认缓存到 `./hf_cache/`(可通过 config 里的 `cache_dir` 修改)。

| 数据集 | HuggingFace Hub | 原始来源 |
|---|---|---|
| Databricks-Dolly-15k | [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | [databrickslabs/dolly](https://github.com/databrickslabs/dolly) |
| GLUE(SST-2、MRPC、QQP、QNLI、RTE)| [`nyu-mll/glue`](https://huggingface.co/datasets/nyu-mll/glue) | [gluebenchmark.com](https://gluebenchmark.com/) |
| CIFAR-10 | [`uoft-cs/cifar10`](https://huggingface.co/datasets/uoft-cs/cifar10) | [cs.toronto.edu/~kriz/cifar](https://www.cs.toronto.edu/~kriz/cifar.html) |
| CIFAR-100 | [`uoft-cs/cifar100`](https://huggingface.co/datasets/uoft-cs/cifar100) | [cs.toronto.edu/~kriz/cifar](https://www.cs.toronto.edu/~kriz/cifar.html) |

LLaMA-7B 主干可以使用 [`huggyllama/llama-7b`](https://huggingface.co/huggyllama/llama-7b)(无需授权)或 [`meta-llama/Llama-2-7b-hf`](https://huggingface.co/meta-llama/Llama-2-7b-hf)(需要 HuggingFace token)。

> 国内用户可以加上 `export HF_ENDPOINT=https://hf-mirror.com` 走镜像。

---

## 数据切分

三个场景统一使用 **逐任务 Dirichlet(α) 切分** —— 这是 FL benchmark 里最常用的非 IID 切分方式(Hsu et al., 2020)。对每一个任务独立做以下操作:

1. 采样一个长度为 `n_clients` 的向量 `p ~ Dir(α, ..., α)`(我们用 **n_clients = 10**,**α = 0.5**)。
2. 把小于 `min_frac = 0.01` 的分量置零并重新归一化 —— 这样部分 client 在某个任务上会**完全没有数据**,更贴近真实的稀疏分布。
3. 按这些比例把该任务的训练集切分到 10 个 client 上。

每个任务使用不同的 seed(`seed + 任务下标`),保证不同任务的切分相互独立。具体实现见 [`fltac/data.py`](fltac/data.py) 中的 `dirichlet_partition` 和 `partition_all`。

我们还提供了一个**完全无需下载数据集、不需要加载模型**的轻量级脚本,可以快速查看任意场景的切分结果:

```bash
# 默认配置:10 个 client、alpha=0.5、seed=42
python scripts/inspect_partition.py --scenario glue
python scripts/inspect_partition.py --scenario cifar
python scripts/inspect_partition.py --scenario dolly

# 试一个更偏斜的切分,并保存为 CSV
python scripts/inspect_partition.py --scenario glue --alpha 0.1 \
    --save partition.csv
```

`--scenario glue` 默认配置下的输出示例:

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

本 README 里所有报告的实验结果统一使用 **n_clients = 10、α = 0.5、seed = 42**。把 `--n_clients`、`--alpha`、`--seed` 传给训练脚本(或这个 inspect 脚本)就可以尝试别的切分 —— 给定这四个输入,切分是完全可复现的。

---

## 复现实验

三个场景统一使用相同的联邦设置:**10 个客户端,所有任务的数据都按 Dirichlet(α = 0.5)** (Hsu et al., 2020)切分到客户端,seed = 42,每轮全员参与。具体细节见上面的 [数据切分](#数据切分) 章节。

### 场景 1 —— GLUE 文本分类(BERT)

| | |
|---|---|
| **主干** | `bert-base-uncased` |
| **数据集** | GLUE:SST-2、MRPC、QQP、QNLI、RTE(共 5 个任务)|
| **评估指标** | GLUE 官方 validation split 上的 accuracy |
| **硬件** | 单卡 ≥ 8 GB 显存即可 |

```bash
# FL-TAC(本文方法)
python main.py --config configs/glue_bert.yaml

# FedIT 基线(单共享 adapter)
python main.py --config configs/glue_bert.yaml --method fedit \
    --output_dir results/glue_bert_fedit
```

### 场景 2 —— CIFAR 图像分类(ViT)

| | |
|---|---|
| **主干** | `google/vit-base-patch16-224-in21k` |
| **数据集** | CIFAR-10、CIFAR-100(共 2 个任务)|
| **评估指标** | 官方 test split 上的 top-1 accuracy |
| **硬件** | 单卡 ≥ 8 GB 显存即可 |

```bash
# FL-TAC(本文方法)
python main.py --config configs/cifar_vit.yaml

# FedIT 基线
python main.py --config configs/cifar_vit.yaml --method fedit \
    --output_dir results/cifar_vit_fedit
```

### 场景 3 —— Databricks-Dolly-15k 指令微调(LLaMA-7B)

| | |
|---|---|
| **主干** | `huggyllama/llama-7b`(无授权) 或 `meta-llama/Llama-2-7b-hf`(需授权) |
| **数据集** | Databricks-Dolly-15k,按其 8 个任务类别(category)切分 |
| **代码默认指标** | 留出 5% 验证集上的逐任务语言模型交叉熵 loss |
| **论文使用指标** | **GPT-4 评分**(详见下方[评估说明](#评估说明)) |
| **硬件** | 最低单卡 24 GB,推荐 2–4 × RTX 4090 / 1 × A100 |

```bash
# FL-TAC(本文方法)—— 单卡
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/dolly_llama.yaml \
    --model_name huggyllama/llama-7b

# FL-TAC(本文方法)—— 多卡(device_map="auto" 自动切分)
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config configs/dolly_llama.yaml \
    --model_name huggyllama/llama-7b

# FedIT 基线
CUDA_VISIBLE_DEVICES=0,1 python main.py --config configs/dolly_llama.yaml \
    --model_name huggyllama/llama-7b --method fedit \
    --output_dir results/dolly_llama_fedit
```

LLaMA-7B 使用 HuggingFace 的 `device_map="auto"` 自动切分到所有可见 GPU,**同一份代码既能跑单卡也能跑多卡**,无需修改任何代码,只用 `CUDA_VISIBLE_DEVICES` 控制即可。

### 修改超参的两种方式

```bash
# 1) 显式命名参数(覆盖最常用的字段)
python main.py --config configs/glue_bert.yaml \
    --rounds 30 --n_clients 8 --lr 1e-4 --lora_rank 8 \
    --method fltac --output_dir results/my_run

# 2) 通用 --override(覆盖任意字段)
python main.py --config configs/glue_bert.yaml \
    --override max_eval_batches=10 dirichlet_alpha=0.3
```

### 默认超参

| 场景 | LoRA rank | 学习率 | Batch | 本地步数 | 通信轮数 |
|---|---|---|---|---|---|
| GLUE / BERT | 4 | 5e-4 | 32 | 50 | 20 |
| CIFAR / ViT | 4 | 1e-3 | 64 | 50 | 20 |
| Dolly / LLaMA-7B | 8 | 3e-4 | 4 | 30 | 10 |

三个场景统一:10 个客户端,Dirichlet α = 0.5,seed = 42。

---

## 评估说明

| 场景 | `metrics.jsonl` 里记录的指标 | 论文里使用的指标 |
|---|---|---|
| **GLUE** | 每个任务每个评估轮的 top-1 **accuracy** | 同左 — GLUE validation split 上的 accuracy |
| **CIFAR** | 每个任务每个评估轮的 top-1 **accuracy** | 同左 — test split 上的 accuracy |
| **Dolly** | 每个任务每个评估轮的**语言模型交叉熵 loss** | **GPT-4 评分**:沿用 [FedIT (Zhang et al., 2023)](https://github.com/JayZhang42/FederatedGPT-Shepherd) 的协议 |

GLUE 和 CIFAR 两个场景下,我们 `metrics.jsonl` 输出的数值可以直接和论文表里的数字对比。

**Dolly** 场景下,论文是用 GPT-4 给模型生成的回答打 0–1 分(对照 reference 回答),然后按任务类别求均值。我们代码默认输出的是 LM loss,这个指标计算快、可以用来观察收敛趋势,但**和论文里的 GPT-4 评分不是同一个量**。要严格复现论文的 Dolly 表格数字,需要:

1. 加载训练好的 FL-TAC 任务 adapter,用 `model.generate(...)` 在留出集上生成回答。
2. 把 `(reference, prediction)` 配对发给 GPT-4,使用一个评分提示词。
3. 按任务类别取平均。

我们提供了一个起步脚本 [`scripts/eval_dolly_gpt4.py`](scripts/eval_dolly_gpt4.py) — 填入你的 `OPENAI_API_KEY`,在 FL-TAC 训练结束后运行即可。

---

## 硬件需求

| 场景 | 最低 GPU | 推荐 |
|---|---|---|
| GLUE + BERT | 8 GB | 任意现代 GPU |
| CIFAR + ViT | 8 GB | 任意现代 GPU |
| Dolly + LLaMA-7B | 单卡 24 GB | A100 40GB / 4 × RTX 4090 |

---

## 输出格式

每次运行会在 `<output_dir>/` 下写入:

- `metrics.jsonl` —— 每行一条 JSON 记录(本地训练 loss / 评估指标 / 每轮总耗时):
  ```json
  {"round": 3, "phase": "eval", "task": "sst2", "metric": 0.81}
  {"round": 3, "phase": "local", "client": 2, "task": "sst2", "loss": 0.42, "n": 124}
  ```
- `cluster_assignments.jsonl` —— 每轮 K-means 的聚类结果、真实任务标签、聚类准确率。
- `adapters_round_<t>.pt` —— 第 `t` 轮聚合后的所有簇适配器,可用于离线 UMAP 可视化(论文图 4)。

这种格式可以直接用 pandas `read_json(..., lines=True)` 读出来画图。

---

## 引用

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

## 致谢

本工作受到深圳市泛在数据使能重点实验室(ZDSYS20220527171406015)、国家重点研发计划项目(No. 2022ZD0160504)和美团的资助。

联邦指令微调基线参考了 [FedIT (Zhang et al., 2023)](https://github.com/JayZhang42/FederatedGPT-Shepherd)。

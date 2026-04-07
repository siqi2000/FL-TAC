"""FL-TAC server: K-means clustering of received adapters + per-cluster FedAvg.

Implements Algorithm 3 from the paper. The server has NO knowledge of which
client trained which task -- it sees only a flat collection of adapter
state-dicts {(client_i, slot_j) -> v_{i,j}}, runs K-means with K=N (total
number of tasks in the system), then averages within each cluster to get N
new global task-specific adapters.
"""
from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans

from .models import flatten_state, average_states


def cluster_and_aggregate(
    submissions: List[Tuple[int, str, dict, int]],
    n_clusters: int,
    seed: int = 0,
) -> Tuple[Dict[int, dict], List[int], List[str]]:
    """Cluster the submitted adapters and aggregate within clusters.

    Args:
        submissions: list of (client_id, true_task_name, adapter_state, n_samples).
                     `true_task_name` is used ONLY for evaluating clustering
                     accuracy after the fact, NOT by the K-means itself.
        n_clusters:  K (= total number of downstream tasks N).
    Returns:
        cluster_to_state: {cluster_id -> aggregated adapter state}
        labels:           K-means label per submission, in input order
        true_tasks:       list of true task names (parallel to labels)
    """
    if not submissions:
        return {}, [], []

    flats = np.stack([flatten_state(s).numpy() for _, _, s, _ in submissions])
    if len(submissions) <= n_clusters:
        # K-means needs n_samples >= n_clusters
        labels = list(range(len(submissions)))
    else:
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
        labels = km.fit_predict(flats).tolist()

    cluster_to_state: Dict[int, dict] = {}
    for cid in set(labels):
        members = [(submissions[i][2], submissions[i][3])
                   for i, l in enumerate(labels) if l == cid]
        states = [m[0] for m in members]
        sizes = [m[1] for m in members]
        total = sum(sizes) or 1
        weights = [s / total for s in sizes]
        cluster_to_state[cid] = average_states(states, weights)

    true_tasks = [s[1] for s in submissions]
    return cluster_to_state, labels, true_tasks


def assign_global_adapters(
    submissions: List[Tuple[int, str, dict, int]],
    cluster_to_state: Dict[int, dict],
    labels: List[int],
) -> Dict[Tuple[int, str], dict]:
    """For each (client_id, task) submission, return its new aggregated state."""
    out: Dict[Tuple[int, str], dict] = {}
    for (cid, task, _, _), lbl in zip(submissions, labels):
        out[(cid, task)] = cluster_to_state[lbl]
    return out


def clustering_accuracy(labels: List[int], true_tasks: List[str]) -> float:
    """Best-match clustering accuracy via Hungarian assignment."""
    if not labels:
        return 0.0
    from scipy.optimize import linear_sum_assignment
    uniq_tasks = sorted(set(true_tasks))
    task_to_idx = {t: i for i, t in enumerate(uniq_tasks)}
    n_tasks = len(uniq_tasks)
    n_clusters = max(labels) + 1
    K = max(n_tasks, n_clusters)
    cost = np.zeros((K, K), dtype=np.int64)
    for lbl, t in zip(labels, true_tasks):
        cost[lbl, task_to_idx[t]] -= 1
    row, col = linear_sum_assignment(cost)
    matched = -cost[row, col].sum()
    return matched / len(labels)

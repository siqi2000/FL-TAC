"""Inspect (and optionally save) the Dirichlet data partition used by FL-TAC.

Runs the same partition logic as the main training loop -- but with NO model
loading and NO training -- so you can quickly look at how each task's data is
sliced across clients without burning a GPU.

Usage:
    python scripts/inspect_partition.py --scenario glue
    python scripts/inspect_partition.py --scenario cifar  --alpha 0.3
    python scripts/inspect_partition.py --scenario dolly  --n_clients 20 --seed 7
    python scripts/inspect_partition.py --scenario glue   --save partition.csv

The partition is reproducible: identical (scenario, n_clients, alpha, seed)
arguments always yield the same per-client sample counts.
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

# allow running without pip install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fltac.data import (
    dirichlet_partition, GLUE_TASKS, CIFAR_TASKS, DOLLY_TASKS,
)


# Hard-coded train-split sizes (so we don't have to download anything just to
# inspect the partition). These match the HuggingFace versions used in the
# paper. If you change the split, update here.
TASK_SIZES = {
    # GLUE
    "sst2":  67349,
    "mrpc":  3668,
    "qqp":   363846,
    "qnli":  104743,
    "rte":   2490,
    # CIFAR
    "cifar10":  50000,
    "cifar100": 50000,
    # Dolly-15k category counts (from databricks/databricks-dolly-15k)
    "brainstorming":         1773,
    "classification":        2136,
    "closed_qa":             1773,
    "creative_writing":      673,
    "general_qa":            2191,
    "information_extraction": 1506,
    "open_qa":               3742,
    "summarization":         1188,
}

SCENARIOS = {
    "glue":  GLUE_TASKS,
    "cifar": CIFAR_TASKS,
    "dolly": DOLLY_TASKS,
}


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--scenario", choices=list(SCENARIOS), required=True)
    ap.add_argument("--n_clients", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Dirichlet concentration (lower = more skewed)")
    ap.add_argument("--min_frac", type=float, default=0.01,
                    help="clients with proportion below this get no data for that task")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", default=None,
                    help="optional CSV path: rows = clients, cols = tasks")
    args = ap.parse_args()

    tasks = SCENARIOS[args.scenario]
    print(f"\n=== {args.scenario.upper()} partition "
          f"(n_clients={args.n_clients}, alpha={args.alpha}, "
          f"min_frac={args.min_frac}, seed={args.seed}) ===\n")

    table = {t: [0] * args.n_clients for t in tasks}
    for i, task in enumerate(tasks):
        n = TASK_SIZES[task]
        parts = dirichlet_partition(
            n, args.n_clients, alpha=args.alpha,
            min_frac=args.min_frac, seed=args.seed + i,
        )
        for cid, idxs in enumerate(parts):
            table[task][cid] = len(idxs)

    # ---- pretty table ----
    width = max(8, max(len(t) for t in tasks) + 1)
    head = "client".ljust(8) + "".join(t.rjust(width) for t in tasks) + "    total"
    print(head)
    print("-" * len(head))
    totals_per_task = [0] * len(tasks)
    for c in range(args.n_clients):
        row_total = 0
        line = f"#{c:<7}"
        for j, t in enumerate(tasks):
            v = table[t][c]
            totals_per_task[j] += v
            row_total += v
            line += str(v).rjust(width)
        line += f"  {row_total:>7}"
        print(line)
    print("-" * len(head))
    print("total".ljust(8)
          + "".join(str(s).rjust(width) for s in totals_per_task)
          + f"  {sum(totals_per_task):>7}")

    # ---- non-zero clients per task ----
    print()
    for j, t in enumerate(tasks):
        nz = sum(1 for c in range(args.n_clients) if table[t][c] > 0)
        print(f"  {t:25s} : {nz}/{args.n_clients} clients hold data")

    # ---- optional CSV ----
    if args.save:
        with open(args.save, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["client"] + list(tasks) + ["total"])
            for c in range(args.n_clients):
                row = [c] + [table[t][c] for t in tasks]
                w.writerow(row + [sum(row[1:])])
        print(f"\nWrote {args.save}")


if __name__ == "__main__":
    main()

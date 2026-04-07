"""FL-TAC entry point.

Two ways to override config values:

  1. Explicit named flags (most common knobs):
       python main.py --config configs/glue_bert.yaml \
           --rounds 5 --n_clients 4 --lr 1e-4 --lora_rank 8 \
           --method fltac --output_dir results/my_run

  2. Generic --override key=value (for any field):
       python main.py --config configs/glue_bert.yaml \
           --override max_eval_batches=10 dirichlet_alpha=0.3
"""
import argparse, yaml
from fltac.trainer import run


COMMON_FLAGS = [
    # (flag, type, help)
    ("scenario",        str,   "glue | cifar | dolly"),
    ("model_name",      str,   "HuggingFace model id, e.g. bert-base-uncased"),
    ("method",          str,   "fltac (ours) or fedit (baseline)"),
    ("n_clients",       int,   "number of FL clients"),
    ("dirichlet_alpha", float, "Dirichlet partition concentration"),
    ("rounds",          int,   "communication rounds"),
    ("local_steps",     int,   "local SGD steps per (client, task) per round"),
    ("lora_rank",       int,   "LoRA rank r"),
    ("lora_alpha",      int,   "LoRA alpha"),
    ("lora_dropout",    float, "LoRA dropout"),
    ("lr",              float, "learning rate"),
    ("batch_size",      int,   "local batch size"),
    ("max_len",         int,   "max sequence length (text scenarios)"),
    ("eval_every",      int,   "evaluate every N rounds"),
    ("max_eval_batches", int,  "cap eval batches (None = full)"),
    ("seed",            int,   "random seed"),
    ("output_dir",      str,   "where to write metrics + adapters"),
    ("cache_dir",       str,   "HF datasets/models cache dir"),
]


def parse_overrides(overrides):
    out = {}
    for o in overrides or []:
        k, v = o.split("=", 1)
        try:
            v = yaml.safe_load(v)
        except Exception:
            pass
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(
        description="FL-TAC: Federated Learning with Task-Specific Adapter Clustering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config", required=True, help="path to YAML config")
    for name, typ, helpstr in COMMON_FLAGS:
        ap.add_argument(f"--{name}", type=typ, default=None, help=helpstr)
    ap.add_argument("--override", nargs="*", default=[],
                    help="extra key=value overrides for any config field")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # 1) explicit flags
    for name, _, _ in COMMON_FLAGS:
        v = getattr(args, name)
        if v is not None:
            cfg[name] = v
    # 2) generic --override
    cfg.update(parse_overrides(args.override))

    print("=== Effective config ===")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print("========================")
    run(cfg)


if __name__ == "__main__":
    main()

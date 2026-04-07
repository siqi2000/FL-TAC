"""FL-TAC entry point.

Usage:
    python main.py --config configs/glue_bert.yaml
    python main.py --config configs/glue_bert.yaml --override rounds=2 n_clients=4
"""
import argparse, yaml
from fltac.trainer import run


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg.update(parse_overrides(args.override))
    print("=== Config ===")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    run(cfg)


if __name__ == "__main__":
    main()

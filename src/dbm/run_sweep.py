"""Expand and launch the configured experiment grid."""
from __future__ import annotations
import argparse, itertools, logging
from pathlib import Path
import yaml
from dbm.config import load_config
from dbm.train import train
from dbm.logging_utils import setup_logging

logger = logging.getLogger(__name__)

def expand_sweep(path: Path, quick: bool = False) -> list[dict]:
    """Return one merged model config for each Cartesian grid point."""
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    repo = Path(__file__).resolve().parents[2]
    base_path = repo / spec["base_config"]
    base = load_config(base_path)
    grid = spec.get("quick" if quick else "grid", {})
    keys = [key for key, value in grid.items() if isinstance(value, list)]
    fixed = {key: value for key, value in grid.items() if key not in keys}
    return [{**base, **fixed, **dict(zip(keys, values))} for values in itertools.product(*(grid[k] for k in keys))]

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Chạy lưới thí nghiệm E1-E3.")
    parser.add_argument("--config", type=Path, action="append", help="Sweep YAML; có thể lặp lại.")
    parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--quick", action="store_true")
    parser.add_argument("--root", type=Path); parser.add_argument("--work-root", type=Path)
    args = parser.parse_args(argv); setup_logging()
    repo = Path(__file__).resolve().parents[2]
    files = args.config or sorted((repo / "configs/sweeps").glob("e*.yaml"))
    runs = [(path, cfg) for path in files for cfg in expand_sweep(path, args.quick)]
    if not args.config:
        runs.insert(0, (Path("E0 threshold baseline"), {"model_type": "threshold_rule", "mode": "threshold grid search"}))
    for i, (path, cfg) in enumerate(runs, 1):
        logger.info("%03d/%03d %s %s", i, len(runs), path.stem, {k: cfg[k] for k in ("model_type", "window_size", "pooling", "loss", "seed") if k in cfg})
        if not args.dry_run:
            if cfg.get("model_type") == "cnn_lstm":
                raise NotImplementedError("CNN-LSTM dataset training is not integrated in train.py yet.")
            raise NotImplementedError("Sweep execution needs a prepared real dataset; synthetic training is unavailable in this repository.")
    logger.info("Tổng số run: %d", len(runs)); return 0

if __name__ == "__main__": raise SystemExit(main())

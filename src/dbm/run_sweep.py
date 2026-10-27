"""Expand and launch the configured experiment grid."""
from __future__ import annotations
import argparse, itertools, logging
from pathlib import Path
import tempfile
import yaml
from dbm.config import load_config
from dbm.train import train
from dbm.data.synthetic import generate_synthetic_dataset
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
        actual = [(path, cfg) for path, cfg in runs if cfg.get("model_type") != "threshold_rule"]
        if not args.quick:
            raise ValueError("Executing the full sweep is intentionally explicit; use --quick for the synthetic smoke grid.")
        with tempfile.TemporaryDirectory(prefix="dbm_sweep_") as temp_root:
            data_root = Path(temp_root) / "dataset"
            generate_synthetic_dataset(data_root, samples_per_class=3, frames_per_segment=2,
                                       fps=2, size=32, seed=42)
            for i, (sweep_path, cfg) in enumerate(actual, 1):
                base_config = yaml.safe_load(Path(sweep_path).read_text(encoding="utf-8"))["base_config"]
                config_path = repo / base_config
                overrides = [f"{key}={yaml.safe_dump(value, default_flow_style=True).strip()}" for key, value in cfg.items()]
                train(config_path, overrides, data_root, data_root, f"_smoke/sweep_{sweep_path.stem}_{cfg['pooling']}_{i}")
    logger.info("Tổng số run: %d", len(runs)); return 0

if __name__ == "__main__": raise SystemExit(main())

"""Evaluate a saved Geo/landmark LSTM checkpoint on an explicit split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dbm.data.datasets import GeoSequenceDataset, FrameClipDataset
from dbm.labels import num_classes
from dbm.metrics import aggregate_segment_probabilities, classification_metrics
from dbm.models.build import build_lstm, build_cnn_lstm
from dbm.paths import DatasetPaths


def evaluate(checkpoint_path: Path, root: Path | None = None, work_root: Path | None = None,
             split_name: str = "test", batch_size: int = 64) -> dict:
    """Evaluate model and return window- and segment-level metrics."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    run_dir = checkpoint_path.parent
    cfg = checkpoint["config"]
    config_path = run_dir / "config.yaml"
    from dbm.paths import DatasetPaths
    paths = DatasetPaths(dataset_root=root, work_root=work_root)
    repo = Path(__file__).resolve().parents[2]
    stats = json.loads((run_dir / "stats.json").read_text(encoding="utf-8"))
    metadata = pd.read_csv(paths.metadata_csv); splits = pd.read_csv(paths.splits_csv)
    cnn_model = cfg.get("model_type") == "cnn_lstm"
    if cnn_model:
        dataset = FrameClipDataset(metadata, splits, paths.frames, split_name,
                                   cfg["window_size"], cfg["stride"], cfg.get("short_segment_policy", "pad"),
                                   int(cfg.get("input_size", 112)), False, False,
                                   tuple(stats.get("mean", (0, 0, 0))), tuple(stats.get("std", (1, 1, 1))))
    else:
        from dbm.train import _load_feature_config
        feature_config = _load_feature_config(repo, str(cfg["feature_config"]))
        dataset = GeoSequenceDataset(metadata, splits, paths.landmarks, split_name, feature_config,
                                     cfg["window_size"], cfg["stride"], cfg.get("short_segment_policy", "pad"), stats)
    if not dataset:
        raise ValueError(f"No windows found for split {split_name!r}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = build_cnn_lstm(cfg) if cnn_model else build_lstm(cfg, int(checkpoint["input_dim"]))
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device).eval()
    ys=[]; probabilities=[]; ids=[]
    with torch.inference_mode():
        for x, y, segment_ids, _ in loader:
            prob = model(x.to(device)).softmax(-1).cpu().numpy()
            probabilities.extend(prob); ys.extend(y.numpy().tolist()); ids.extend(segment_ids)
    result = {"window": classification_metrics(ys, [p.argmax() for p in probabilities], num_classes())}
    seg_ids, seg_prob, seg_y = aggregate_segment_probabilities(ids, probabilities, ys)
    result["segment"] = classification_metrics(seg_y, seg_prob.argmax(1), num_classes())
    result["segment_ids"] = seg_ids
    result["synthetic"] = bool(cfg.get("synthetic", False))
    (run_dir / f"{split_name}_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Đánh giá checkpoint trên một split.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Đường dẫn best.pt.")
    parser.add_argument("--root", type=Path, default=None); parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    args=parser.parse_args(argv)
    try:
        result=evaluate(args.checkpoint,args.root,args.work_root,args.split)
        print(json.dumps(result, indent=2)); return 0
    except Exception as exc:
        parser.exit(1, f"Lỗi đánh giá: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())


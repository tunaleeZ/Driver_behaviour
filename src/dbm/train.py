"""Train Geo-LSTM or Landmark-LSTM from precomputed features."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from dbm.baselines.threshold_rule import grid_search_thresholds
from dbm.config import load_config, save_config
from dbm.data.datasets import GeoSequenceDataset, compose_features
from dbm.data.stats import fit_feature_stats, save_stats
from dbm.labels import label_to_index, num_classes
from dbm.logging_utils import setup_logging
from dbm.losses import build_loss, class_weights_from_labels
from dbm.metrics import aggregate_segment_probabilities, classification_metrics
from dbm.models.build import build_lstm
from dbm.paths import DatasetPaths
from dbm.seed import set_global_seed, worker_init_fn

logger = logging.getLogger(__name__)


def _load_feature_config(repo: Path, variant: str) -> dict:
    cfg = yaml.safe_load((repo / "configs" / "features.yaml").read_text(encoding="utf-8"))
    selected = cfg["feature_configs"][variant]
    selected = {**selected, "normalization_origin": cfg.get("normalization_origin", "nose"),
                "normalization_scale": cfg.get("normalization_scale", "inter_ocular"),
                "normalize_roll": bool(cfg.get("normalize_roll", False)),
                "selected_landmarks": cfg.get("selected_landmarks", [])}
    return selected


def _train_stats(metadata: pd.DataFrame, splits: pd.DataFrame, feature_root: Path, feature_config: dict) -> dict:
    mapping = dict(zip(splits.subject_id.astype(str), splits.split.astype(str)))
    train_meta = metadata[(metadata.quality.astype(str).str.lower() != "bad") &
                          (metadata.subject_id.astype(str).map(mapping) == "train")]
    seqs = []
    for sid in train_meta.segment_id.astype(str):
        with np.load(feature_root / f"{sid}.npz") as archive:
            seqs.append(compose_features({k: archive[k] for k in archive.files}, feature_config))
    return fit_feature_stats(seqs)


def _run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=False, clip=1.0):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0; count = 0; ys = []; preds = []; probs_all = []; ids_all = []
    for x, y, segment_ids, _padded in loader:
        x = x.to(device); y = y.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)
        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), clip); optimizer.step()
        batch = y.size(0); loss_sum += float(loss.detach()) * batch; count += batch
        prob = logits.detach().softmax(-1).cpu().numpy()
        ys.extend(y.cpu().tolist()); preds.extend(prob.argmax(-1).tolist()); probs_all.extend(prob); ids_all.extend(segment_ids)
    return loss_sum / max(count, 1), ys, preds, ids_all, np.asarray(probs_all)


def train(config_path: Path, overrides: list[str] | None = None, root: Path | None = None,
          work_root: Path | None = None, run_name: str | None = None) -> Path:
    """Run model selection on validation and evaluate test once with best checkpoint."""
    repo = Path(__file__).resolve().parents[2]
    config = load_config(config_path, overrides)
    set_global_seed(int(config.get("seed", 42)))
    paths = DatasetPaths(dataset_root=root, work_root=work_root)
    metadata = pd.read_csv(paths.metadata_csv)
    splits = pd.read_csv(paths.splits_csv)
    feature_config = _load_feature_config(repo, str(config.get("feature_config", "geo")))
    stats = _train_stats(metadata, splits, paths.landmarks, feature_config)
    train_ds = GeoSequenceDataset(metadata, splits, paths.landmarks, "train", feature_config,
                                  config["window_size"], config["stride"], config.get("short_segment_policy", "pad"), stats)
    val_ds = GeoSequenceDataset(metadata, splits, paths.landmarks, "val", feature_config,
                                config["window_size"], config["stride"], config.get("short_segment_policy", "pad"), stats)
    test_ds = GeoSequenceDataset(metadata, splits, paths.landmarks, "test", feature_config,
                                 config["window_size"], config["stride"], config.get("short_segment_policy", "pad"), stats)
    if not train_ds or not val_ds:
        raise ValueError("Train and validation datasets must contain at least one window")
    train_loader = DataLoader(train_ds, batch_size=int(config["batch_size"]), shuffle=True,
                              num_workers=int(config.get("num_workers", 0)), worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=int(config["batch_size"]), shuffle=False,
                            num_workers=int(config.get("num_workers", 0)), worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_ds, batch_size=int(config["batch_size"]), shuffle=False,
                             num_workers=int(config.get("num_workers", 0)), worker_init_fn=worker_init_fn) if len(test_ds) else None
    feature_dim = train_ds[0][0].shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_lstm(config, feature_dim).to(device)
    labels = [window.label for window in train_ds.windows]
    weights = class_weights_from_labels(labels, num_classes(), config.get("loss_weight_cap")).to(device)
    criterion = build_loss(str(config.get("loss", "cross_entropy")), weights,
                           float(config.get("focal_gamma", 2.0)), cost_lambda=float(config.get("cost_sensitive_lambda", .5)))
    if config.get("loss") == "cost_sensitive":
        cost_raw = yaml.safe_load((repo / "configs" / "cost_matrix.yaml").read_text(encoding="utf-8"))["cost_matrix"]
        names = yaml.safe_load((repo / "configs" / "labels.yaml").read_text(encoding="utf-8"))["labels"]
        order = [item["name"] for item in sorted(names, key=lambda item: item["index"])]
        criterion = build_loss("cost_sensitive", weights, cost_matrix=torch.tensor([cost_raw[name] for name in order]),
                               cost_lambda=float(config.get("cost_sensitive_lambda", .5))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config.get("weight_decay", 0)))
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=int(config.get("scheduler_patience", 5)),
                factor=float(config.get("scheduler_factor", .5))) if config.get("scheduler") == "reduce_on_plateau" else
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["max_epochs"])))
    amp = bool(config.get("use_amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    name = run_name or f"{config_path.stem}_{pd.Timestamp.utcnow():%Y%m%d_%H%M%S}"
    out = paths.outputs / name; out.mkdir(parents=True, exist_ok=True)
    save_config(config, out / "config.yaml"); save_stats(stats, out / "stats.json")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    (out / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    best = -1.0; stale = 0; history = []
    for epoch in range(1, int(config["max_epochs"]) + 1):
        train_loss, *_ = _run_epoch(model, train_loader, criterion, device, optimizer, scaler, amp,
                                    float(config.get("gradient_clip", 1.0)))
        val_loss, y, _pred, _ids, probs = _run_epoch(model, val_loader, criterion, device)
        val_metrics = classification_metrics(y, probs.argmax(1), num_classes())
        score = val_metrics["macro_f1"]
        scheduler.step(val_loss if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else epoch)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_macro_f1": score})
        logger.info("epoch=%d train_loss=%.4f val_loss=%.4f val_macro_f1=%.4f", epoch, train_loss, val_loss, score)
        torch.save({"model": model.state_dict(), "config": config, "input_dim": feature_dim, "epoch": epoch}, out / "last.pt")
        if score > best:
            best = score; stale = 0
            torch.save({"model": model.state_dict(), "config": config, "input_dim": feature_dim, "epoch": epoch}, out / "best.pt")
        else:
            stale += 1
        if stale >= int(config.get("early_stopping_patience", 10)):
            break
    with (out / "epochs.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    _save_learning_curves(out, history)
    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    result = {}
    if test_loader:
        _loss, y, _pred, ids, probs = _run_epoch(model, test_loader, criterion, device)
        result["window"] = classification_metrics(y, probs.argmax(1), num_classes())
        seg_ids, seg_probs, seg_y = aggregate_segment_probabilities(ids, probs, y)
        result["segment"] = classification_metrics(seg_y, seg_probs.argmax(1), num_classes())
        result["segment_ids"] = seg_ids
    (out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _save_confusion(out, result)
    return out


def _save_confusion(out: Path, result: dict) -> None:
    """Save confusion matrices as PNG when matplotlib is available."""
    if "window" not in result:
        return
    try:
        import matplotlib.pyplot as plt
        matrix = np.asarray(result["window"]["confusion_matrix"])
        normalized = np.asarray(result["window"]["confusion_matrix_normalized"])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for ax, values, title, fmt in ((axes[0], matrix, "Counts", "d"),
                                       (axes[1], normalized, "Row normalized", ".2f")):
            ax.imshow(values, cmap="Blues", vmin=0)
            ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            for row in range(values.shape[0]):
                for col in range(values.shape[1]):
                    ax.text(col, row, format(values[row, col], fmt), ha="center", va="center", fontsize=7)
        fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=150); plt.close(fig)
    except ImportError:
        logger.warning("matplotlib unavailable; confusion_matrix.png not created")


def _save_learning_curves(out: Path, history: list[dict]) -> None:
    """Save loss and validation macro-F1 curves when matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
        epochs = [row["epoch"] for row in history]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
        axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
        axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
        axes[1].plot(epochs, [row["val_macro_f1"] for row in history], label="validation macro-F1")
        axes[1].set_title("Validation macro-F1"); axes[1].set_xlabel("Epoch"); axes[1].set_ylim(0, 1)
        fig.tight_layout(); fig.savefig(out / "learning_curves.png", dpi=150); plt.close(fig)
    except ImportError:
        logger.warning("matplotlib unavailable; learning_curves.png not created")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Huấn luyện Geo-LSTM hoặc Landmark-LSTM.")
    parser.add_argument("--config", type=Path, required=True, help="Đường dẫn YAML cấu hình model.")
    parser.add_argument("--override", action="append", default=[], help="Ghi đè key=value (có thể lặp).")
    parser.add_argument("--root", type=Path, default=None, help="Dataset root.")
    parser.add_argument("--work-root", type=Path, default=None, help="Work root ghi đặc trưng/output.")
    parser.add_argument("--run-name", default=None, help="Tên thư mục run.")
    args = parser.parse_args(argv); setup_logging()
    try:
        out = train(args.config, args.override, args.root, args.work_root, args.run_name)
        logger.info("Run outputs: %s", out); return 0
    except Exception as exc:
        logger.exception("Training failed: %s", exc); return 1


if __name__ == "__main__":
    raise SystemExit(main())


"""Run sliding-window inference for sequence and frame models."""
from __future__ import annotations

import argparse
import csv
import logging
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from dbm.data.transforms import resize_frame
from dbm.features.mediapipe_extractor import MediaPipeFaceExtractor
from dbm.labels import index_to_label
from dbm.logging_utils import setup_logging
from dbm.models.build import build_cnn_lstm, build_lstm

logger = logging.getLogger(__name__)


def smooth_probabilities(probs: np.ndarray, method: str = "ema", span: int = 3) -> np.ndarray:
    """Smooth probability rows using EMA or a trailing majority vote."""
    if span <= 0:
        raise ValueError("span must be positive")
    if method == "vote":
        ids = probs.argmax(axis=1)
        return np.stack([np.bincount(ids[max(0, i - span + 1):i + 1], minlength=probs.shape[1]) /
                         min(span, i + 1) for i in range(len(ids))])
    if method != "ema":
        raise ValueError("method must be 'ema' or 'vote'")
    result = []
    alpha = 2 / (span + 1)
    for row in probs:
        result.append(row.copy() if not result else alpha * row + (1 - alpha) * result[-1])
    return np.asarray(result)


def _read_video_frames(video: Path, input_size: int, target_fps: float | None = None) -> tuple[np.ndarray, float]:
    """Decode RGB frames and apply the same letterbox/center-crop eval transform."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    fps = float(target_fps or source_fps)
    frames = []
    source_index = 0
    next_time = 0.0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            timestamp = source_index / source_fps
            source_index += 1
            if timestamp + 1e-9 < next_time:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            stored = resize_frame(rgb, (128, 128), "letterbox")
            top = (128 - input_size) // 2
            frames.append(stored[top:top + input_size, top:top + input_size])
            next_time += 1.0 / fps
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video}")
    return np.stack(frames), fps


def infer_video(video: Path, checkpoint_path: Path, output: Path, extractor=None,
                window_size: int | None = None, stride: int | None = None,
                smoothing: str = "ema", span: int = 3) -> Path:
    """Predict each sliding window and write a labeled timeline CSV."""
    video, checkpoint_path, output = Path(video), Path(checkpoint_path), Path(output)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    run_dir = checkpoint_path.parent
    stats = json.loads((run_dir / "stats.json").read_text(encoding="utf-8"))
    model_type = config.get("model_type", "lstm_classifier")
    steps = int(window_size or config.get("window_size", 30))
    stride = int(stride or config.get("stride", max(1, steps // 2)))
    if steps <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")

    if model_type == "cnn_lstm":
        input_size = int(config.get("input_size", 112))
        if extractor is not None and hasattr(extractor, "extract_frames_from_video"):
            frames, fps = extractor.extract_frames_from_video(video, input_size=input_size,
                                                               target_fps=config.get("target_fps"))
        else:
            frames, fps = _read_video_frames(video, input_size, config.get("target_fps"))
        frames = np.asarray(frames)
        clips, starts = [], []
        for start in range(0, max(1, len(frames) - steps + 1), stride):
            clip = frames[start:start + steps]
            if len(clip) < steps:
                clip = np.concatenate((clip, np.repeat(clip[-1:], steps - len(clip), axis=0)))
            clips.append(torch.from_numpy(clip.copy()).permute(0, 3, 1, 2).float() / 255.0)
            starts.append(start)
        model = build_cnn_lstm(config)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        mean = torch.tensor(stats.get("mean", (0.0, 0.0, 0.0)), dtype=torch.float32)[None, :, None, None]
        std = torch.tensor(stats.get("std", (1.0, 1.0, 1.0)), dtype=torch.float32)[None, :, None, None].clamp_min(1e-6)
        clips = [(clip - mean) / std for clip in clips]
        with torch.inference_mode():
            probabilities = torch.cat([model(clip.unsqueeze(0)).softmax(-1) for clip in clips]).numpy()
        total_steps = len(frames)
    else:
        extractor = extractor or MediaPipeFaceExtractor()
        data = extractor.extract_from_video(video, target_fps=config.get("feature_fps", 30))
        from dbm.train import _load_feature_config
        from dbm.data.datasets import compose_features
        from dbm.data.stats import normalize_features
        repo = Path(__file__).resolve().parents[2]
        feature_config = _load_feature_config(repo, config.get("feature_config", "geo"))
        values = normalize_features(compose_features(data, feature_config), stats)
        windows, starts = [], []
        for start in range(0, max(1, len(values) - steps + 1), stride):
            chunk = values[start:start + steps]
            if len(chunk) < steps:
                chunk = np.pad(chunk, ((0, steps - len(chunk)), (0, 0)), mode="edge")
            windows.append(torch.from_numpy(chunk[None].astype(np.float32)))
            starts.append(start)
        model = build_lstm(config, int(checkpoint["input_dim"]))
        model.load_state_dict(checkpoint["model"])
        model.eval()
        with torch.inference_mode():
            probabilities = torch.cat([model(window).softmax(-1) for window in windows]).numpy()
        fps = float(data.get("fps", 30.0))
        total_steps = len(values)

    smoothed = smooth_probabilities(probabilities, smoothing, span)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["window_index", "start_seconds", "end_seconds", "prediction", "probabilities", "synthetic"])
        for index, (start, probability) in enumerate(zip(starts, smoothed)):
            end = min(start + steps, total_steps)
            writer.writerow([index, start / fps, end / fps, index_to_label(int(probability.argmax())),
                             " ".join(str(float(item)) for item in probability), bool(config.get("synthetic", False))])
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Suy luận trượt trên video bằng Geo, Landmark hoặc CNN-LSTM.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoothing", choices=("ema", "vote"), default="ema")
    parser.add_argument("--span", type=int, default=3)
    args = parser.parse_args(argv)
    setup_logging()
    try:
        infer_video(args.video, args.checkpoint, args.output, smoothing=args.smoothing, span=args.span)
        return 0
    except Exception as exc:
        logger.exception("Inference failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

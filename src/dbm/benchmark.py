"""Benchmark model forward latency on synthetic tensors without reporting claims."""
from __future__ import annotations
import argparse, logging, time
import sys
from pathlib import Path
import numpy as np
import torch
from dbm.models.build import build_lstm, build_cnn_lstm
from dbm.data.datasets import compose_features
from dbm.paths import DatasetPaths
from dbm.logging_utils import setup_logging

logger=logging.getLogger(__name__)

def benchmark_extraction(video: Path, extractor, feature_config: dict,
                        target_fps: float | None = 30.0) -> dict:
    """Time landmark extraction and feature composition without persisting results."""
    start = time.perf_counter()
    payload = extractor.extract_from_video(video, target_fps=target_fps)
    extraction_ms = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    features = compose_features(payload, feature_config)
    feature_ms = (time.perf_counter() - start) * 1000.0
    return {"landmark_extraction_ms": extraction_ms, "feature_composition_ms": feature_ms,
            "frames": len(features), "feature_dim": int(features.shape[1]),
            "synthetic": bool(np.asarray(payload.get("synthetic", False)).item())}

def benchmark_extraction_if_available(video: Path | None, model_path: Path,
                                      feature_config: dict, extractor=None) -> dict | None:
    """Run extraction timing only when both a model file and real video exist."""
    if not model_path.is_file():
        logger.warning("Bỏ qua benchmark landmark: thiếu face_landmarker.task tại %s", model_path)
        return None
    if video is None or not Path(video).is_file():
        logger.warning("Bỏ qua benchmark landmark: thiếu đường dẫn video thật hợp lệ.")
        return None
    if extractor is None:
        from dbm.features.mediapipe_extractor import MediaPipeFaceExtractor
        extractor = MediaPipeFaceExtractor(model_path=model_path)
    return benchmark_extraction(Path(video), extractor, feature_config)
def benchmark_model(config: dict, steps: int=30, warmup: int=5, device: str|None=None) -> dict:
    """Measure local tensor inference; returned values are diagnostic only."""
    dev=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_type=config.get("model_type", "lstm")
    model=build_cnn_lstm(config) if model_type=="cnn_lstm" else build_lstm(config, int(config.get("input_dim",4)))
    model.to(dev).eval(); t=int(config.get("window_size",16)); batch=int(config.get("batch_size",1))
    shape=(batch,t,3,int(config.get("input_size",112)),int(config.get("input_size",112))) if model_type=="cnn_lstm" else (batch,t,int(config.get("input_dim",4)))
    x=torch.rand(shape,device=dev)
    with torch.inference_mode():
        for _ in range(warmup): model(x)
        if dev.type=="cuda": torch.cuda.synchronize()
        times=[]
        for _ in range(steps):
            start=time.perf_counter(); model(x)
            if dev.type=="cuda": torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000)
    return {"device":str(dev),"latency_ms_mean":float(np.mean(times)),"latency_ms_p95":float(np.percentile(times,95)),"steps":steps,"synthetic":True}

def main(argv=None):
    # Keep Vietnamese help readable in Windows consoles using legacy code pages.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    p=argparse.ArgumentParser(description="Benchmark forward model bằng tensor tổng hợp.")
    p.add_argument("--config",type=Path,required=True); p.add_argument("--steps",type=int,default=30); p.add_argument("--warmup",type=int,default=5)
    p.add_argument("--video",type=Path); p.add_argument("--checkpoint",type=Path)
    a=p.parse_args(argv); setup_logging()
    import yaml
    cfg=yaml.safe_load(a.config.read_text(encoding="utf-8")); print(benchmark_model(cfg,a.steps,a.warmup))
    if a.video or a.checkpoint:
        from dbm.train import _load_feature_config
        paths=DatasetPaths()
        repo=Path(__file__).resolve().parents[2]
        feature_config=_load_feature_config(repo, cfg.get("feature_config", "geo"))
        result=benchmark_extraction_if_available(a.video, paths.face_landmarker, feature_config)
        if result is not None: print(result)
    return 0
if __name__=="__main__": raise SystemExit(main())

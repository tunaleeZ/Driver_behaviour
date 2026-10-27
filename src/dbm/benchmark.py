"""Benchmark model forward latency on synthetic tensors without reporting claims."""
from __future__ import annotations
import argparse, logging, time, warnings
from pathlib import Path
import numpy as np
import torch
from dbm.models.build import build_lstm, build_cnn_lstm
from dbm.paths import DatasetPaths
from dbm.logging_utils import setup_logging

logger=logging.getLogger(__name__)
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
    p=argparse.ArgumentParser(description="Benchmark forward model bằng tensor tổng hợp.")
    p.add_argument("--config",type=Path,required=True); p.add_argument("--steps",type=int,default=30); p.add_argument("--warmup",type=int,default=5)
    p.add_argument("--video",type=Path); p.add_argument("--checkpoint",type=Path)
    a=p.parse_args(argv); setup_logging()
    if a.video or a.checkpoint:
        paths=DatasetPaths()
        if not paths.face_landmarker.is_file() or not a.video or not a.video.is_file(): logger.warning("Bỏ qua benchmark trích landmark: cần face_landmarker.task và video thật hợp lệ.")
        else: logger.warning("Benchmark trích landmark chưa được bật trong chế độ tensor-only.")
    import yaml
    cfg=yaml.safe_load(a.config.read_text(encoding="utf-8")); print(benchmark_model(cfg,a.steps,a.warmup)); return 0
if __name__=="__main__": raise SystemExit(main())

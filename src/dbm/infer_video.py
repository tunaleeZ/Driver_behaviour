"""Run sliding-window video inference with a supplied feature extractor."""
from __future__ import annotations
import argparse, csv, logging
from pathlib import Path
import cv2
import numpy as np
import torch
from dbm.features.mediapipe_extractor import MediaPipeFaceExtractor
from dbm.models.build import build_lstm
from dbm.paths import DatasetPaths
from dbm.logging_utils import setup_logging
from dbm.labels import num_classes

logger=logging.getLogger(__name__)
def smooth_probabilities(probs: np.ndarray, method="ema", span=3) -> np.ndarray:
    """Smooth probabilities using trailing majority votes or EMA."""
    if method=="vote":
        ids=probs.argmax(1); return np.stack([np.bincount(ids[max(0,i-span+1):i+1],minlength=probs.shape[1])/min(span,i+1) for i in range(len(ids))])
    out=[]; alpha=2/(span+1)
    for row in probs: out.append(row.copy() if not out else alpha*row+(1-alpha)*out[-1])
    return np.asarray(out)

def infer_video(video: Path, checkpoint_path: Path, output: Path, extractor=None, window_size=None, stride=None, smoothing="ema", span=3):
    """Extract features, predict each window and write a timeline CSV."""
    ckpt=torch.load(checkpoint_path,map_location="cpu",weights_only=False); cfg=ckpt["config"]
    extractor=extractor or MediaPipeFaceExtractor()
    data=extractor.extract_from_video(video,target_fps=cfg.get("feature_fps",30))
    from dbm.train import _load_feature_config
    from dbm.data.datasets import compose_features
    from dbm.data.stats import normalize_features
    repo=Path(__file__).resolve().parents[2]; fc=_load_feature_config(repo,cfg.get("feature_config","geo"))
    values=compose_features(data,fc); stats=__import__("json").loads((checkpoint_path.parent/"stats.json").read_text())
    values=normalize_features(values,stats); t=int(window_size or cfg.get("window_size",30)); step=int(stride or cfg.get("stride",t//2))
    windows=[values[i:i+t] for i in range(0,max(1,len(values)-t+1),step)]
    if not windows: windows=[np.pad(values,((0,max(0,t-len(values))),(0,0)),mode="edge")]
    model=build_lstm(cfg,int(ckpt["input_dim"])); model.load_state_dict(ckpt["model"]); model.eval()
    with torch.inference_mode(): probs=torch.cat([model(torch.from_numpy(w[None].astype("float32"))).softmax(-1) for w in windows]).numpy()
    smooth=smooth_probabilities(probs,smoothing,span); names={v:k for k,v in __import__("dbm.labels",fromlist=["label_to_index"]).LABEL_TO_INDEX.items()} if hasattr(__import__("dbm.labels",fromlist=["LABEL_TO_INDEX"]),"LABEL_TO_INDEX") else None
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["window_index","start_seconds","end_seconds","prediction","probabilities"])
        fps=float(data.get("fps",30));
        for i,p in enumerate(smooth): w.writerow([i,i*step/fps,min(i*step+t,len(values))/fps,int(p.argmax())," ".join(map(str,p.tolist()))])
    return output

def main(argv=None):
    p=argparse.ArgumentParser(description="Suy luận trượt trên video (Geo/Landmark-LSTM).")
    p.add_argument("--video",type=Path,required=True); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--smoothing",choices=("ema","vote"),default="ema"); p.add_argument("--span",type=int,default=3)
    a=p.parse_args(argv); setup_logging()
    try: infer_video(a.video,a.checkpoint,a.output,smoothing=a.smoothing,span=a.span); return 0
    except Exception as e: logger.exception("Inference failed: %s",e); return 1
if __name__=="__main__": raise SystemExit(main())

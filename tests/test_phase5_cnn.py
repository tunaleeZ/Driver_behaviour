from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import torch

from dbm.data.datasets import FrameClipDataset
from dbm.models.cnn_lstm import CNNLSTM
from dbm.models.pooling import TemporalPool


def test_cnn_lstm_report_shapes_and_first_conv_parameter_count():
    model = CNNLSTM().eval()
    with torch.no_grad():
        logits = model(torch.zeros(8, 16, 3, 112, 112))
    assert logits.shape == (8, 5)
    assert model.backbone.features[0].weight.numel() + model.backbone.features[0].bias.numel() == 896


def test_temporal_pooling_outputs_and_attention_sum():
    sequence = torch.randn(3, 7, 128)
    for mode in ("last", "mean", "attention"):
        pooled, alpha = TemporalPool(mode, 128)(sequence)
        assert pooled.shape == (3, 128)
        if mode == "attention":
            torch.testing.assert_close(alpha.sum(dim=1), torch.ones(3))


def test_frame_clip_dataset_padding_eval_and_consistent_augmentation(tmp_path, monkeypatch):
    frames = tmp_path / "frames" / "SEG000001"
    frames.mkdir(parents=True)
    for index in range(3):
        image = np.full((128, 128, 3), 40, dtype=np.uint8)
        assert cv2.imwrite(str(frames / f"{index + 1:06}.jpg"), image)
    metadata = pd.DataFrame([{"segment_id": "SEG000001", "subject_id": "S001",
                              "quality": "good", "label": "normal"}])
    splits = pd.DataFrame([{"subject_id": "S001", "split": "train"}])
    dataset = FrameClipDataset(metadata, splits, tmp_path / "frames", "train", window_size=5,
                               stride=2, input_size=112, training=False)
    clip, label, sid, padded = dataset[0]
    assert clip.shape == (5, 3, 112, 112)
    assert label.item() == 0 and sid == "SEG000001" and padded
    torch.testing.assert_close(clip[-1], clip[-2])

    augmented = FrameClipDataset(metadata, splits, tmp_path / "frames", "train", window_size=3,
                                 stride=1, input_size=112, training=True)
    monkeypatch.setattr(np.random, "randint", lambda low, high=None: low)
    monkeypatch.setattr(np.random, "uniform", lambda low, high: 0.0 if low < 0 else 1.0)
    clip, *_ = augmented[0]
    assert clip.shape == (3, 3, 112, 112)
    # Identical source frames remain identical because every sampled transform is clip-shared.
    torch.testing.assert_close(clip[0], clip[1])


def test_cnn_lstm_cpu_training_step():
    model = CNNLSTM(channels=(4, 8), hidden_size=8, head_dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    clips = torch.rand(2, 3, 3, 32, 32)
    labels = torch.tensor([0, 1])
    model.train()
    loss = torch.nn.functional.cross_entropy(model(clips), labels)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)

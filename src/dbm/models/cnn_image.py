"""Frame-wise CNN classifier for image datasets and video inference."""
from torch import nn
from dbm.models.cnn_backbone import CNNBackbone

class CNNImageClassifier(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256), n_classes=5, dropout=0.3):
        super().__init__()
        self.backbone = CNNBackbone(channels)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.output_dim, n_classes))

    def forward(self, images):
        return self.classifier(self.backbone(images))

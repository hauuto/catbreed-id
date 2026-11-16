"""Notebook-compatible model definitions for deployment checkpoints."""

import torch
import torch.nn as nn
from torchvision import models

CAT_BREEDS = [
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British_Shorthair",
    "Egyptian_Mau",
    "Maine_Coon",
    "Persian",
    "Ragdoll",
    "Russian_Blue",
    "Siamese",
    "Sphynx",
]
CLASS_TO_IDX = {name: index for index, name in enumerate(CAT_BREEDS)}
MODEL_NAMES = ["resnet50", "mobilenet_v2", "efficientnet_b0", "cnn_baseline"]

BACKBONES = {
    "resnet50": models.resnet50,
    "mobilenet_v2": models.mobilenet_v2,
    "efficientnet_b0": models.efficientnet_b0,
}


def baseline_features(batchnorm=False):
    blocks, channels_in = [], 3
    for channels in (32, 64, 128):
        norm = [nn.BatchNorm2d(channels)] if batchnorm else []
        blocks.append(
            nn.Sequential(
                nn.Conv2d(channels_in, channels, 3, padding=1),
                *norm,
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
        )
        channels_in = channels
    return nn.Sequential(*blocks), channels_in


class CatClassifier(nn.Module):
    def __init__(self, name, pretrained=True, dropout=0.3, head_hidden=0, baseline_batchnorm=False):
        super().__init__()
        self.name = name
        if name == "cnn_baseline":
            self.features, dimension = baseline_features(baseline_batchnorm)
        else:
            net = BACKBONES[name](weights=None if not pretrained else None)
            if name == "resnet50":
                self.features, dimension = nn.Sequential(*list(net.children())[:-2]), net.fc.in_features
            else:
                self.features, dimension = net.features, net.classifier[-1].in_features
        self.dimension = dimension
        self.pool = nn.AdaptiveAvgPool2d(1)
        if head_hidden:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(dimension, head_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, len(CAT_BREEDS)),
            )
        else:
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dimension, len(CAT_BREEDS)))

    def forward_features(self, x):
        return self.pool(self.features(x)).flatten(1)

    def forward(self, x):
        return self.head(self.forward_features(x))


def build_from_checkpoint(saved, device="cpu"):
    mapping = saved.get("class_mapping")
    if not isinstance(mapping, dict) or len(mapping) != len(CAT_BREEDS):
        raise RuntimeError("Class mapping không hợp lệ.")
    if set(mapping.values()) != set(range(len(CAT_BREEDS))):
        raise RuntimeError("Class mapping phải có chỉ số từ 0 đến 11.")
    model = CatClassifier(
        saved["model_name"],
        pretrained=False,
        dropout=saved["config"]["dropout"],
        head_hidden=saved["config"].get("head_hidden", 0),
        baseline_batchnorm=saved["config"].get("baseline_batchnorm", False),
    ).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    return model

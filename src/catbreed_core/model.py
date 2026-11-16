"""Mô hình phân loại, chép nguyên từ notebooks/kaggle_runner.ipynb (mục 9).

Khác biệt so với notebook (chỉ phần bọc tối thiểu):
- CONFIG ở đây chỉ chứa các khóa mà CatClassifier đọc, giá trị bằng cấu hình mặc định của notebook.
- Không chép ensure_pretrained_weights (phụ thuộc file_hash, progress, CONFIG mạng của notebook);
  nhánh pretrained=True bỏ lời gọi này, torchvision tự tải trọng số. Suy luận luôn dùng pretrained=False.
- Thêm build_from_checkpoint, cùng nội dung với phần dựng mô hình của reload_checkpoint.
"""
import torch
import torch.nn as nn
from torchvision import models

CONFIG = {
    "dropout": 0.3,
    "head_hidden": 0,
    "baseline_batchnorm": False,
    "fine_tune_fraction": None,
    "fine_tune_blocks": 2,
}

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


# ─── 9.1 Trọng số ImageNet và khối tích chập của CNN baseline ────────────────────────────────────────
BACKBONES = {
    "resnet50": models.resnet50,
    "mobilenet_v2": models.mobilenet_v2,
    "efficientnet_b0": models.efficientnet_b0,
}


PRETRAINED_WEIGHTS = {  # Giữ IMAGENET1K_V1; DEFAULT của ResNet50/MobileNetV2 là V2
    "resnet50": models.ResNet50_Weights.IMAGENET1K_V1,
    "mobilenet_v2": models.MobileNet_V2_Weights.IMAGENET1K_V1,
    "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
}


def baseline_features(batchnorm=None):
    batchnorm = CONFIG["baseline_batchnorm"] if batchnorm is None else batchnorm
    # §2.5.1: ba khối Conv3×3 + ReLU + MaxPool 32/64/128 kênh, không BatchNorm
    blocks, channels_in = [], 3
    for channels in [32, 64, 128]:
        norm = [nn.BatchNorm2d(channels)] if batchnorm else []
        blocks.append(
            nn.Sequential(nn.Conv2d(channels_in, channels, 3, padding=1), *norm, nn.ReLU(), nn.MaxPool2d(2))
        )
        channels_in = channels
    return nn.Sequential(*blocks), channels_in


# ─── 9.2 Mô hình phân loại: backbone, GAP, Dropout, Linear ───────────────────────────────────────────
class CatClassifier(nn.Module):
    def __init__(self, name, pretrained=True, dropout=None, head_hidden=None, baseline_batchnorm=None):
        super().__init__()
        self.name = name
        if name == "cnn_baseline":
            self.features, dimension = baseline_features(baseline_batchnorm)
        else:
            net = BACKBONES[name](weights=PRETRAINED_WEIGHTS[name] if pretrained else None)
            if name == "resnet50":
                self.features = nn.Sequential(*list(net.children())[:-2])
                dimension = net.fc.in_features
            else:
                self.features, dimension = net.features, net.classifier[-1].in_features
        self.pool = nn.AdaptiveAvgPool2d(1)
        rate = CONFIG["dropout"] if dropout is None else dropout
        hidden = CONFIG["head_hidden"] if head_hidden is None else head_hidden
        if hidden:
            self.head = nn.Sequential(
                nn.Dropout(rate),
                nn.Linear(dimension, hidden),
                nn.ReLU(),
                nn.Dropout(rate),
                nn.Linear(hidden, len(CAT_BREEDS)),
            )
        else:
            self.head = nn.Sequential(nn.Dropout(rate), nn.Linear(dimension, len(CAT_BREEDS)))
        self.tune_indices, self.tune_units = [], []

    def forward_features(self, x):
        return self.pool(self.features(x)).flatten(1)

    def forward(self, x):
        return self.head(self.forward_features(x))

    def configure_stage(self, stage):
        for parameter in self.features.parameters():
            parameter.requires_grad = stage == "scratch"
        for parameter in self.head.parameters():
            parameter.requires_grad = True
        self.tune_units = self.unfrozen_units() if stage == "finetuned" else []
        self.tune_indices = [name for name, _ in self.tune_units]  # ghi vào metrics.json
        for _, module in self.tune_units:
            for parameter in module.parameters():
                parameter.requires_grad = True

    def feature_units(self):
        # Đơn vị mở khóa: tách các nhóm nn.Sequential thuần (layer1–4 của ResNet50, các stage của
        # EfficientNet-B0) thành từng khối con (Bottleneck, MBConv); InvertedResidual của MobileNetV2 và
        # Conv2dNormActivation (lớp con của Sequential) giữ nguyên là một đơn vị
        units = []
        for index, block in enumerate(self.features):
            if type(block) is nn.Sequential:
                units += [(f"{index}.{child}", module) for child, module in block.named_children()]
            else:
                units.append((str(index), block))
        return units

    def unfrozen_units(self):
        # fine_tune_fraction = None: giữ cách cũ, mở fine_tune_blocks phần tử cuối của features (§2.5.2).
        # Có fine_tune_fraction: mở số khối con cuối có tỷ lệ tham số gần mục tiêu nhất (hòa thì ít khối hơn)
        if CONFIG["fine_tune_fraction"] is None:
            count = len(self.features)
            return [(str(i), self.features[i]) for i in range(count - CONFIG["fine_tune_blocks"], count)]
        units = self.feature_units()
        sizes = [sum(p.numel() for p in module.parameters()) for _, module in units]
        target, cumulative, best = CONFIG["fine_tune_fraction"] * sum(sizes), 0, None
        for count in range(1, len(units) + 1):
            cumulative += sizes[-count]
            gap = abs(cumulative - target)
            if best is None or gap < best[0]:
                best = (gap, count)
        return units[len(units) - best[1]:]

    def training_mode(self, stage):
        self.train()
        if stage != "scratch":
            self.features.eval()  # §2.5.2: BatchNorm của phần đóng băng giữ chế độ đánh giá
            for _, module in self.tune_units:
                module.train()

    def trainable_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def build_from_checkpoint(saved, device="cpu"):
    # Phần dựng mô hình của reload_checkpoint (notebook, mục 10), tách khỏi torch.load
    if saved["class_mapping"] != CLASS_TO_IDX:
        raise RuntimeError("Class mapping không khớp.")
    model = CatClassifier(
        saved["model_name"],
        pretrained=False,
        dropout=saved["config"]["dropout"],
        head_hidden=saved["config"].get("head_hidden", 0),
        baseline_batchnorm=saved["config"].get("baseline_batchnorm", False),
    ).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model

"""Shared notebook-compatible inference core for CatBreedID."""

from .model import CAT_BREEDS, CatClassifier, build_from_checkpoint
from .package import Package, decide, load_package, predict_arrays
from .preprocess import eval_transform, read_rgb, read_rgb_image
from .scores import METHODS, DistanceScores, logit_scores, unit_norm

__all__ = [
    "CAT_BREEDS",
    "CatClassifier",
    "DistanceScores",
    "METHODS",
    "Package",
    "build_from_checkpoint",
    "decide",
    "eval_transform",
    "load_package",
    "logit_scores",
    "predict_arrays",
    "read_rgb",
    "read_rgb_image",
    "unit_norm",
]

"""Notebook-compatible image preprocessing."""

import io

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CONFIG = {"image_size": 224, "resize_size": 256}
NORMALIZE = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)


def read_rgb(path):
    with Image.open(path) as source:
        source.load()
        if source.mode in ("RGBA", "LA") or "transparency" in source.info:
            rgba = source.convert("RGBA")
            return Image.alpha_composite(
                Image.new("RGBA", rgba.size, (128, 128, 128, 255)), rgba
            ).convert("RGB")
        return source.convert("RGB")


def read_rgb_image(image):
    if isinstance(image, Image.Image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return read_rgb(buffer)
    return read_rgb(image)


def evaluation_geometry():
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(224),
        ]
    )


def eval_transform():
    return transforms.Compose([evaluation_geometry(), transforms.ToTensor(), NORMALIZE])

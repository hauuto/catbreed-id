"""Official notebook preprocessing: alpha composite, EXIF policy and ImageNet normalization."""
from pathlib import Path
import unittest

import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms as T
from app.api.real_model import DeploymentClassifier

PACKAGE = Path(__file__).resolve().parents[1] / "model_package"


@unittest.skipUnless((PACKAGE / "checkpoint.pt").is_file(), "Official package has not been provisioned")
class OfficialPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DeploymentClassifier(PACKAGE, exif_transpose=False)

    def test_rgb_geometry_matches_notebook(self):
        pixels = np.arange(311 * 197 * 3, dtype=np.uint32).reshape(197, 311, 3) % 256
        image = Image.fromarray(pixels.astype(np.uint8))
        transform = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
                               T.CenterCrop(224), T.ToTensor(),
                               T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        torch.testing.assert_close(self.model.preprocess(image)[0], transform(image), rtol=0, atol=0)

    def test_alpha_uses_gray_background(self):
        for mode, color in (("RGBA", (255, 0, 0, 0)), ("LA", (255, 0))):
            with self.subTest(mode=mode):
                image = Image.new(mode, (230, 240), color)
                self.assertEqual(self.model.input_image(image).getpixel((100, 100)), (128, 128, 128))

    def test_exif_is_explicitly_optional(self):
        image = Image.new("RGB", (300, 200), (255, 0, 0))
        image.paste((0, 0, 255), (0, 0, 150, 200))
        image.getexif()[274] = 6
        unrotated = np.asarray(self.model.input_image(image))
        rotated_reference = np.asarray(self.model.input_image(ImageOps.exif_transpose(image)))
        self.model.exif_transpose = True
        try:
            np.testing.assert_array_equal(np.asarray(self.model.input_image(image)), rotated_reference)
            self.assertFalse(np.array_equal(unrotated, rotated_reference))
        finally:
            self.model.exif_transpose = False

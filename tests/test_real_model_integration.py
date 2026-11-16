"""State-dict and forward compatibility with the actual reference notebook source.

Set CATBREED_REFERENCE_NOTEBOOK to the checkout's kaggle_runner.ipynb to run this check.
No notebook cells are executed; only the two architecture definitions are compiled.
"""
import ast
import json
import os
from pathlib import Path
import unittest

import torch
from torch import nn
from torchvision import models
from app.api.real_model import NotebookClassifier

REFERENCE = os.environ.get("CATBREED_REFERENCE_NOTEBOOK")


@unittest.skipUnless(REFERENCE and Path(REFERENCE).is_file(), "CATBREED_REFERENCE_NOTEBOOK is not configured")
class NotebookArchitectureParity(unittest.TestCase):
    def test_all_four_architectures_and_both_heads_match_reference(self):
        torch.set_num_threads(1)
        notebook = json.loads(Path(REFERENCE).read_text(encoding="utf-8"))
        definitions = []
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            if "def baseline_features(" not in source and "class CatClassifier(" not in source:
                continue
            tree = ast.parse(source)
            definitions.extend(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                               and n.name in ("baseline_features", "CatClassifier"))
        self.assertEqual(len(definitions), 2)
        context = {"nn": nn, "CONFIG": {"baseline_batchnorm": False, "dropout": 0.2, "head_hidden": 0},
                   "CAT_BREEDS": list(range(12)), "BACKBONES": {
                       name: getattr(models, name) for name in ("resnet50", "mobilenet_v2", "efficientnet_b0")}}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(REFERENCE), "exec"), context)
        for name in ("cnn_baseline", "resnet50", "mobilenet_v2", "efficientnet_b0"):
            for hidden in (0, 512):
                with self.subTest(name=name, hidden=hidden):
                    reference = context["CatClassifier"](name, pretrained=False, dropout=0.2,
                                                          head_hidden=hidden, baseline_batchnorm=True).eval()
                    actual = NotebookClassifier(name, 0.2, hidden, True).eval()
                    actual.load_state_dict(reference.state_dict(), strict=True)
                    image = torch.linspace(0, 1, 3 * 224 * 224).reshape(1, 3, 224, 224)
                    with torch.no_grad():
                        torch.testing.assert_close(actual(image), reference(image), rtol=0, atol=0)
                    del reference, actual

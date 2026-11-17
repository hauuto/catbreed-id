import unittest
import numpy as np
from scripts.report_serving_mode_effect import summarize

class ServingModeStatsTest(unittest.TestCase):
    def test_changes_and_boundary(self):
        result = summarize(["a", "b"], np.array([[2., 1.], [1., 2.]]),
                           np.array([[1., 2.], [1., 2.]]),
                           np.array([0., -1.]), np.array([-0.1, -1.]), 0.)
        self.assertEqual(result["top1_changed"], 1)
        self.assertEqual(result["knn_decision_changed"], 1)
        self.assertEqual(result["knn_decision_changed_ids"], ["a"])
        self.assertEqual(result["min_notebook_knn_margin"], 0.)
        self.assertAlmostEqual(result["knn_abs_median"], 0.05)
        self.assertEqual(result["logits_abs_median"], 0.5)
        self.assertNotIn("passed", result)

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            summarize(["a"], np.ones((1,2)), np.ones((2,2)), np.ones(1), np.ones(1), 0.)

if __name__ == "__main__":
    unittest.main()

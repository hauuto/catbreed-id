import unittest
from scripts.benchmark_protocol import collect_trials


class BenchmarkProtocolTest(unittest.TestCase):
    def test_exactly_50_calls_first_five_excluded_without_repeating_images(self):
        seen = []
        def predict(image_id):
            seen.append(image_id)
            return {"image_id": image_id, "total": image_id + 1}
        rows = collect_trials(range(50), predict)
        self.assertEqual(seen, list(range(50)))
        self.assertEqual([r["trial"] for r in rows if r["warmup"]], [1, 2, 3, 4, 5])
        self.assertEqual([r["trial"] for r in rows if not r["warmup"]], list(range(6, 51)))
        self.assertEqual(len([r for r in rows if not r["warmup"]]), 45)

    def test_incomplete_dataset_does_not_produce_partial_benchmark(self):
        called = []
        with self.assertRaises(ValueError):
            collect_trials(range(49), lambda item: called.append(item))
        self.assertFalse(called)

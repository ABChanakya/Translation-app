from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.detection_metrics import box_iou, match_predictions_to_ground_truth  # noqa: E402


class DetectionMetricsTests(unittest.TestCase):
    def test_box_iou_returns_expected_overlap(self) -> None:
        iou = box_iou((0.0, 0.0, 10.0, 10.0), (5.0, 5.0, 15.0, 15.0))
        self.assertAlmostEqual(iou, 25.0 / 175.0)

    def test_prediction_matching_is_confidence_ordered_and_one_to_one(self) -> None:
        predictions = [
            ((0.0, 0.0, 10.0, 10.0), 0.9),
            ((1.0, 1.0, 9.0, 9.0), 0.8),
        ]
        ground_truths = [(0.0, 0.0, 10.0, 10.0)]
        matches = match_predictions_to_ground_truth(predictions, ground_truths, match_iou_threshold=0.5)
        self.assertEqual(len(matches), 1)
        self.assertAlmostEqual(matches[0], 1.0)


if __name__ == "__main__":
    unittest.main()

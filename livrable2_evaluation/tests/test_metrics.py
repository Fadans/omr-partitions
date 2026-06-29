import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from metrics import detection_f1, iou_box, note_scores


class TestDetection(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(iou_box([0, 0, 2, 2], [0, 0, 2, 2]), 1.0)
        self.assertAlmostEqual(iou_box([0, 0, 2, 2], [2, 2, 4, 4]), 0.0)

    def test_f1(self):
        gt = [[0, 0, 10, 10], [20, 20, 30, 30]]
        pred = [[0, 0, 10, 10], [100, 100, 110, 110]]
        r = detection_f1(pred, gt)
        self.assertEqual((r["tp"], r["fp"], r["fn"]), (1, 1, 1))
        self.assertAlmostEqual(r["f1"], 0.5)


class TestNotes(unittest.TestCase):
    def test_accuracies(self):
        gt = [{"pitch": "C4", "duration": "q"},
              {"pitch": "E4", "duration": "q"},
              {"pitch": "G4", "duration": "h"}]
        pred = [{"pitch": "C4", "duration": "q"},
                {"pitch": "F4", "duration": "q"},
                {"pitch": "G4", "duration": "q"}]
        r = note_scores(pred, gt)
        self.assertAlmostEqual(r["pitch_accuracy"], 100 * 2 / 3)
        self.assertAlmostEqual(r["rhythm_accuracy"], 100 * 2 / 3)
        self.assertAlmostEqual(r["note_accuracy"], 100 * 1 / 3)

    def test_note_manquee(self):
        gt = [{"pitch": "C4", "duration": "q"}, {"pitch": "E4", "duration": "q"}]
        pred = [{"pitch": "C4", "duration": "q"}]
        r = note_scores(pred, gt)
        self.assertAlmostEqual(r["pitch_accuracy"], 50.0)


if __name__ == "__main__":
    unittest.main()

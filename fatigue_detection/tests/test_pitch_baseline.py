from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fatigue import FatigueScorer
from features import PitchBaseline


class PitchBaselineTests(unittest.TestCase):
    def test_stable_offsets_at_different_frame_rates(self):
        for offset, fps in [(-4, 15), (9, 30)]:
            with self.subTest(offset=offset, fps=fps):
                baseline = PitchBaseline(3.0, 6.0, 0.25)
                for frame in range(3 * fps + 1):
                    baseline.update(offset + (frame % 3 - 1) * 0.2, frame / fps)
                self.assertAlmostEqual(baseline.value, offset)
                baseline.interrupt()
                baseline.update(offset + 15, 4.0)
                self.assertAlmostEqual(baseline.value, offset)

    def test_movement_and_gaps_restart_acquisition(self):
        baseline = PitchBaseline(3.0, 6.0, 0.25)
        for i in range(21):
            baseline.update(9, i / 10)
        baseline.update(25, 2.1)
        self.assertIsNone(baseline.value)
        baseline.update(None, 2.2)
        for i in range(31):
            baseline.update(-3, 4 + i / 10)
        self.assertEqual(baseline.value, -3)
        baseline.reset()
        self.assertIsNone(baseline.value)

    def test_head_down_cap_and_score_clear_leave_eye_evidence_intact(self):
        scorer = FatigueScorer()
        for i in range(4200):
            scorer.update(1 / 30, {"head_down": True}, now=i / 30)
        self.assertAlmostEqual(scorer.get_score(), 3.0)
        self.assertEqual(scorer.get_level()[0], 1)
        scorer.update(1 / 30, {"eye_closed_sec": 2.0}, now=140.0)
        scorer.clear_head_down()
        self.assertAlmostEqual(scorer.get_score(), 4.0)
        self.assertEqual(dict(scorer.get_active_sources()), {"micro_sleep": 4.0})


if __name__ == "__main__":
    unittest.main()

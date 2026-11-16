import csv
import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import main
from features import PERCLOSCounter, calculate_head_pose
from fatigue import FatigueScorer


def replay(frames):
    """Feed timestamp, landmark count, yaw and EAR through the application loop."""
    source = iter(frames)
    sample = frames[0]
    observed = []
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    clock = SimpleNamespace(monotonic=lambda: sample[0], strftime=lambda _: "replay")
    camera = SimpleNamespace(
        open=lambda: True, resolution=(640, 480), fps=30, release=lambda: None,
    )

    def read():
        nonlocal sample
        next_sample = next(source, None)
        if next_sample is None:
            return False, None
        sample = next_sample
        return True, image

    camera.read = read
    detector = SimpleNamespace(
        detect=lambda _: None,
        has_face=lambda _: sample[1] > 0,
        get_landmarks=lambda _: [(0.5, 0.5)] * sample[1],
        close=lambda: None,
    )

    class RecordingScorer(FatigueScorer):
        def update(self, dt, events, **kwargs):
            super().update(dt, events, **kwargs)
            observed.append((sample[0], events["eye_closed_sec"], self.get_level()[0], kwargs))

    with TemporaryDirectory() as directory:
        with (
            patch.multiple(
                main, Camera=lambda: camera, FaceMeshDetector=lambda: detector,
                FatigueScorer=RecordingScorer, time=clock, LOG_DIR=directory,
                calculate_ear=lambda *_: sample[3], calculate_mar=lambda *_: 0.1,
                calculate_head_pose=lambda *_, **__: None if sample[2] is None else (-4.0, sample[2], 0.0),
                _HAS_SOUND=False, SHOW_FACE_MESH=False, SHOW_HEAD_POSE_AXIS=False,
                SHOW_EYE_POINTS=False, SHOW_MOUTH_POINTS=False,
            ),
            patch.object(main, "_HAS_DISTRACTION", False, create=True),
            patch.multiple(main.cv2, imshow=lambda *_: None, waitKey=lambda *_: -1,
                           destroyAllWindows=lambda: None),
            redirect_stdout(io.StringIO()),
        ):
            main.main()
        with (Path(directory) / "run_replay.csv").open(encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file)), observed


class ObservationTimingTests(unittest.TestCase):
    def test_no_face_short_landmarks_and_invalid_pose_are_logged(self):
        rows, observed = replay([(0, 0, 0, 0), (1, 100, 0, 0.1), (2, 478, None, 0.1)])
        self.assertEqual([row["face_valid"] for row in rows], ["0", "0", "1"])
        self.assertTrue(all(row["eye_valid"] == row["pose_valid"] == "0" for row in rows))
        self.assertTrue(all(row["perclos_valid_sec"] == "0.000" for row in rows))
        self.assertTrue(all(closed == 0 and level == 0 for _, closed, level, _ in observed))

    def test_side_face_low_ear_is_excluded_from_closure_and_perclos(self):
        rows, observed = replay([(i / 30, 478, 35, 0.18) for i in range(121)])
        self.assertTrue(all(row["eye_valid"] == "0" for row in rows))
        self.assertTrue(all(row["perclos_valid_sec"] == "0.000" for row in rows))
        self.assertTrue(all(closed == 0 and level == 0 for _, closed, level, _ in observed))

    def test_face_loss_breaks_continuous_closure(self):
        frames = [(i / 30, 478 if i != 45 else 0, 0, 0.12) for i in range(91)]
        _, observed = replay(frames)
        self.assertLess(max(closed for _, closed, _, _ in observed), 1.6)
        self.assertEqual(observed[45][1], 0.0)
        self.assertEqual(observed[46][1], 0.0)

    def test_camera_stall_only_decays_score_and_restarts_closure(self):
        frames = [(i / 30, 478, 0, 0.12) for i in range(31)]
        frames.extend((2 + i / 30, 478, 0, 0.12) for i in range(31))
        _, observed = replay(frames)
        self.assertEqual(observed[31][1], 0.0)
        self.assertEqual(observed[31][3]["observed_dt"], 0.0)
        self.assertLessEqual(max(closed for _, closed, _, _ in observed), 1.0)

    def test_sustained_frontal_closure_still_reaches_level_three(self):
        _, observed = replay([(i / 30, 478, 0, 0.12) for i in range(91)])
        self.assertEqual(observed[60][1:3], (2.0, 3))

    def test_pose_solver_failure_can_be_reported_as_invalid(self):
        with patch("features.cv2.solvePnP", return_value=(False, None, None)):
            self.assertIsNone(calculate_head_pose([(0.5, 0.5)] * 478, 640, 480, invalid_as_none=True))
            self.assertEqual(calculate_head_pose([(0.5, 0.5)] * 478, 640, 480), (0.0, 0.0, 0.0))

    def test_perclos_is_consistent_at_15_and_30_fps(self):
        for fps in (15, 30):
            with self.subTest(fps=fps):
                counter = PERCLOSCounter()
                for i in range(40 * fps + 1):
                    counter.update(0.12 if i >= 30 * fps else 0.35, i / fps)
                value, valid_sec = counter.get_stats()
                self.assertAlmostEqual(value, 1 / 3)
                self.assertAlmostEqual(valid_sec, 30)

    def test_short_blink_is_exempt_before_and_after_reopening_in_full_window(self):
        counter = PERCLOSCounter()
        for i in range(30 * 30):
            counter.update(0.35, i / 30)
        for i in range(6):
            counter.update(0.12, 30 + i / 30)
            self.assertEqual(counter.get_perclos(), 0.0)
        counter.update(0.35, 30.2)
        self.assertEqual(counter.get_perclos(), 0.0)

    def test_invalid_samples_and_stalls_are_excluded_from_valid_time(self):
        counter = PERCLOSCounter()
        for t, ear in [(0, 0.12), (0.2, 0.12), (0.3, None), (5, 0.12), (5.2, 0.12)]:
            counter.update(ear, t)
        value, valid_sec = counter.get_stats()
        self.assertEqual(value, 0.0)
        self.assertAlmostEqual(valid_sec, 0.4)
        counter.update(None, 36)
        self.assertEqual(counter.get_stats(), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()

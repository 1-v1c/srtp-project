from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import YAW_EAR_GATE
from features import (
    MOUTH_CORNER_LEFT,
    MOUTH_CORNER_RIGHT,
    MOUTH_VERTICAL_PAIRS,
    _FACE_LANDMARK_IDS,
    _FACE_MODEL_3D,
    calculate_head_pose,
    calculate_mar,
)


def landmarks_for_pose(*, pitch=0.0, yaw=0.0, mouth_height=0.02):
    width, height = 640, 480
    camera = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], dtype=np.float64)
    rotation = np.radians([pitch, yaw, 0.0])
    translation = np.array([0.0, 0.0, 700.0])
    image_points, _ = cv2.projectPoints(_FACE_MODEL_3D, rotation, translation, camera, None)
    points = [(0.5, 0.5)] * 478
    for index, (x, y) in zip(_FACE_LANDMARK_IDS, image_points.reshape(-1, 2)):
        points[index] = (x / width, y / height)
    points[MOUTH_CORNER_LEFT] = (0.4, 0.63)
    points[MOUTH_CORNER_RIGHT] = (0.6, 0.63)
    points[152] = (0.5, 0.80 + mouth_height / 2)
    for upper, lower in MOUTH_VERTICAL_PAIRS:
        points[upper] = (0.5, 0.63 - mouth_height / 2)
        points[lower] = (0.5, 0.63 + mouth_height / 2)
    return points


class UpperFacePoseTests(unittest.TestCase):
    def test_opening_mouth_does_not_change_eye_yaw_gate(self):
        closed = landmarks_for_pose(yaw=8.0)
        opened = landmarks_for_pose(yaw=8.0, mouth_height=0.18)
        self.assertLess(calculate_mar(closed), 0.2)
        self.assertGreater(calculate_mar(opened), 0.8)
        opened_pose = calculate_head_pose(opened, 640, 480, invalid_as_none=True)
        closed_pose = calculate_head_pose(closed, 640, 480, invalid_as_none=True)
        np.testing.assert_allclose(opened_pose, closed_pose, atol=0.01)
        self.assertLess(abs(opened_pose[1]), YAW_EAR_GATE)

    def test_actual_side_face_still_exceeds_gate_with_open_mouth(self):
        points = landmarks_for_pose(yaw=35.0, mouth_height=0.18)
        _, yaw, _ = calculate_head_pose(points, 640, 480, invalid_as_none=True)
        self.assertAlmostEqual(yaw, 35.0, delta=0.01)
        self.assertGreater(abs(yaw), YAW_EAR_GATE)

    def test_looking_down_is_positive_pitch(self):
        points = landmarks_for_pose(pitch=18.0)
        pitch, _, _ = calculate_head_pose(points, 640, 480, invalid_as_none=True)
        self.assertAlmostEqual(pitch, 18.0, delta=0.01)

    def test_pose_behind_camera_is_invalid(self):
        points = landmarks_for_pose()
        rotation = np.zeros((3, 1))
        translation = np.array([[0.0], [0.0], [-700.0]])
        with patch("features.cv2.solvePnP", return_value=(True, rotation, translation)):
            self.assertIsNone(calculate_head_pose(points, 640, 480, invalid_as_none=True))

    def test_pose_with_large_projection_error_is_invalid(self):
        points = landmarks_for_pose()
        rotation = np.zeros((3, 1))
        translation = np.array([[260.0], [0.0], [700.0]])
        with patch("features.cv2.solvePnP", return_value=(True, rotation, translation)):
            self.assertIsNone(calculate_head_pose(points, 640, 480, invalid_as_none=True))


if __name__ == "__main__":
    unittest.main()

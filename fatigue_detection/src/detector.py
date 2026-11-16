"""
MediaPipe Face Mesh 检测器模块（Tasks API 版本）
- 输入 BGR 图像 → 输出 478 个人脸关键点
- 使用 MediaPipe 0.10.x 新版 Tasks API
"""

import os
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as tasks
from mediapipe.tasks.python import vision

try:
    from .config import (
        MAX_NUM_FACES,
        MIN_DETECTION_CONFIDENCE,
        MIN_TRACKING_CONFIDENCE,
        FACE_LANDMARKER_MODEL_PATH,
    )
except ImportError:
    from config import (
        MAX_NUM_FACES,
        MIN_DETECTION_CONFIDENCE,
        MIN_TRACKING_CONFIDENCE,
        FACE_LANDMARKER_MODEL_PATH,
    )


class FaceMeshDetector:
    """MediaPipe Face Landmarker 封装（Tasks API）"""

    def __init__(self, model_path: str = None):
        configured_path = model_path or FACE_LANDMARKER_MODEL_PATH
        if os.path.exists(configured_path):
            base_options = tasks.BaseOptions(model_asset_path=configured_path)
        elif model_path is None:
            bundled_model = Path(__file__).resolve().parents[1] / "models/face_landmarker.task"
            base_options = tasks.BaseOptions(model_asset_buffer=bundled_model.read_bytes())
        else:
            raise FileNotFoundError(f"模型文件未找到: {model_path}")

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=MAX_NUM_FACES,
            min_face_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            output_face_blendshapes=False,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self._timestamp = 0  # 毫秒计数器

    def detect(self, bgr_image: np.ndarray):
        """
        检测人脸关键点（VIDEO 模式，需要时间戳）

        参数:
            bgr_image: OpenCV BGR 格式图像

        返回:
            results: FaceLandmarkerResult 对象
                - results.face_landmarks: 每张脸的 478 个关键点列表
                  每个关键点有 x, y, z（归一化坐标）
        """
        # OpenCV BGR → MediaPipe RGB Image
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )

        self._timestamp += 33  # 约 30 FPS，每帧 +33ms
        results = self.landmarker.detect_for_video(mp_image, self._timestamp)
        return results

    def get_landmarks(self, results, face_idx: int = 0):
        """
        从检测结果中提取关键点（归一化坐标）

        参数:
            results: detect() 返回的结果
            face_idx: 第几张脸（默认第一张）

        返回:
            list of (x, y) 归一化坐标，无人脸时返回空列表
        """
        if not results.face_landmarks:
            return []
        if face_idx >= len(results.face_landmarks):
            return []

        landmarks = results.face_landmarks[face_idx]
        return [(lm.x, lm.y) for lm in landmarks]

    def has_face(self, results) -> bool:
        """检测到人脸了吗？"""
        return (results.face_landmarks is not None and
                len(results.face_landmarks) > 0)

    def draw_mesh(self, image: np.ndarray, results):
        """
        在图像上绘制 Face Mesh 绿网（直接在 image 上修改）

        参数:
            image: BGR numpy 图像
            results: detect() 返回的结果
        """
        if not results.face_landmarks:
            return

        from mediapipe.tasks.python.vision import drawing_utils

        # 绘制每张脸
        for face_landmarks in results.face_landmarks:
            # 轮廓连线（眼睛、嘴、眉毛、鼻子核心线条）
            drawing_utils.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                landmark_drawing_spec=drawing_utils.DrawingSpec(
                    color=(0, 255, 0), thickness=1, circle_radius=1
                ),
                connection_drawing_spec=drawing_utils.DrawingSpec(
                    color=(0, 255, 0), thickness=1, circle_radius=0
                ),
                is_drawing_landmarks=False,  # 不画点，只画线（更清爽）
            )
            # 虹膜（眼睛里面的圈）
            drawing_utils.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_utils.DrawingSpec(
                    color=(0, 255, 255), thickness=1, circle_radius=0
                ),
                is_drawing_landmarks=False,
            )
            drawing_utils.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_utils.DrawingSpec(
                    color=(0, 255, 255), thickness=1, circle_radius=0
                ),
                is_drawing_landmarks=False,
            )

    def close(self):
        """释放 MediaPipe 资源"""
        self.landmarker.close()

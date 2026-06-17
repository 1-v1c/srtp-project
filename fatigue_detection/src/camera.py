"""
摄像头驱动模块
- 打开/关闭摄像头
- 读取单帧画面
- 可选的画面大小调整
"""

import cv2
import time

try:
    from .config import CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT
except ImportError:
    from config import CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT


class Camera:
    """USB/内置摄像头封装"""

    def __init__(self, camera_id: int = None):
        self.camera_id = camera_id if camera_id is not None else CAMERA_ID
        self.cap = None
        self._fps = 0.0
        self._frame_count = 0
        self._start_time = None

    def open(self) -> bool:
        """打开摄像头，返回是否成功"""
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self._start_time = time.time()
        self._frame_count = 0
        return True

    def read(self):
        """读取一帧，返回 (成功, BGR画面)"""
        if self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        if ret:
            self._frame_count += 1
            self._update_fps()
        return ret, frame

    def _update_fps(self):
        """每秒更新一次 FPS"""
        elapsed = time.time() - self._start_time
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._start_time = time.time()

    @property
    def fps(self) -> float:
        """当前帧率（约每秒刷新一次）"""
        return round(self._fps, 1)

    @property
    def resolution(self) -> tuple:
        """当前分辨率 (w, h)"""
        if self.cap is None:
            return (0, 0)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def release(self):
        """释放摄像头"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

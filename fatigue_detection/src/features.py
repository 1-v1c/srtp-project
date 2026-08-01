"""
特征提取模块
- EAR（Eye Aspect Ratio）眼部长宽比
- MAR（Mouth Aspect Ratio）嘴部长宽比
- Head Pose 头部姿态（阶段 5）

MediaPipe Face Mesh 关键点编号（旧版 468 点模型，新版 478 点前 468 个相同）：
  左眼: 33(左角), 160(上外), 158(上内), 133(右角), 153(下内), 144(下外)
  右眼: 362(左角), 385(上外), 387(上内), 263(右角), 373(下内), 380(下外)
  嘴部: 61(左角), 291(右角), 13(上唇), 14(下唇)
"""

import time
import cv2
import numpy as np
from collections import deque


# ============================================================
# 关键点索引（MediaPipe Face Mesh）
# ============================================================

# 左眼 6 点：从左眼角 → 上眼睑外 → 上眼睑内 → 右眼角 → 下眼睑内 → 下眼睑外
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# 右眼 6 点
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# 嘴部 MAR 计算点
# 水平参考：左嘴角、右嘴角
MOUTH_CORNER_LEFT  = 61
MOUTH_CORNER_RIGHT = 291
# 三组垂直测量（DrowSAFE 增强版）：(上唇点, 下唇点)
MOUTH_VERTICAL_PAIRS = [
    (13, 14),    # 中央：上唇中点 ↔ 下唇中点
    (39, 181),   # 左侧：上唇外缘 ↔ 下唇外缘
    (269, 405),  # 右侧：上唇外缘 ↔ 下唇外缘
]


# ============================================================
# EAR 计算
# ============================================================

def calculate_ear(landmarks, eye_indices, img_w: int = 1, img_h: int = 1):
    """
    计算单只眼睛的 Eye Aspect Ratio

    公式: EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

    参数:
        landmarks: list of (x, y) 归一化坐标或像素坐标
        eye_indices: 6 个关键点的索引列表 [p1..p6]
        img_w, img_h: 图像宽高（若 landmarks 是归一化坐标，传入 1 即可）

    返回:
        float: EAR 值
    """
    pts = []
    for idx in eye_indices:
        x, y = landmarks[idx]
        pts.append(np.array([x * img_w, y * img_h]))

    p1, p2, p3, p4, p5, p6 = pts

    v1 = np.linalg.norm(p2 - p6)  # 外侧眼睑间距
    v2 = np.linalg.norm(p3 - p5)  # 内侧眼睑间距
    h = np.linalg.norm(p1 - p4)   # 眼角间距

    if h < 1e-6:
        return 0.0

    ear = (v1 + v2) / (2.0 * h)
    return float(ear)


# ============================================================
# MAR 计算
# ============================================================

def calculate_mar(landmarks, img_w: int = 1, img_h: int = 1):
    """
    计算 Mouth Aspect Ratio（嘴部长宽比）—— DrowSAFE 3 组垂直测量增强版

    公式: MAR = (|39-181| + |13-14| + |269-405|) / (3 × |61-291|)

    三组上下唇点：左(39,181) + 中(13,14) + 右(269,405)，取平均消除微笑干扰
    水平参考：嘴角间距 |61-291|

    参数:
        landmarks: list of (x, y) 归一化坐标
        img_w, img_h: 图像宽高（归一化坐标传 1）

    返回:
        float: MAR 值
    """
    def _pt(idx):
        return np.array([landmarks[idx][0] * img_w, landmarks[idx][1] * img_h])

    # 三组垂直距离
    v_sum = 0.0
    for up_idx, lo_idx in MOUTH_VERTICAL_PAIRS:
        v_sum += np.linalg.norm(_pt(up_idx) - _pt(lo_idx))

    # 水平距离（嘴角）
    horizontal = np.linalg.norm(_pt(MOUTH_CORNER_LEFT) - _pt(MOUTH_CORNER_RIGHT))

    if horizontal < 1e-6:
        return 0.0

    return float(v_sum / (3.0 * horizontal))


# ============================================================
# 时序平滑器
# ============================================================

class EARSmoother:
    """用滑动窗口平滑 EAR 值，减少抖动"""

    def __init__(self, window_size: int = 5):
        self.left_ear = deque(maxlen=window_size)
        self.right_ear = deque(maxlen=window_size)

    def update(self, left: float, right: float) -> tuple:
        """添加新值，返回平滑后的 (left_ear, right_ear, avg_ear)"""
        self.left_ear.append(left)
        self.right_ear.append(right)
        avg_left = np.mean(self.left_ear) if self.left_ear else left
        avg_right = np.mean(self.right_ear) if self.right_ear else right
        avg = (avg_left + avg_right) / 2.0
        return avg_left, avg_right, avg


class MARSmoother:
    """用滑动窗口平滑 MAR 值，减少抖动"""

    def __init__(self, window_size: int = 5):
        self.mar_values = deque(maxlen=window_size)

    def update(self, mar: float) -> float:
        """添加新值，返回平滑后的 MAR"""
        self.mar_values.append(mar)
        return float(np.mean(self.mar_values)) if self.mar_values else mar


# ============================================================
# Head Pose — 3D 头部姿态估计 (solvePnP)
# ============================================================

# 6 个参考点的 3D 坐标（通用人脸模型，单位 mm）
# 原始 MediaPipe 模型是 Y 轴朝上；OpenCV solvePnP 使用 Y 轴朝下，故 Y 值取反
_FACE_MODEL_3D = np.array([
    (0.0, 0.0, 0.0),             #   1: 鼻尖
    (0.0, 330.0, -65.0),         # 152: 下巴（在鼻尖下方 → Y 正）
    (-225.0, -170.0, -135.0),    #  33: 左眼左眼角（在鼻尖上方 → Y 负）
    (225.0, -170.0, -135.0),     # 263: 右眼右眼角
    (-150.0, 150.0, -125.0),     #  61: 左嘴角（在鼻尖下方 → Y 正）
    (150.0, 150.0, -125.0),      # 291: 右嘴角
], dtype=np.float64)

_FACE_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


def _solve_head_pnp(landmarks, img_w, img_h):
    """
    执行 solvePnP 求解头部旋转 + 平移。

    返回 (rvec, tvec, camera_matrix) 或 None。
    """
    # 提取 2D 像素坐标
    img_points = []
    for idx in _FACE_LANDMARK_IDS:
        if idx >= len(landmarks):
            return None
        x, y = landmarks[idx]
        img_points.append([x * img_w, y * img_h])

    # 近似相机内参（无需精确标定）
    focal_length = max(img_w, img_h)
    camera_matrix = np.array([
        [focal_length, 0, img_w / 2],
        [0, focal_length, img_h / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    try:
        _, rvec, tvec = cv2.solvePnP(
            _FACE_MODEL_3D,
            np.array(img_points, dtype=np.float64),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        return rvec, tvec, camera_matrix
    except cv2.error:
        return None


def calculate_head_pose(landmarks, img_w, img_h):
    """
    估计头部 3D 姿态。

    参数:
        landmarks: MediaPipe Face Mesh 478 个关键点 [(x,y), ...]
        img_w, img_h: 图像宽高（像素）

    返回:
        (pitch, yaw, roll) 角度值，失败时返回 (0.0, 0.0, 0.0)

    角度约定（可经实际测试调整符号）:
        pitch > 0 = 低头
        yaw   > 0 = 向左转
        roll  > 0 = 向左歪头
    """
    result = _solve_head_pnp(landmarks, img_w, img_h)
    if result is None:
        return (0.0, 0.0, 0.0)

    rvec, tvec, _ = result
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # 从旋转矩阵分解欧拉角（XYZ 约定）
    # 正 pitch = 低头，正 yaw = 左转，正 roll = 左歪
    pitch = -np.arctan2(rotation_matrix[2][1], rotation_matrix[2][2])
    yaw   = np.arcsin(-rotation_matrix[2][0])
    roll  = np.arctan2(rotation_matrix[1][0], rotation_matrix[0][0])

    # 防 NaN：极端角度下 arcsin 可能产出 NaN，归零保护
    if np.isnan(pitch) or np.isinf(pitch): pitch = 0.0
    if np.isnan(yaw)   or np.isinf(yaw):   yaw   = 0.0
    if np.isnan(roll)  or np.isinf(roll):  roll  = 0.0

    return (float(np.degrees(pitch)),
            float(np.degrees(yaw)),
            float(np.degrees(roll)))


def draw_head_axes(image, landmarks, img_w, img_h):
    """
    在面部绘制 RGB 3D 坐标轴（可视化头部朝向）。

    颜色约定:
        红色 = X 轴（人脸右侧）
        绿色 = Y 轴（人脸下方）
        蓝色 = Z 轴（人脸前方）
    """
    result = _solve_head_pnp(landmarks, img_w, img_h)
    if result is None:
        return

    rvec, tvec, camera_matrix = result
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # 3D 坐标轴端点（从鼻尖出发，80mm 长）
    axis_3d = np.float32([
        [0, 0, 0],
        [80, 0, 0],     # X
        [0, -80, 0],    # Y（模型 Y 朝下，取反使绿色指向上方）
        [0, 0, -80],    # Z（模型 Z 朝前，取反使蓝色指向画外）
    ])

    # 投影到 2D
    axis_2d, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)

    # 绘制
    origin = tuple(axis_2d[0].ravel().astype(int))
    cv2.line(image, origin, tuple(axis_2d[1].ravel().astype(int)), (0, 0, 255), 3)   # X: 红
    cv2.line(image, origin, tuple(axis_2d[2].ravel().astype(int)), (0, 255, 0), 3)   # Y: 绿
    cv2.line(image, origin, tuple(axis_2d[3].ravel().astype(int)), (255, 0, 0), 3)   # Z: 蓝


class HeadPoseSmoother:
    """滑动窗口平滑 Pitch / Yaw / Roll 三轴角度"""

    def __init__(self, window_size: int = 5):
        self.pitch = deque(maxlen=window_size)
        self.yaw = deque(maxlen=window_size)
        self.roll = deque(maxlen=window_size)

    def update(self, pitch: float, yaw: float, roll: float) -> tuple:
        """填入新值，返回平滑后的 (pitch, yaw, roll)"""
        self.pitch.append(pitch)
        self.yaw.append(yaw)
        self.roll.append(roll)
        return (
            float(np.mean(self.pitch)),
            float(np.mean(self.yaw)),
            float(np.mean(self.roll)),
        )


class PERCLOSCounter:
    """
    滑动窗口闭眼比例统计器。

    规则:
    - EAR < ear_threshold 视为闭眼帧
    - 无人脸帧用 update(None) 传入，跳过不计
    - 窗口长度由 window_sec × fps 决定
    - 眨眼豁免：闭眼持续 < blink_exempt_sec 视为正常眨眼，不计入闭眼帧
      （对齐文献标准：PERCLOS 统计排除正常眨眼）
    """

    def __init__(self, window_sec: int = 30, ear_threshold: float = 0.22, fps: int = 30,
                 blink_exempt_sec: float = 0.3):
        self.window_sec = max(1, int(window_sec))
        self.ear_threshold = ear_threshold
        self.fps = max(1, int(fps))
        self.window_frames = max(1, self.window_sec * self.fps)
        self.blink_exempt_frames = max(0, int(blink_exempt_sec * self.fps))

        # 保存每帧的闭眼状态 True/False
        self._closed_states = deque(maxlen=self.window_frames)

        # 眨眼豁免状态：记录当前闭眼会话的起始位置，未超过豁免帧数前可回溯豁免
        self._pending_start: int | None = None      # 闭眼开始帧在队列中的位置
        self._pending_committed = False             # 闭眼已超过豁免帧数（转正，不再回溯）

    def update(self, ear_value: float | None) -> None:
        """
        每帧调用一次。
        ear_value 为 None 表示无人脸，跳过不计。
        """
        if ear_value is None:
            return
        closed = ear_value < self.ear_threshold

        # 先按原始状态入队（保证窗口帧数完整）
        self._closed_states.append(closed)

        if closed:
            if self._pending_start is None:
                # 闭眼会话开始
                self._pending_start = len(self._closed_states) - 1
                self._pending_committed = False
            elif not self._pending_committed and (
                len(self._closed_states) - self._pending_start >= self.blink_exempt_frames
            ):
                # 闭眼持续超过豁免帧数 → 转正（此前的帧保持 True，不再修改）
                self._pending_committed = True
        else:
            # 睁眼：若闭眼会话在豁免期内结束 → 视为眨眼，回溯豁免
            if self._pending_start is not None and not self._pending_committed:
                start = max(0, self._pending_start)
                # 当前睁眼帧保留 False，回溯修改闭眼会话期间的帧
                for i in range(start, len(self._closed_states) - 1):
                    self._closed_states[i] = False
            self._pending_start = None
            self._pending_committed = False

    def get_perclos(self) -> float:
        """
        返回当前窗口内的闭眼帧比例，取值 0.0 ~ 1.0。
        窗口未满时，按已有数据计算，不做零填充。
        正常眨眼（< 豁免期）不计入闭眼。
        """
        if not self._closed_states:
            return 0.0
        return float(sum(self._closed_states) / len(self._closed_states))

    def reset(self) -> None:
        """清空窗口数据"""
        self._closed_states.clear()
        self._pending_start = None
        self._pending_committed = False


# ============================================================
# 眨眼动力学分析器（P0 升级）
# 从 EAR 时间序列检测眨眼，统计疲劳早期信号：
#   - 眨眼频率（次/分）：疲劳时升高
#   - 平均眨眼时长：疲劳时延长（正常 ~200ms，疲劳 >400ms）
#   - 慢眨眼频率（≥ slow_blink_sec）：疲劳时增多
#   - 平均闭合速度：疲劳时变慢（眼睑闭合缓慢，文献 AVR 信号）
# ============================================================

class BlinkAnalyzer:
    """
    眨眼动力学分析器。

    眨眼检测状态机：
        open → (ear < threshold) → closed → (ear ≥ threshold) → 完成一次眨眼
    闭眼时长分档：
        < slow_blink_sec           → 正常眨眼（计入统计）
        slow_blink_sec ~ long_closure_sec → 慢眨眼（疲劳信号，计入统计）
        ≥ long_closure_sec         → 长闭眼/微睡眠（不计为眨眼，由闭眼逻辑处理）

    用法（每帧调用，t 用 time.time()）：
        analyzer.update(ear_raw, time.time())   # ear 为未平滑的原始值
        stats = analyzer.get_stats()            # 60s 窗口统计
    """

    def __init__(self, ear_threshold: float = 0.22, window_sec: float = 60.0,
                 slow_blink_sec: float = 0.5, long_closure_sec: float = 2.0,
                 history_len: int = 15):
        self._threshold = ear_threshold
        self._window_sec = max(1.0, window_sec)
        self._slow_blink_sec = slow_blink_sec
        self._long_closure_sec = max(slow_blink_sec, long_closure_sec)

        # (t, ear) 历史缓冲：用于计算闭合速度（峰值 → 跌破阈值）
        self._history = deque(maxlen=history_len)
        # 眨眼事件：(完成时间, 持续时长, 闭合速度)
        self._blinks: deque = deque()
        self._state = "open"          # open / closed
        self._closed_t0: float | None = None   # 跌破阈值的时刻

    def update(self, ear: float | None, t: float) -> None:
        """每帧调用。ear=None 表示无人脸，重置闭合状态。"""
        if ear is None:
            self._state = "open"
            self._closed_t0 = None
            self._history.clear()
            return

        self._history.append((t, ear))

        if self._state == "open":
            if ear < self._threshold:
                self._state = "closed"
                self._closed_t0 = t
        else:  # closed
            if ear >= self._threshold:
                # 眨眼/长闭眼结束
                duration = t - self._closed_t0
                if duration < self._long_closure_sec:
                    # 正常眨眼 / 慢眨眼 → 记录事件（长闭眼不计为眨眼）
                    self._blinks.append((t, duration, self._calc_closing_speed()))
                self._state = "open"
                self._closed_t0 = None

    def _calc_closing_speed(self) -> float:
        """
        计算本次闭眼的闭合速度：EAR 从峰值跌到阈值以下的速度。
        近似：speed = (peak_ear - threshold) / 闭合时间
        """
        peak_ear = -1.0
        peak_t = None
        for h_t, h_ear in self._history:
            if h_ear > peak_ear:
                peak_ear = h_ear
                peak_t = h_t
        if peak_t is None or self._closed_t0 is None or peak_ear <= self._threshold:
            return 0.0
        closing_time = self._closed_t0 - peak_t
        if closing_time <= 0:
            return 0.0
        return (peak_ear - self._threshold) / closing_time

    def get_stats(self, now: float | None = None) -> dict:
        """
        返回滑动窗口内的眨眼统计：
        {blink_rate(次/分), avg_duration(秒), slow_blink_rate(次/分), avg_closing_speed(EAR/秒)}
        """
        now = now if now is not None else time.time()
        recent = [(t, d, s) for t, d, s in self._blinks if now - t <= self._window_sec]
        if not recent:
            return {
                "blink_rate": 0.0,
                "avg_duration": 0.0,
                "slow_blink_rate": 0.0,
                "avg_closing_speed": 0.0,
            }
        per_min = 60.0 / self._window_sec
        avg_dur = sum(d for _, d, _ in recent) / len(recent)
        slow_rate = sum(1 for _, d, _ in recent if d >= self._slow_blink_sec) * per_min
        speeds = [s for _, _, s in recent if s > 0]
        return {
            "blink_rate": len(recent) * per_min,
            "avg_duration": avg_dur,
            "slow_blink_rate": slow_rate,
            "avg_closing_speed": sum(speeds) / len(speeds) if speeds else 0.0,
        }

    def reset(self) -> None:
        """清空全部状态"""
        self._blinks.clear()
        self._history.clear()
        self._state = "open"
        self._closed_t0 = None

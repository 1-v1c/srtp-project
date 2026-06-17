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

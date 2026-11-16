"""
疲劳评分器 — 模块 07
将 EAR / MAR / HeadPose / PERCLOS 多特征融合为连续疲劳评分 + L0~L3 四级状态。

设计依据（Bergasa 2006 多级疲劳状态机 + 时间衰减思想）：
- 事件源：一次性加分（哈欠完成、微睡眠发生）
- 状态源：状态持续期间按速率累积分数（闭眼、PERCLOS 超限、低头、偏头）
- 全局指数衰减：疲劳症状停止后分数按半衰期回落，避免历史事件永久拉高分数
- 可解释性：每个分数源的贡献独立记录，可追溯"为什么判疲劳"
"""

import time
from collections import deque

from config import (
    EYE_CLOSED_05S_SCORE, EYE_CLOSED_2S_SCORE,
    PERCLOS_OVER_SCORE, YAWN_SCORE, YAWN_FREQ_SCORE,
    HEAD_DOWN_SCORE, HEAD_DOWN_SCORE_CAP, YAW_DEVIATION_SCORE,
    BLINK_RATE_SCORE, BLINK_DUR_SCORE,
    MICRO_SLEEP_SEC, FATIGUE_DECAY_HALF_LIFE,
    YAWN_FREQ_WINDOW_SEC, YAWN_FREQ_THRESHOLD,
    LEVEL_L0, LEVEL_L1, LEVEL_L2,
)


class FatigueScorer:
    """
    疲劳评分器：多特征融合 → 连续分数 → L0~L3 等级

    事件接口（main 每帧调用 update，传入事件标志）：
        events = {
            "eye_closed_sec": float,     # 当前连续闭眼秒数（0 表示未闭眼）
            "yawn_just_completed": bool, # 本帧刚确认一次哈欠
            "perclos_over": bool,        # PERCLOS 超阈值
            "head_down": bool,           # 持续低头超时
            "yaw_dev": bool,             # 长时间偏头
            "blink_rate_over": bool,     # 眨眼频率异常（疲劳早期信号）
            "blink_dur_over": bool,      # 平均眨眼时长 / 慢眨眼异常
        }
    """

    def __init__(self, decay_half_life: float = FATIGUE_DECAY_HALF_LIFE):
        # 每秒衰减系数：score *= (0.5)^(1/half_life)
        self._decay = 0.5 ** (1.0 / max(1.0, decay_half_life))

        # 分数源：状态源带累积速率（分/秒），事件源只做一次性加分
        self._sources = {
            "eye_close":   {"score": 0.0, "rate": EYE_CLOSED_05S_SCORE / 0.5},   # 闭眼 0.5s 加 2 分
            "perclos":     {"score": 0.0, "rate": PERCLOS_OVER_SCORE / 30.0},    # 每 30s 加 3 分
            "yawn_freq":   {"score": 0.0, "rate": YAWN_FREQ_SCORE / 60.0},       # 每 60s 加 3 分
            "head_down":   {"score": 0.0, "rate": HEAD_DOWN_SCORE / 2.0},        # 每 2s 加 2 分
            "yaw_dev":     {"score": 0.0, "rate": YAW_DEVIATION_SCORE / 5.0},    # 每 5s 加 1 分
            "blink_rate":  {"score": 0.0, "rate": BLINK_RATE_SCORE / 60.0},      # 每 60s 加 1 分
            "blink_dur":   {"score": 0.0, "rate": BLINK_DUR_SCORE / 60.0},       # 每 60s 加 2 分
            "micro_sleep": {"score": 0.0, "rate": 0.0},                          # 事件 +4
            "yawn":        {"score": 0.0, "rate": 0.0},                          # 事件 +2
        }
        self._yawn_times: deque = deque()       # 哈欠完成时间戳（滑动窗口）
        self._prev_eye_closed_sec = 0.0         # 上一帧闭眼秒数（用于微睡眠边缘检测）

    # ── 每帧调用 ──
    def update(self, dt: float, events: dict, *, now: float | None = None,
               observed_dt: float | None = None) -> None:
        dt = max(0.0, dt)
        if observed_dt is None:
            observed_dt = dt

        # ── 状态源累积（本帧持续的状态贡献）──
        eye_closed_sec = float(events.get("eye_closed_sec", 0.0))
        now = time.monotonic() if now is None else now
        while self._yawn_times and now - self._yawn_times[0] > YAWN_FREQ_WINDOW_SEC:
            self._yawn_times.popleft()
        yawn_freq_active = events.get("face_valid", True) and len(self._yawn_times) >= YAWN_FREQ_THRESHOLD
        flags = {
            "eye_close":  0.5 <= eye_closed_sec < MICRO_SLEEP_SEC,   # 闭眼 0.5~2s 区间
            "perclos":    events.get("perclos_over", False),
            "yawn_freq":  yawn_freq_active,
            "head_down":  events.get("head_down", False),
            "yaw_dev":    events.get("yaw_dev", False),
            "blink_rate": events.get("blink_rate_over", False),
            "blink_dur":  events.get("blink_dur_over", False),
        }
        for name, active in flags.items():
            if active:
                self._sources[name]["score"] += self._sources[name]["rate"] * observed_dt

        # ── 全局指数衰减（本帧流逝的时间）──
        d = self._decay ** dt
        for s in self._sources.values():
            s["score"] *= d
        head_down = self._sources["head_down"]
        head_down["score"] = min(head_down["score"], HEAD_DOWN_SCORE_CAP)

        # ── 事件加分（本帧事件，不受本帧衰减影响，下一帧起开始衰减）──
        if events.get("yawn_just_completed"):
            self._yawn_times.append(now)
            self._sources["yawn"]["score"] += YAWN_SCORE

        # 微睡眠：闭眼时长跨过 MICRO_SLEEP_SEC 阈值瞬间触发一次（+4）
        if eye_closed_sec >= MICRO_SLEEP_SEC and self._prev_eye_closed_sec < MICRO_SLEEP_SEC:
            self._sources["micro_sleep"]["score"] += EYE_CLOSED_2S_SCORE
        self._prev_eye_closed_sec = eye_closed_sec

    # ── 查询接口 ──
    def get_score(self) -> float:
        """当前总疲劳分数（0 起，越高越疲劳）"""
        return sum(s["score"] for s in self._sources.values())

    def get_level(self) -> tuple:
        """
        返回 (等级 0~3, 文字标签)
        L0 正常 <2 | L1 轻度 <5 | L2 中度 <9 | L3 重度 ≥9
        """
        score = self.get_score()
        if score >= LEVEL_L2:
            return 3, "L3 重度疲劳"
        if score >= LEVEL_L1:
            return 2, "L2 中度疲劳"
        if score >= LEVEL_L0:
            return 1, "L1 轻度疲劳"
        return 0, "L0 正常"

    def get_active_sources(self, min_score: float = 0.05) -> list:
        """返回分数贡献 > min_score 的源 [(名称, 贡献值)]，用于可解释性显示"""
        return [(n, s["score"]) for n, s in self._sources.items() if s["score"] > min_score]

    def clear_head_down(self) -> None:
        """Drop the old posture contribution when the pitch baseline changes."""
        self._sources["head_down"]["score"] = 0.0

    def reset(self) -> None:
        """清空全部状态（换人 / 重新开始）"""
        for s in self._sources.values():
            s["score"] = 0.0
        self._yawn_times.clear()
        self._prev_eye_closed_sec = 0.0

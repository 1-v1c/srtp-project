"""
校准数据日志模块
负责写入 CSV 格式的标定记录
"""

import csv
import os
from datetime import datetime


class CalibLogger:
    """标定数据 CSV 写入器"""

    COLUMNS = [
        "timestamp",    # 记录时间
        "phase",        # EAR / MAR / HEAD
        "label",        # 标签名（如"正常睁眼"）
        "duration_s",   # 录制时长（秒）
        "frames",       # 录制帧数
        "metric",       # 指标名（EAR_L / MAR / Pitch 等）
        "min",          # 最小值
        "max",          # 最大值
        "avg",          # 平均值
    ]

    def __init__(self, log_dir: str = "data/calibration"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._filepath: str | None = None
        self._writer = None
        self._file = None

    def open_session(self) -> str:
        """开启新会话，创建带时间戳的 CSV 文件"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._filepath = os.path.join(self.log_dir, f"calib_{ts}.csv")
        self._file = open(self._filepath, "w", newline="", encoding="utf-8-sig")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.COLUMNS)
        return self._filepath

    def write_record(
        self,
        phase: str,
        label: str,
        duration_s: float,
        frames: int,
        metrics: dict,  # {"EAR_L": (min,max,avg), "MAR": (min,max,avg), ...}
    ):
        """写入一条记录（自动按指标展开多行）"""
        if self._writer is None:
            return
        ts = datetime.now().strftime("%H:%M:%S")
        for metric_name, (v_min, v_max, v_avg) in metrics.items():
            self._writer.writerow([
                ts,
                phase,
                label,
                f"{duration_s:.1f}",
                frames,
                metric_name,
                f"{v_min:.4f}",
                f"{v_max:.4f}",
                f"{v_avg:.4f}",
            ])
        self._file.flush()

    def close(self):
        if self._file:
            self._file.close()
            self._file = None
        self._writer = None

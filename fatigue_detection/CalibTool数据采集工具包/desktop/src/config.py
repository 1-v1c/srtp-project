# ============================================================
# 疲劳检测系统 — 全局配置文件
# 所有可调参数集中在这里，调参只改这个文件
# ============================================================

# ---- 摄像头 ----
CAMERA_ID = 0                 # 摄像头设备 ID（0=默认，如果有多个摄像头可能需要改）
FRAME_WIDTH = 640             # 画面宽度（降低可提升 FPS）
FRAME_HEIGHT = 480            # 画面高度
FPS_TARGET = 30               # 目标帧率（仅用于参考，实际取决于硬件）

# ---- MediaPipe Face Mesh ----
MAX_NUM_FACES = 1             # 最多检测几张脸（1 就够了）
MIN_DETECTION_CONFIDENCE = 0.5  # 初始检测置信度（0~1，越高越严格）
MIN_TRACKING_CONFIDENCE = 0.5   # 持续跟踪置信度

# ---- 眼部 EAR ----
EAR_THRESHOLD = 0.22          # EAR 低于此值判定为"闭眼"
                              # 越小 → 越不敏感（不容易误判）
                              # 越大 → 越敏感（更容易检出闭眼）
EAR_CONSECUTIVE_FRAMES = 3    # 连续多少帧闭眼才算"持续闭眼"（过滤眨眼）

# ---- 嘴部 MAR ----
MAR_THRESHOLD = 0.65          # MAR 高于此值判定为"张嘴"
                              # 越大 → 越不敏感（需要嘴张更大才触发）
MAR_YAWN_DURATION_SEC = 2.8   # 持续张嘴超过此秒数，判定为一次哈欠
MAR_YAWN_GRACE_SEC = 0.2      # 宽容缓冲：MAR 短暂跌回阈值此秒数内不计为中断

# 超过此间隔视为观测中断，避免把遮挡或卡顿计成持续闭眼。
MAX_OBSERVATION_GAP_SEC = 0.25

# ---- PERCLOS ----
PERCLOS_WINDOW_SEC = 30       # PERCLOS 统计窗口（秒）
                              # 越大 → 变化越平滑，但对突发事件反应慢
                              # 越小 → 越灵敏，但数值波动大
PERCLOS_THRESHOLD = 0.20      # 窗口内闭眼比例超过此值触发 PERCLOS 警告
                              # 0.20 = 窗口内 20% 时间闭眼
BLINK_EXEMPT_SEC = 0.3        # 闭眼持续 < 此秒数视为正常眨眼，不计入 PERCLOS
                              # （对齐文献标准：PERCLOS 排除正常眨眼）

# ---- 眨眼特征（疲劳早期信号，P0 升级）----
SLOW_BLINK_SEC = 0.5          # 眨眼持续 ≥ 此秒数视为"慢眨眼"（疲劳信号）
BLINK_WINDOW_SEC = 60.0       # 眨眼统计滑动窗口（秒）
BLINK_RATE_THRESHOLD = 25.0   # 眨眼频率 ≥ 此值（次/分）→ 频率疲劳源激活
                              # （正常 10~20 次/分，疲劳时升高）
AVG_BLINK_DUR_THRESHOLD = 0.40  # 平均眨眼时长 ≥ 此值（秒）→ 时长疲劳源激活
                              # （正常 ~0.2s，疲劳时 >0.4s）
SLOW_BLINK_RATE_THRESHOLD = 2.0 # 慢眨眼 ≥ 此值（次/分）→ 慢眨眼疲劳源激活
BLINK_RATE_SCORE = 1          # 眨眼频率异常（每 60s 加 1 分）
BLINK_DUR_SCORE = 2           # 眨眼时长 / 慢眨眼异常（每 60s 加 2 分）

# ---- 头部姿态 ----
PITCH_DOWN_THRESHOLD = 22     # Pitch 超过此角度判定为"低头"（度）
PITCH_BASELINE_DURATION_SEC = 3.0
PITCH_BASELINE_MAX_SPREAD_DEG = 6.0
PITCH_BASELINE_MAX_YAW_DEG = 15.0
PITCH_MAX_ROLL_DEG = 20.0
HEAD_DOWN_DURATION_SEC = 2.0  # 持续低头超过此秒数触发警告
YAW_DEVIATION_THRESHOLD = 30  # Yaw 偏离正前方超过此角度（度）

YAW_EAR_GATE = 25             # 侧脸时暂停使用 EAR（度）

# ---- 疲劳评分 ----
# 每个条件满足时累加的分数
EYE_CLOSED_05S_SCORE = 2      # 连续闭眼 > 0.5 秒（每秒 4 分累积）
EYE_CLOSED_2S_SCORE = 4       # 连续闭眼 > 2 秒（微睡眠，事件 +4）
PERCLOS_OVER_SCORE = 3        # PERCLOS 超过阈值（每 30s 加 3 分）
YAWN_SCORE = 2                # 检测到一次哈欠（事件 +2）
YAWN_FREQ_SCORE = 3           # 1 分钟内哈欠超过 2 次（每 60s 加 3 分）
HEAD_DOWN_SCORE = 2           # 持续低头（每 2s 加 2 分）
HEAD_DOWN_SCORE_CAP = 3.0     # 低头一项至多贡献 3 分，单独最多达到 L1
YAW_DEVIATION_SCORE = 1       # 长时间不看前方（每 5s 加 1 分）

# 评分器运行参数
MICRO_SLEEP_SEC = 2.0         # 连续闭眼超过此秒数判定为"微睡眠"事件
FATIGUE_DECAY_HALF_LIFE = 45.0  # 疲劳分数半衰期（秒）：症状停止后分数回落的速度
YAWN_FREQ_WINDOW_SEC = 60.0   # 哈欠频率统计窗口（秒）
YAWN_FREQ_THRESHOLD = 2       # 窗口内哈欠 ≥ 此数量 → 频率疲劳源激活
YAW_DEVIATION_DURATION_SEC = 2.0  # 偏头持续超过此秒数才算"长时间不看前方"

# ---- 疲劳等级（L0~L3）----
# 连续疲劳分数分级边界（分数 = 各事件源贡献之和，随时间衰减）
LEVEL_L0 = 2.0                # score < 2  → L0 正常
LEVEL_L1 = 5.0                # 2 ≤ score < 5 → L1 轻度疲劳（提示）
LEVEL_L2 = 9.0                # 5 ≤ score < 9 → L2 中度疲劳（报警）
                              # score ≥ 9 → L3 重度疲劳（强报警）

# ---- 报警 ----
ALERT_COOLDOWN_SEC = 5.0      # 两次报警最小间隔（防连续骚扰）

# ---- 日志 ----
# 统一存到 fatigue_detection/data/logs/（与标定数据 data/calibration/ 同级）
import os as _os_log
LOG_DIR = _os_log.path.join(_os_log.path.dirname(_os_log.path.abspath(__file__)), "..", "data", "logs")

# ---- 截图 ----
SCREENSHOT_DIR = _os_log.path.join(_os_log.path.dirname(_os_log.path.abspath(__file__)), "..", "data", "screenshots")

# ---- MediaPipe 模型 ----
# 注意: MediaPipe 的底层 C 库不支持中文路径，模型必须放在纯英文路径下
# 模型文件位置: E:\ObsidianVault\fatigue_detection_models\face_landmarker.task
FACE_LANDMARKER_MODEL_PATH = "E:/ObsidianVault/.codex/face_landmarker.task"

# ---- 可视化开关 ----
SHOW_FACE_MESH = True         # 是否显示 Face Mesh 绿网
SHOW_EYE_POINTS = True        # 是否标注眼部关键点
SHOW_MOUTH_POINTS = True      # 是否标注嘴部关键点
SHOW_HEAD_POSE_AXIS = True    # 是否绘制头部姿态坐标轴

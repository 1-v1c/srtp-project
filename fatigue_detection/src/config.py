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

# ---- PERCLOS ----
PERCLOS_WINDOW_SEC = 30       # PERCLOS 统计窗口（秒）
                              # 越大 → 变化越平滑，但对突发事件反应慢
                              # 越小 → 越灵敏，但数值波动大
PERCLOS_THRESHOLD = 0.20      # 窗口内闭眼比例超过此值触发 PERCLOS 警告
                              # 0.20 = 窗口内 20% 时间闭眼

# ---- 头部姿态 ----
PITCH_DOWN_THRESHOLD = 22     # Pitch 超过此角度判定为"低头"（度）
HEAD_DOWN_DURATION_SEC = 2.0  # 持续低头超过此秒数触发警告
YAW_DEVIATION_THRESHOLD = 30  # Yaw 偏离正前方超过此角度（度）

# ---- 疲劳评分 ----
# 每个条件满足时累加的分数
EYE_CLOSED_05S_SCORE = 2      # 连续闭眼 > 0.5 秒
EYE_CLOSED_2S_SCORE = 4       # 连续闭眼 > 2 秒（微睡眠）
PERCLOS_OVER_SCORE = 3        # PERCLOS 超过阈值
YAWN_SCORE = 2                # 检测到一次哈欠
YAWN_FREQ_SCORE = 3           # 1 分钟内哈欠超过 2 次
HEAD_DOWN_SCORE = 2           # 持续低头
YAW_DEVIATION_SCORE = 1       # 长时间不看前方

# ---- 疲劳等级 ----
SCORE_NORMAL = 2              # 0~2 分 → 正常
SCORE_RISK = 5                # 3~5 分 → 风险
                              # ≥6 分 → 报警

# ---- 报警 ----
ALERT_COOLDOWN_SEC = 5.0      # 两次报警最小间隔（防连续骚扰）

# ---- 日志 ----
LOG_DIR = "logs"              # CSV 日志保存目录

# ---- MediaPipe 模型 ----
# face_landmarker.task 模型文件路径
# 注意: MediaPipe 的 C 库不支持中文路径，请放到纯英文路径下
FACE_LANDMARKER_MODEL_PATH = "E:/ObsidianVault/fatigue_detection_models/face_landmarker.task"

# ---- 可视化开关 ----
SHOW_FACE_MESH = True         # 是否显示 Face Mesh 绿网
SHOW_EYE_POINTS = True        # 是否标注眼部关键点
SHOW_MOUTH_POINTS = True      # 是否标注嘴部关键点
SHOW_HEAD_POSE_AXIS = True    # 是否绘制头部姿态坐标轴

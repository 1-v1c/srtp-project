"""
制动算法模块 - brake_algorithm.py
使用驾驶员疲劳等级缩放 TTC 阈值，并与 TTB 预警时长共同决定制动等级。
"""
import math
from numbers import Integral

# ============================================================
# 配置参数
# ============================================================
class BrakeConfig:
    A_MAX = 8.0
    T_PREFILL = 0.1

    T_REACTION_BASELINE = 0.7
    FATIGUE_K = 0.25
    BRAKE_SHORT_RATIO = 0.6

    # 相对速度线性放大基准：每10km/h相对速度增加0.15倍预警时长
    VREL_STEP_SCALE = 0.15
    VREL_STEP_KMH = 10
    SCALE_MAX = 2.5        # 放大系数上限
    VREL_FULL_DISTANCE_M = 25.0
    VREL_ZERO_DISTANCE_M = 30.0
    VREL_START_CLOSING_MPS = 3.0
    VREL_FULL_CLOSING_MPS = 4.0

    # 基准TTC分级阈值(L0基准值)
    BASE_TTC_EMERGENCY = 1.2
    BASE_TTC_PARTIAL = 1.5
    BASE_TTC_WARNING = 2.0

    # 小接近速度下的纵向净距制动参数
    LOW_SPEED_MAX_CLOSING_MPS = 1.0
    LOW_SPEED_STANDSTILL_GAP_M = 1.0
    LOW_SPEED_RESPONSE_SEC = 0.30
    LOW_SPEED_DECEL_MPS2 = 1.5

    # 疲劳缩放梯度 L0/L1/L2/L3，等级越高阈值放大越多
    FATIGUE_TTC_SCALE = [1.00, 1.30, 1.65, 1.90]

    # 权重表 [疲劳L0, L1, L2, L3]
    W_CONSIST = [0.0, 0.1, 0.2, 0.3]
    W_NO_REACT = [0.85, 0.88, 0.92, 0.98]
    W_CONFLICT = [0.5, 0.65, 0.85, 1.0]


# ============================================================
# 核心算法函数
# ============================================================
def compute_brake(fatigue_level,
                  v_ego, v_target, a_target,
                  d, dx, dy,
                  heading_ego, heading_target,
                  a_driver, *,
                  bumper_gap_m=None):
    """计算有效威胁目标的制动。a_driver 为正表示制动，为负表示加速。

    bumper_gap_m 为纵向保险杠净距。旧调用省略时，低速判定沿用同车道净距 d。
    若 d 为中心距，调用方需要单独提供 bumper_gap_m。
    """
    config = BrakeConfig()
    if isinstance(fatigue_level, bool) or not isinstance(fatigue_level, Integral) or not 0 <= fatigue_level <= 3:
        raise ValueError("fatigue_level must be an integer from 0 to 3")

    # ========== 根据疲劳等级动态放大TTC阈值 ===========
    scale = config.FATIGUE_TTC_SCALE[fatigue_level]
    TTC_EMERGENCY = config.BASE_TTC_EMERGENCY * scale
    TTC_PARTIAL   = config.BASE_TTC_PARTIAL * scale
    TTC_WARNING   = config.BASE_TTC_WARNING * scale

    # ========== 1. 速度矢量分解 ==========
    ego_dir_x = math.cos(heading_ego)
    ego_dir_y = math.sin(heading_ego)
    ve_x = v_ego * ego_dir_x
    ve_y = v_ego * ego_dir_y
    vt_x = v_target * math.cos(heading_target)
    vt_y = v_target * math.sin(heading_target)

    dv_x = ve_x - vt_x
    dv_y = ve_y - vt_y
    v_rel_mag = math.hypot(dv_x, dv_y)

    # ========== 2. 接近速率 closing_speed【已修正符号】 ==========
    if d < 1e-6:
        closing_speed = 0.0
    else:
        closing_speed = (dv_x * dx + dv_y * dy) / d

    # ========== 3. TTC计算：仅靠近时有效 ==========
    if closing_speed > 0.1:
        ttc = d / closing_speed
    else:
        ttc = 999.0

    # ========== 4. 航向角 & 运动类型 ==========
    angle_diff = heading_ego - heading_target
    while angle_diff > math.pi:
        angle_diff -= 2 * math.pi
    while angle_diff < -math.pi:
        angle_diff += 2 * math.pi
    motion_type = "comoving" if abs(angle_diff) < math.pi / 2 else "oncoming"

    # ========== 5. TTB计算 + 数值稳定性保护 ==========
    t_reaction = config.T_REACTION_BASELINE * (1 + config.FATIGUE_K * fatigue_level)
    ttb = 0.0
    if motion_type == "comoving":
        a_target_abs = abs(a_target) if a_target < 0 else 0
        denom = config.A_MAX - a_target_abs
        if denom > 0.1 and closing_speed > 0.1:
            sqrt_term = closing_speed ** 2 + 2 * denom * d
            if sqrt_term >= 0:
                ttb = (-closing_speed + math.sqrt(sqrt_term)) / denom
            else:
                ttb = 999.0
        else:
            # 无法利用前车减速度，降级仅考虑自车最大制动
            sqrt_term = v_ego ** 2 + 2 * config.A_MAX * d
            if sqrt_term >= 0:
                ttb = (-v_ego + math.sqrt(sqrt_term)) / config.A_MAX
            else:
                ttb = 999.0
    else:
        # 相向场景，不考虑目标加速度
        sqrt_term = v_ego ** 2 + 2 * config.A_MAX * d
        if sqrt_term >= 0:
            ttb = (-v_ego + math.sqrt(sqrt_term)) / config.A_MAX
        else:
            ttb = 999.0
    ttb = max(ttb, 0)
    t_warning = config.T_PREFILL + t_reaction + ttb

    # ========== 6. 距离和接近速度连续过渡的预警放大 ==========
    distance_weight = (config.VREL_ZERO_DISTANCE_M - d) / (
        config.VREL_ZERO_DISTANCE_M - config.VREL_FULL_DISTANCE_M
    )
    closing_weight = (closing_speed - config.VREL_START_CLOSING_MPS) / (
        config.VREL_FULL_CLOSING_MPS - config.VREL_START_CLOSING_MPS
    )
    distance_weight = min(1.0, max(0.0, distance_weight))
    closing_weight = min(1.0, max(0.0, closing_weight))
    full_scale = min(
        1.0 + (v_rel_mag * 3.6 / config.VREL_STEP_KMH) * config.VREL_STEP_SCALE,
        config.SCALE_MAX,
    )
    t_warning *= 1.0 + distance_weight * closing_weight * (full_scale - 1.0)

    # ========== 7. AEB期望制动强度分级【方案2核心修改】 ==========
    ratio_emo = TTC_EMERGENCY / TTC_WARNING
    ratio_par = TTC_PARTIAL / TTC_WARNING
    t_emo_line = t_warning * ratio_emo
    t_par_line = t_warning * ratio_par

    # 核心：取理论刹停基准t_warning与疲劳放大阈值的较大值作为触发边界
    trigger_threshold = max(t_warning, TTC_WARNING)
    warning = (ttc < trigger_threshold)

    if ttc < t_emo_line:
        a_AEB = config.A_MAX
        brake_level = 3
    elif ttc < t_par_line:
        a_AEB = config.A_MAX * 0.75
        brake_level = 2
    elif ttc < trigger_threshold:
        a_AEB = config.A_MAX * 0.68
        brake_level = 1
    else:
        a_AEB = 0.0
        brake_level = 0
        # 同向或静止前车在低接近速度下仍可能耗尽净距。
        forward_distance = dx * ego_dir_x + dy * ego_dir_y
        longitudinal_closing = dv_x * ego_dir_x + dv_y * ego_dir_y
        gap = d if bumper_gap_m is None else bumper_gap_m
        if (
            forward_distance > 0
            and (motion_type == "comoving" or v_target == 0)
            and 0 < longitudinal_closing <= config.LOW_SPEED_MAX_CLOSING_MPS
            and math.isfinite(gap)
        ):
            distance_threshold = (
                config.LOW_SPEED_STANDSTILL_GAP_M
                + longitudinal_closing * config.LOW_SPEED_RESPONSE_SEC
                + longitudinal_closing ** 2 / (2 * config.LOW_SPEED_DECEL_MPS2)
            )
            if gap <= distance_threshold:
                a_AEB = config.LOW_SPEED_DECEL_MPS2
                brake_level = 1
                warning = True

    # ========== 8. 驾驶员意图判定 ==========
    driver_brake = max(0.0, a_driver)
    brake_not_enough = warning and (driver_brake < a_AEB * config.BRAKE_SHORT_RATIO)

    if a_driver < 0:
        intent_type = "conflict"
        intent_conflict = True
    elif driver_brake >= 0.5:
        intent_type = "conflict" if brake_not_enough else "consist"
        intent_conflict = brake_not_enough
    else:
        intent_type = "no_react"
        intent_conflict = False

    # ========== 9. 人车减速度融合逻辑 ==========
    if intent_type == "consist":
        w = config.W_CONSIST[fatigue_level]
    elif intent_type == "no_react":
        w = config.W_NO_REACT[fatigue_level]
    else:
        w = config.W_CONFLICT[fatigue_level]

    if driver_brake < 0.1:
        # 驾驶员踩油门或制动处于死区时，制动输出取 AEB 需求。
        a_final = a_AEB
    elif driver_brake > a_AEB:
        a_final = driver_brake
    else:
        a_final = w * a_AEB + (1 - w) * driver_brake

    a_final = min(a_final, config.A_MAX)

    return {
        'a_final': round(a_final, 3),
        'w': round(w, 2),
        'brake_level': brake_level,
        'warning': warning,
        'ttc': round(ttc, 3),
        't_warning': round(t_warning, 3),
        'motion_type': motion_type,
        'intent_conflict': intent_conflict,
        'a_AEB': round(a_AEB, 3),
        't_reaction': round(t_reaction, 3),
        'closing_speed': round(closing_speed, 3),
        'TTC_WARNING_thr': round(TTC_WARNING, 3),
        'TTC_PARTIAL_thr': round(TTC_PARTIAL, 3),
        'TTC_EMERGENCY_thr': round(TTC_EMERGENCY, 3),
        'trigger_threshold': round(trigger_threshold, 3)
    }

def update_brake_result(box):
    """
    从数据盒子读数据 → 调用核心算法 → 把结果写回盒子
    由CARLA主循环每帧调用一次
    """
    # target_valid 由上游设置，表示目标已确认为自车行驶路径上的威胁。
    if not box.target_valid:
        box.a_final = min(BrakeConfig.A_MAX, box.a_driver) if box.a_driver >= 0.1 else 0.0
        box.warning = False
        box.brake_level = 0
        box.ttc = 999.0
        return

    result = compute_brake(
        fatigue_level=box.fatigue_level,           # 来自疲劳检测同学
        v_ego=box.v_ego,                           # 来自CARLA仿真同学
        v_target=box.v_target,                     # 来自障碍物识别同学
        a_target=box.a_target,                     # 来自障碍物识别同学
        d=box.d,                                   # 来自障碍物识别同学
        dx=box.dx,
        dy=box.dy,
        heading_ego=box.heading_ego,               # 来自CARLA仿真同学
        heading_target=box.heading_target,         # 来自障碍物识别同学
        a_driver=box.a_driver,                     # 来自CARLA仿真同学
        bumper_gap_m=getattr(box, 'bumper_gap_m', None),
    )
    
    # 2. 把结果写回盒子
    box.a_final = result['a_final']                # 车辆最终期望制动减速度
    box.warning = result['warning']                # 预警触发标志
    box.brake_level = result['brake_level']        # 制动强度等级 0/1/2/3
    box.ttc = result['ttc']                        # 实时碰撞时间


# 本地自测
if __name__ == "__main__":
    print("==== AEB 方案2 疲劳自适应TTC 自测 ====")
    res = compute_brake(
        fatigue_level=0,
        v_ego=50/3.6,
        v_target=20/3.6,
        a_target=0,
        d=25,
        dx=25,
        dy=0,
        heading_ego=0,
        heading_target=0,
        a_driver=0
    )
    for k, v in res.items():
        print(f"{k}: {v}")

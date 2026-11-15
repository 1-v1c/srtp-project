"""
制动算法模块 - brake_algorithm.py
升级修改：
1. 增加疲劳等级对TTC阈值缩放，疲劳越高，预警阈值越大
2. 采用方案2：trigger_threshold = max(t_warning, TTC_WARNING)，保留TTB原有架构，实现疲劳工况AEB更早介入
原有全部基础逻辑、接口保持不变
"""
import math

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

    # 基准TTC分级阈值(L0基准值)
    BASE_TTC_EMERGENCY = 1.2
    BASE_TTC_PARTIAL = 1.5
    BASE_TTC_WARNING = 2.0

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
                  a_driver):
    config = BrakeConfig()

    # ========== 根据疲劳等级动态放大TTC阈值 ==========
    scale = config.FATIGUE_TTC_SCALE[fatigue_level]
    TTC_EMERGENCY = config.BASE_TTC_EMERGENCY * scale
    TTC_PARTIAL   = config.BASE_TTC_PARTIAL * scale
    TTC_WARNING   = config.BASE_TTC_WARNING * scale

    # ========== 1. 速度矢量分解 ==========
    ve_x = v_ego * math.cos(heading_ego)
    ve_y = v_ego * math.sin(heading_ego)
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

    # ========== 6. 相对速度预警时长放大（带上限） ==========
    v_rel_kmh = v_rel_mag * 3.6
    if closing_speed > 3.0 and d < 30.0:
        scale_v = 1.0 + (v_rel_kmh / config.VREL_STEP_KMH) * config.VREL_STEP_SCALE
        scale_v = min(scale_v, config.SCALE_MAX)
        t_warning = t_warning * scale_v

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

    # ========== 8. 驾驶员意图判定 ==========
    driver_brake_abs = abs(a_driver)
    brake_not_enough = warning and (driver_brake_abs < a_AEB * config.BRAKE_SHORT_RATIO)

    if a_driver < 0:
        intent_type = "conflict"
        intent_conflict = True
    elif driver_brake_abs >= 0.5 and not brake_not_enough:
        intent_type = "consist"
        intent_conflict = False
    elif driver_brake_abs >= 0.5 and brake_not_enough:
        intent_type = "conflict"
        intent_conflict = True
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

    if abs(a_driver) < 0.1:
        # 驾驶员无操作，AEB直接输出期望制动
        a_final = a_AEB
    elif driver_brake_abs > a_AEB:
        # 驾驶员制动力度大于AEB需求，信任驾驶员
        a_final = driver_brake_abs
    elif a_driver < 0:
        # 驾驶员踩油门，仅使用AEB制动力
        a_final = a_AEB
    else:
        # 驾驶员制动但力度不足，加权协同
        a_final = w * a_AEB + (1 - w) * driver_brake_abs

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
    # 1. 从盒子读取所有人写入的数据
    result = compute_brake(
        fatigue_level=box.fatigue_level,           # 来自疲劳检测同学
        v_ego=box.v_ego,                           # 来自CARLA仿真同学
        v_target=box.v_target,                     # 来自障碍物识别同学
        a_target=box.a_target,                     # 来自障碍物识别同学
        d=box.d,                                   # 来自障碍物识别同学
        heading_ego=box.heading_ego,               # 来自CARLA仿真同学
        heading_target=box.heading_target,         # 来自障碍物识别同学
        a_driver=box.a_driver                      # 来自CARLA仿真同学
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
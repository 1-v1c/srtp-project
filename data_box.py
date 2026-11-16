class DataBox:
    def __init__(self):
        # ====== 疲劳检测模块输出 =====
        self.fatigue_level = 0          # 疲劳等级 0/1/2/3 (L0清醒~L3重度疲劳)

        # ====== 障碍物信息 =====
        self.v_target = 0.0             # 障碍物速度 (m/s)
        self.a_target = 0.0             # 障碍物加速度 (m/s²，负值为减速)
        self.d = 999.0                  # 两车直线距离 (m)
        self.dx = 999.0                 # 纵向相对距离 (m)
        self.dy = 0.0                   # 横向相对距离 (m)
        self.heading_target = 0.0       # 障碍物航向角 (rad)
        self.target_valid = False       # 上游确认目标威胁自车行驶路径后设为 True，目标丢失时清零

        # ====== 自车信息 =====
        self.v_ego = 0.0                # 自车速度 (m/s)
        self.heading_ego = 0.0          # 自车航向角 (rad)
        self.a_driver = 0.0             # 驾驶员纵向操作 (m/s²)，正值为制动，负值为加速

        # ====== 制动算法模块最终输出 =====
        self.a_final = 0.0              # 车辆最终期望制动减速度 (m/s²)
        self.warning = False            # 预警触发标志（联合声光提示）
        self.brake_level = 0            # 制动强度等级 0/1/2/3
        self.ttc = 999.0                # 实时碰撞时间 (s)

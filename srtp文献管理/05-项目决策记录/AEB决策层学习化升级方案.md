---
tags:
  - type/project-decision
  - topic/srtp
  - topic/AEB
  - topic/深度学习
  - topic/模仿学习
  - topic/强化学习
  - topic/CARLA
  - topic/疲劳检测
created: 2026-06-15
updated: 2026-06-16
---

# AEB 决策层学习化升级方案

> **背景**：当前 SRTP 项目的 AEB 决策层采用「疲劳等级 → TTC 阈值」的手写查表映射。本文档讨论将其替换为基于模仿学习的决策网络，作为项目后续升级方向。
>
> **2026-06-16 更新**：DMS 疲劳输出从 L0~L3 升级为连续 fatigue_score [0,1]（§1.5）；新增 IL vs DRL 分阶段策略，V1=IL 基线、V2=RL 优化（§2.5）。

---

## 0. 现状回顾

### 当前决策逻辑（已更新）

> **更新于 2026-06-16**：DMS 端疲劳输出已从 L0~L3 离散等级升级为连续疲劳分数 fatigue_score [0,1]。

```
fatigue_score [0,1] → TTC 阈值查表（4 档）
  0.00-0.35  → TTC阈=1.5s → 较晚制动
  0.35-0.55  → TTC阈=2.0s → 正常制动
  0.55-0.75  → TTC阈=2.5s → 较早制动
  0.75-1.00  → TTC阈=3.0s → 保守制动
```

**优点**：
- 可解释、可调试
- 无需训练数据
- 适合 SRTP 第一版原型

**局限**：
- 仅考虑了疲劳等级，忽略了交通密度、天气、目标类型等上下文
- 阈值需要手工调参，无法自适应
- 4 档离散输出，无法处理边缘情况
- 不具备学习能力，天花板低

### 当前项目已有的基础设施

| 模块 | 状态 | 说明 |
|------|------|------|
| CARLA 仿真 | ✅ 已部署 | 支持多场景、多天气 |
| DMS 疲劳检测 | ✅ MediaPipe 路线 | 输出连续疲劳分数 fatigue_score [0,1]（已从 L0~L3 升级） |
| AEB 感知 | ✅ YOLOv11n + 视觉-雷达融合 | 输出目标类别+距离+速度+TTC |
| TCP 通信 | ✅ | DMS→AEB 实时传输 |
| 多场景测试 | ✅ | 3 场景 × 2 天气 + 消融实验 |

---

## 1. 升级目标

**不改动已有模块，只在 AEB 决策层插入一个可学习组件。**

| | 当前 | 升级后 |
|---|------|--------|
| 输入维度 | 1 维（连续疲劳分数 0.0~1.0） | 7+ 维（疲劳分数+PERCLOS+环境+交通） |
| 输出形式 | 疲劳分数 → TTC 阈值查表（4 档） | 连续制动强度 + 预警时机 |
| 决策方式 | 查表映射 | 模仿学习（IL），V2 可升级为强化学习（RL） |
| 泛化能力 | 无 | 可推广到训练中未见过的场景组合 |

---

## 1.5 DMS 输出升级：从离散等级到连续疲劳分数

> **决策日期**：2026-06-16
> **依据**：文献综述 [[2026-06-11 SRTP疲劳检测文献阅读重点结论]] 和 Hassan et al. (2026) DMS 综述均推荐连续输出优于离散等级。

### 1.5.1 为什么弃用 L0-L3

离散等级的固有问题：

- **信息压缩丢失**：PERCLOS = 0.21 和 PERCLOS = 0.38 都被归为 L2，但实际危险程度差近一倍
- **边界刚性**：L1/L2 边界附近的状态抖动导致同一驾驶员前后两帧被归为不同等级
- **与学习型 AEB 不兼容**：离散等级做网络输入时丢失了疲劳程度的细粒度信息

### 1.5.2 新方案：多特征线性加权 fatigue_score

**架构**：

```
摄像头 → MediaPipe Face Mesh
  ├── EAR 持续闭眼时长  → normalize → w₁ = 0.30
  ├── PERCLOS             → [0,1]    → w₂ = 0.25
  ├── MAR 哈欠频率        → normalize → w₃ = 0.15
  ├── HeadPose 低头角度   → normalize → w₄ = 0.15
  └── HeadPose 偏头角度   → normalize → w₅ = 0.10
                                          ↓ 加权求和
                                    fatigue_score [0,1]
```

**每个特征的归一化**：

```python
# 连续闭眼时长 → [0,1]，2s以上=1.0
def normalize_eye_closed(duration_sec):
    if duration_sec < 0.15: return 0.0
    return min(duration_sec / 2.0, 1.0)

# 哈欠频率 → [0,1]，3次/min以上=1.0
def normalize_yawn(count_per_min):
    return min(count_per_min / 3.0, 1.0)

# 低头角度 → [0,1]，<10°=0, >30°=1
def normalize_pitch(pitch_deg):
    if pitch_deg < 10: return 0.0
    return min((pitch_deg - 10) / 20.0, 1.0)
```

**输出含义**（仅做参考区间，实际传给 AEB 的是连续值）：

| fatigue_score | 状态 | 参考行动 |
|:---:|------|------|
| 0.00-0.15 | 完全清醒 | 无干预 |
| 0.15-0.35 | 轻度疲劳 | 记录不报警 |
| 0.35-0.55 | 明显疲劳 | 预警 |
| 0.55-0.75 | 严重疲劳 | 报警 + AEB 提前 |
| 0.75-1.00 | 极端危险 | 紧急制动 |

### 1.5.3 对 AEB 升级的意义

连续 fatigue_score 直接作为决策网络输入，替换原来的离散等级。AEB 不再只知道「疲劳 L2」，而是知道疲劳程度具体是 0.62——可以学习更精细的制动策略。

```python
# 旧：输入离散等级
state = [fatigue_level]  # 0, 1, 2, 3

# 新：输入连续分数 + 原始特征
state = [
    fatigue_score,   # 融合后的连续分数 [0,1]
    perclos,         # 单独传入，让网络自行学习
    ttc,             # 碰撞时间
    ego_speed,       # 自车速度
    lead_distance,   # 前车距离
    target_type,     # 目标类型
    weather          # 天气
]
```

**好处**：
- AEB 可以学习 PERCLOS 和 fatigue_score 各自的独立贡献（消融实验有东西可做）
- 后续个性化只需调整归一化参数，网络不用重新训练
- 查表版本也可以直接用 fatigue_score 做更细粒度的映射（过渡方案）

---

## 2. 技术方案

### 2.1 决策网络结构

```python
# 轻量 MLP，~500 参数，CPU 可推理
class DecisionNet(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=32, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
            nn.Sigmoid()  # 输出归一化到 [0,1]
        )

    def forward(self, x):
        # x: [fatigue_score, PERCLOS, TTC, 自车速度, 前车距离,
        #      目标类型编码, 天气编码, 交通密度]
        return self.net(x)
```

### 2.2 输入特征设计

| 特征 | 取值范围 | 来源 | 说明 |
|------|----------|------|------|
| **fatigue_score** | 0.0 ~ 1.0（多特征加权融合） | DMS（TCP 传输） | 替代离散等级，连续值提供细粒度疲劳信息 |
| **PERCLOS** | 0.0 ~ 1.0（原始值） | DMS | 单独传入，让网络学习其独立贡献 |
| TTC | 0 ~ 10s | AEB 感知模块 | 碰撞时间 |
| 自车速度 | 0 ~ 120 km/h（归一化） | CARLA API | 影响制动距离 |
| 前车距离 | 0 ~ 100m（归一化） | 视觉-雷达融合 | 直接距离信息 |
| 目标类型 | one-hot: [车, 行人, 骑行] | YOLOv11n | 不同类型安全距离不同 |
| 天气编码 | one-hot: [晴, 雨, 夜] | CARLA API | 影响路面附着系数 |
| 交通密度 | 0.0 ~ 1.0 | CARLA API（附近车辆数归一化） | 变道空间可用性 |

> 输入从 7 维开始。保留 PERCLOS 原始值作为独立特征，使 AEB 可以同时学习融合分数和单一特征各自的贡献——这为 §3.2 消融实验提供了额外维度。

### 2.3 训练方式：行为克隆（Behavior Cloning）

选择 IL 而非 RL 作为第一版训练方式的原因详见 §2.5。此处描述 IL 方案。

**数据采集流程**：

```
1. CARLA 启动，加载测试场景（高速公路 / 城市 / 行人横穿）
2. DMS 实时输出 fatigue_score [0,1]（而非离散等级）
3. 手动驾驶并做出制动决策
4. 每 0.1 秒自动记录一条样本：

   {
     "input": [fatigue_score, PERCLOS, TTC, 车速, 前车距离, 目标类型, 天气],
     "label": [制动强度, 是否预警]  ← 你的手动决策
   }

5. 采集 30 分钟 → ~18000 条样本 → 8:2 划分训练/测试集
```

> 相比旧方案（按键模拟注入 L0~L3），新的数据采集流程中 fatigue_score 由 DMS 实时计算，更接近真实部署场景，且不需要手动切换疲劳等级。

**训练**：

```
Loss: MSE（制动强度）+ Binary Cross Entropy（预警）
Optimizer: Adam, lr=0.001
Epochs: 50~100
Batch Size: 64
正则化: Dropout(0.2) + Weight Decay(1e-4)
```

### 2.4 部署方式

```python
# 替换原有查表函数
# 之前：
def get_ttc_threshold(fatigue_level):
    table = {0: 1.5, 1: 2.0, 2: 2.5, 3: 3.0}
    return table[fatigue_level]

# 之后：
def get_brake_decision(state_vector):
    with torch.no_grad():
        brake_intensity, warning_flag = model(state_vector)
    return brake_intensity, warning_flag
```

---

## 2.5 IL vs DRL 方案对比与分阶段策略

> **决策日期**：2026-06-16
> **结论**：V1 用 IL 出基线，V2 用 RL 优化。两阶段共享同一网络架构，不浪费前期工作。

### 2.5.1 RL 的定位

RL 替代的是 AEB 决策模块的训练方式，不是疲劳检测模块：

```
疲劳检测模块（DMS，不变）
    ↓ 输出 fatigue_score [0,1]
AEB 决策模块 ← 训练方式可选：
    方案 A（IL / 行为克隆）：人开 → 记录 → 模仿
    方案 B（DRL / 强化学习）：agent 自博弈 → 撞了受罚/安全受奖
```

### 2.5.2 DRL 相比 IL 的优势

**1. 可能超越人类驾驶员**

IL 的上限 = 手动驾驶水平。你的制动习惯不一定是最优的（偏保守 → 不必要急刹）。DRL agent 通过大量试错可能找到比人类更好的策略。

**2. 奖励函数天然支持多目标优化**

```python
reward = (
    -10.0 * collision          # 碰撞：严重惩罚
    - 0.5 * unnecessary_brake  # 不必要刹车：轻微惩罚
    - 0.1 * jerk               # 急刹：舒适度惩罚
    + 1.0 * safe_arrival       # 安全到达：奖励
)
```

IL 无法做到——人类驾驶数据里安全、舒适、效率是混在一起的，分不开。RL 的奖励函数可以对每个维度独立赋权。

**3. 论文厚度更高**

DRL + 连续疲劳分数 + CARLA → 完整故事线。可以设计消融实验：去掉奖励函数某一项会怎样、不同疲劳程度的学习曲线、IL 初始网络 + RL 微调的效果对比等。

### 2.5.3 DRL 相比 IL 的劣势

**1. 样本效率极低（最大实际障碍）**

```
IL：30 分钟数据 ≈ 18,000 条 → 训练 50 epoch → 几十分钟
DRL：PPO agent 约需 100,000+ 步 → 1.4 小时纯仿真 + 训练时间
```

保守估计 DRL 总工作量 3-5 倍于 IL。奖励函数调偏一次 → 半天白跑。

**2. 奖励函数设计的隐性成本**

- fatigue_score 从 0.3 到 0.5 时，agent 需要学到「制动时机提前多少」——这个映射不是线性的，agent 能不能学到不确定
- 疲劳与制动间存在几十秒到几分钟的延迟反馈。t=0 时 fatigue_score=0.3 未制动，t=120 时撞了——agent 要把惩罚归因到两分钟前的决策，这比即时反馈的 Atari 游戏难得多
- 安全约束无法编码进普通奖励函数。RL 社区花了十几年解决 Safe RL（你们文档里标 ⭐⭐⭐⭐ 的方向）

**3. 调试耦合度高**

IL 效果不好 → 查哪条数据标错了，修一修重跑。

DRL 效果不好 → 是 RL 算法问题？还是奖励函数不对？还是探索策略有问题？还是 CARLA 有 bug？还是疲劳信号本身不准？四个环节的交互需要同时排查。

**4. 本科生答辩风险**

评审老师会追问：为什么不先用 IL 做基线？奖励函数为什么这么设计？agent 学到的策略和人类驾驶策略有什么不同，哪个更好？每个问题都需要扎实实验回答。目前 fatigue_score 输出还不稳定（任务 05-09 未完成），此时把不稳定的信号喂给敏感的 DRL agent，出问题都找不到根因。

### 2.5.4 推荐的分阶段策略

**同一网络架构，两种训练方式，分两个阶段推进：**

```python
# DecisionNet 的 forward 保持不变，只换 training loop
class DecisionNet(nn.Module):
    def forward(self, state):
        return brake_intensity, warning_flag

# V1: IL 训练
model = DecisionNet()
for state, action in human_driving_data:
    loss = mse(model(state), action)
    loss.backward()

# V2: RL 训练（同一网络，不同训练循环）
model = DecisionNet()
ppo = PPO(model, lr=3e-4)
for episode in range(n_episodes):
    state = env.reset()
    while not done:
        action = model(state)
        next_state, reward, done = env.step(action)
        ppo.update(state, action, reward, next_state, done)
```

**各版本定位**：

| 版本 | 训练方式 | 干什么 | 工作量 | 风险 |
|------|------|------|:--:|:--:|
| **V1** | IL（行为克隆） | 跑通完整 pipeline，拿到基线结果 | 15-20h | 低 |
| **V2** | DRL（PPO） | 替换训练方式，同一网络，对比改进 | V1 基础上 +30h | 中 |
| **V3** | Safe RL + CBF | 加安全约束，保证策略安全 | 长期 | 高 |

**论文叙述策略**：

> "我们首先用行为克隆训练一个初始决策网络作为基线（§4.1），然后引入 PPO 强化学习对策略进行在线优化（§4.2），实验表明……（§5.3）。两种训练方式共享同一网络架构，便于公平对比。"

### 2.5.5 对比总结

| 维度 | IL（行为克隆） | DRL（强化学习） |
|------|:--:|:--:|
| 所需数据 | 30 分钟人开车 | 几万回合自博弈 |
| 训练时间 | 几十分钟 | 几小时到几天 |
| 策略上限 | 人类水平 | 可能超越人类 |
| 调试难度 | 低 | 高 |
| 答辩风险 | 低 | 中高 |
| 论文厚度 | 中等 | 高 |
| **适合阶段** | **现在（V1）** | **V2（如果时间充裕）** |

---

### 3.1 对比实验

| 实验 | 对照组 | 实验组 | 评估指标 |
|------|--------|--------|----------|
| 查表 vs IL | 固定 TTC 查表 | 学习型决策 | 碰撞率、误制动率、制动舒适度 |
| 二分类 vs 多级 | 仅输出刹/不刹 | 连续制动强度 | 制动平滑度 |
| 不同疲劳等级 | — | — | 各等级下的决策合理性 |
| 训练集外场景 | 训练: 晴+城市 | 测试: 雨+高速 | 泛化能力 |

### 3.2 消融实验

| 移除的输入 | 预期影响 | 验证假设 |
|-----------|----------|----------|
| 移除 fatigue_score | 碰撞率上升 | 融合疲劳分数对决策有用 |
| 移除 PERCLOS（仅保留 fatigue_score） | 部分场景安全性下降 | PERCLOS 提供 fatigue_score 之外的额外信息 |
| 移除天气 | 雨天制动距离不足 | 天气影响安全裕度 |
| 移除目标类型 | 行人场景安全性下降 | 不同目标需要不同策略 |
| 移除交通密度 | 高密度下误制动增多 | 密度影响制动时机选择 |

**V2 阶段额外消融**：

| 实验 | 对照组 | 实验组 | 验证问题 |
|------|--------|--------|----------|
| IL vs RL | IL 策略 | RL 微调策略 | RL 能否超越人类演示？ |
| 奖励函数消融 | 完整 reward | 去掉 comfort / efficiency 项 | 多目标是否互相牵制？ |
| fatigue_score 离散 vs 连续 | 输入 L0-L3 的网络 | 输入 fatigue_score 的网络 | 连续输入是否提升策略质量？ |

### 3.3 评估指标

| 指标 | 定义 | 目标 |
|------|------|------|
| 碰撞率 | 发生碰撞的测试次数/总测试次数 | IL ≤ 查表 |
| 误制动率 | 不必要制动的次数（TTC>3s 时触发） | IL < 查表 |
| 制动舒适度 | 最大 Jerk 值的均值 | IL < 查表 |
| 决策延迟 | 从疲劳变化到制动策略调整的时间 | < 100ms |

---

## 4. 预期创新点

做完这个升级后，论文的贡献可以写成：

> 1. **We propose a lightweight imitation-learned decision module** that replaces fixed threshold-based TTC mapping with a context-aware multi-factor policy, enabling graduated braking intervention based on driver fatigue state and environmental context.
> 2. **We demonstrate via CARLA simulation** that the learned policy reduces unnecessary braking while maintaining equivalent collision avoidance compared to the baseline table-lookup approach, especially under unseen weather and traffic conditions.
> 3. **We provide ablation studies** quantifying the individual contribution of fatigue level, weather, target type, and traffic density to braking safety.

---

## 5. 工作量估算

| 任务                     | 工作量    | Agent 可辅助比例 |
| ---------------------- | ------ | :---------: |
| 数据采集脚本（CARLA 里记录状态+动作） | ~150 行 |     90%     |
| 决策网络定义 + 训练脚本          | ~100 行 |     90%     |
| 手动驾驶采集 30 分钟数据         | 1-2 小时 |  0%（必须人开）   |
| 训练 + 调参                | 2-4 小时 |     60%     |
| 评估脚本（对比实验+消融实验）        | ~200 行 |     85%     |
| 画图 + 分析                | 2-4 小时 |     50%     |

**总新写代码量**：~500 行
**总人时**：约 15-20 小时（分散在 2-3 周）
**侵入性**：仅替换 AEB 端一个函数，不影响 DMS 和通信模块

---

## 6. 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|:---:|------|------|
| 30 分钟数据不够，网络欠拟合 | 中 | 决策质量差 | 数据增强（加噪声）+ 多采集几个场景 |
| MLP 太简单，效果不如查表 | 低 | 实验失败 | 换 LSTM 处理时序、增大网络、或混合方案（查表+MLP残差） |
| 手动驾驶数据质量差 | 中 | 网络学到坏习惯 | 多采集几轮，用一致性筛选 |
| 和现有 AEB 代码冲突 | 低 | 联调困难 | 决策函数独立封装，两个版本并存（通过配置切换） |

---

## 7. 后续延展方向

> **更新于 2026-06-16**：RL 已从「后续延展」提升为 V2 正式方案。

本方案完成后，可沿以下方向继续深化：

| 方向 | 说明 | 难度 | 与当前方案关系 |
|------|------|:---:|------|
| **RL 替代 IL（V2）** | 用 PPO/SAC 在线学习，同一网络架构，见 §2.5 | ⭐⭐⭐ | V1 的直接升级 |
| IL→RL 迁移 | 用 IL 训练的网络作为 RL 初始策略（warm-start），减少探索成本 | ⭐⭐ | V1+V2 衔接 |
| Safe RL | 加入 Control Barrier Function 保证安全 | ⭐⭐⭐⭐ | V3 长远方向 |
| 多特征权重学习 | 将线性加权改为可学习的注意力融合，替代手工设定 w₁-w₅ | ⭐⭐ | DMS 端升级 |
| 扩展到变道靠边 | 输出空间从制动扩展到横向控制 | ⭐⭐⭐⭐ | AEB 端扩展 |
| World Model | 用 CarDreamer 框架在 World Model 中训练 | ⭐⭐⭐⭐⭐ | 前沿探索 |
| 端到端联合优化 | DMS 检测 + AEB 决策联合训练 | ⭐⭐⭐⭐⭐ | 终极目标 |

---

## 8. 相关文档

- 疲劳检测方案：[[疲劳判定模块问题清单]]
- 文献阅读结论：[[2026-06-11 SRTP疲劳检测文献阅读重点结论]]
- DMS 综述阅读笔记：[[论文阅读-2026-AI-Review-DMS综述]]
- 本周文献指南：[[本周文献阅读指南]]
- 今日讨论记录：[[2026-06-15]]
- 最新讨论记录：[[2026-06-16]]

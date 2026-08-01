# 疲劳检测模块（DMS）— Fatigue Detection

SRTP 项目「人车协同安全系统」的 DMS 端：基于 MediaPipe 的**驾驶员疲劳检测系统**。
输入：摄像头实时帧 → 输出：EAR/MAR/HeadPose/PERCLOS/眨眼动力学特征 → L0~L3 疲劳等级。

## 功能特性

- **EAR** 眼部长宽比（阈值 0.22，5 帧平滑，持续闭眼检测）
- **MAR** 嘴部长宽比（阈值 0.65，2.8s 哈欠判定 + 0.2s 宽容缓冲）
- **HeadPose** 头部姿态（solvePnP 六点法，Pitch/Yaw/Roll，PITCH_OFFSET 基线校准）
- **PERCLOS** 闭眼比例统计（30s 窗口，眨眼 <0.3s 豁免不计入）
- **眨眼动力学**（疲劳早期信号）：频率 / 平均时长 / 慢眨眼 / 闭合速度
- **疲劳评分器**（fatigue.py）：9 个分数源融合 → 连续评分 → L0~L3 状态机（45s 半衰期衰减，分数来源可解释）
- **报警 + 日志**：等级横幅（L3 红白闪烁）、分级蜂鸣、每秒 CSV 快照
- **标定工具**：CalibTool（桌面版 + 网页版），支持 EAR/MAR/HEAD 三种模式个人标定

## 模块进度

| 模块 | 状态 |
|:---|:---:|
| 01-08（脚手架/摄像头/EAR/MAR/HeadPose/PERCLOS/评分器/报警日志） | ✅ 完成 |
| P0 升级（眨眼动力学 + PERCLOS 豁免） | ✅ 完成 |
| 09 集成测试 | 🔶 单元测试 14 项 + 摄像头联调通过；阈值调优等采集数据 |

## 目录结构

```
fatigue_detection/
├── src/
│   ├── config.py        ← 全局配置（调参只改这个文件）
│   ├── camera.py        ← 摄像头管理
│   ├── detector.py      ← MediaPipe Face Mesh 封装
│   ├── features.py      ← EAR / MAR / HeadPose / PERCLOS / BlinkAnalyzer
│   ├── fatigue.py       ← 疲劳评分器（L0~L3 状态机）
│   ├── main.py          ← 主运行入口（q 退出 / s 截图 / r 重置评分）
│   ├── calibtool.py     ← 桌面版标定采集工具
│   └── logger.py        ← CSV 日志模块
├── tests/
│   └── test_core.py     ← 核心逻辑单元测试（14 项）
├── data/                ← 数据产物（不入库）
│   ├── calibration/     ← 标定 CSV
│   ├── logs/            ← 运行日志（每秒快照）
│   └── screenshots/     ← 截图
├── CalibTool数据采集工具包/  ← 交付工具包（队友用，含网页版+桌面版+实验手册）
├── calibtool_web/       ← 网页版标定工具源码
├── CLAUDE.md            ← AI 协作上下文
└── requirements.txt
```

## 安装与运行

```powershell
pip install -r requirements.txt

# 主程序（疲劳检测）
cd src
python main.py

# 单元测试
cd fatigue_detection
python tests\test_core.py

# 标定工具（桌面版）
cd src
python calibtool.py
```

**数据采集**：见 `CalibTool数据采集工具包/README.md` 与 `实验流程手册.md`（队友交付文档）。

## 关键阈值（config.py）

| 参数 | 值 | 说明 |
|:---|:---|:---|
| EAR_THRESHOLD | 0.22 | 闭眼判定 |
| MAR_THRESHOLD | 0.65 | 哈欠判定 |
| PITCH_DOWN_THRESHOLD | 22° | 低头判定 |
| PERCLOS_THRESHOLD | 0.20 | 30s 窗口闭眼比例报警 |
| 疲劳分级 | <2 / <5 / <9 / ≥9 | L0 正常 / L1 轻度 / L2 中度 / L3 重度 |
| 眨眼疲劳源 | ≥25 次/分 或 平均时长 ≥0.4s 或 慢眨眼 ≥2 次/分 | 疲劳早期信号 |

## 与 AEB 端对接

DMS 输出 `疲劳等级 (L0~L3)` 作为 AEB 决策的输入之一（疲劳调制）。
详见 `项目进度记录/2026-08-01 AEB算法快速验证方案.md` 与 `2026-08-01 AEB算法学习化升级方案.md`。

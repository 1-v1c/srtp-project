# CLAUDE.md — 疲劳检测 DMS 项目

## 项目简介
基于 MediaPipe 的**驾驶员疲劳检测系统**（SRTP 项目）。负责 DMS 端的视觉特征提取与疲劳评级。
输入：摄像头实时帧。输出：EAR / MAR / HeadPose / PERCLOS 四维特征 → L0~L3 疲劳评分。

## 技术栈
- Python 3.13 + OpenCV 5.0 + MediaPipe 0.10.35 + NumPy 2.1 + Pillow 11.1
- **无深度学习训练**，纯规则法特征工程路线
- 运行环境：Windows (D:\Anaconda3\local\python.exe)

## 目录结构
```
fatigue_detection/
├── src/
│   ├── config.py        ← 全局配置（调参只改这个文件）
│   ├── camera.py        ← 摄像头管理
│   ├── detector.py      ← MediaPipe Face Mesh 封装
│   ├── features.py      ← EAR / MAR / HeadPose / PERCLOS 特征计算
│   ├── fatigue.py       ← 疲劳评分器（L0~L3 状态机，模块 07）
│   ├── main.py          ← 主运行入口
│   ├── calibtool.py     ← 标定采集工具（独立 UI）
│   ├── logger.py        ← CSV 日志
│   └── __init__.py
├── tests/
│   └── test_core.py     ← 核心逻辑单元测试（python tests/test_core.py）
├── models/              ← 模型文件占位（实际在 .codex/）
├── data/
│   ├── calibration/     ← 标定 CSV
│   ├── screenshots/     ← 截图
│   └── logs/            ← 主程序运行日志（每秒快照）
├── CalibTool数据采集工具包/  ← 交付工具包（队友用，含网页版+桌面版+实验手册）
├── calibtool_web/       ← 网页版标定工具源码
├── CLAUDE.md            ← 本文件
└── requirements.txt
```

## 当前进度
- ✅ 01 脚手架（config + requirements）
- ✅ 02 摄像头 + FaceMesh 检测器
- ✅ 03 EAR 眼部长宽比（阈值 0.22，5 帧平滑）
- ✅ 04 MAR 嘴部长宽比（阈值 0.65，2.8s+0.2s 缓冲）
- ✅ 05 HeadPose 头部姿态（solvePnP 六点法，Pitch/Yaw/Roll）
- ✅ 06 PERCLOS 时间窗口统计（按 EAR 比例，30s 窗口，HUD 显示 + 超限报警）
- ✅ 07 疲劳评分融合 + L0~L3 状态机（fatigue.py FatigueScorer：事件源 + 状态源 + 指数衰减 + 可解释分数来源）
- ✅ 08 可视化报警 + 运行日志（等级横幅、L3 红白闪烁、蜂鸣、每秒 CSV 快照）
- ✅ P0 升级（2026-08-02）：眨眼动力学特征（频率/时长/慢眨眼/闭合速度，BlinkAnalyzer）+ PERCLOS 眨眼豁免（<0.3s 不计入）
- 🔶 09 集成测试：单元测试 14 项通过 + 真实摄像头联调通过（2026-08-02）；阈值调优等采集数据

## 重要常量和阈值
- EAR_THRESHOLD = 0.22（闭眼判定）
- MAR_THRESHOLD = 0.65（哈欠判定）
- PITCH_DOWN_THRESHOLD = 22°（低头判定）
- PERCLOS_WINDOW_SEC = 30（统计窗口，眨眼 <0.3s 豁免不计入）
- 疲劳分级：score < 2 → L0 正常 / <5 → L1 轻度 / <9 → L2 中度 / ≥9 → L3 重度
- 分数半衰期 45s；闭眼 0.5s 起累积、2s 触发微睡眠事件
- 眨眼疲劳源：频率 ≥25 次/分 或 平均时长 ≥0.4s 或 慢眨眼 ≥2 次/分 激活（每 60s +1/+2 分）

## 标定工具
`python calibtool.py` — 独立标定工具，支持 EAR / MAR / HEAD 三种模式。
快捷键：Space(录制) / Tab(下一标签) / B(上一标签) / 1/2/3(切模式) / E(导出) / H(帮助)

## 运行方式
```powershell
cd fatigue_detection\src
python main.py         # 主程序（q 退出 / s 截图 / r 重置疲劳评分）
python calibtool.py    # 标定工具
cd fatigue_detection   # 单元测试
python tests\test_core.py
```

## DO NOT — 不要做的事
- ❌ 不要在 features.py 里加入 CNN / 深度学习模块（本项目不走这路线）
- ❌ 不要修改 config.py 的模型路径格式（纯英文路径，已锁定到 .codex/）
- ❌ 不要删除 calibtool.py 的 UI 绘制代码（磨砂面板、圆角按钮等）
- ❌ 不要引入新的重量级依赖（保持项目轻量化实时运行）
- ❌ 不要重命名 src/ 下的文件（会破坏现有 import 链）
- ❌ 不要用 pip install 直接装包（必须写在 requirements.txt 里并通知我）

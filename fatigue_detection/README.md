# 疲劳检测模块（DMS）— Fatigue Detection

SRTP 项目「人车协同安全系统」的 DMS 端：基于 MediaPipe 的**驾驶员疲劳检测系统**。
输入：摄像头实时帧 → 输出：EAR/MAR/HeadPose/PERCLOS/眨眼动力学特征 → L0~L3 疲劳等级。

## 功能特性

- **EAR** 眼部长宽比（阈值 0.22，5 帧平滑，持续闭眼检测）
- **MAR** 嘴部长宽比（阈值 0.65，2.8s 哈欠判定 + 0.2s 宽容缓冲）
- **HeadPose** 头部姿态（使用上半脸参考点求解 Pitch/Yaw/Roll，正视自动归零后输出相对 Pitch）
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

### 浏览器仪表盘

Windows 双击 `启动DMS网页版.bat`，或在项目根目录运行：

```powershell
python src/main.py --web
```

服务默认在本机 `http://127.0.0.1:8765/`，摄像头画面、疲劳等级与观测数据分开显示。可通过 `--camera 0` 指定内置摄像头。未指定设备时优先使用配置里的设备，打开失败时尝试设备 0。需要分心检测时，请使用包含 YOLO 依赖和本地权重的运行环境。

页面上的「暂停监测」释放摄像头，并暂停写入日志。「继续监测」会重新取帧，沿用同一份 CSV，闭眼计时在恢复时重新开始。终端按 `Ctrl+C` 退出服务。浏览器使用等级颜色与状态提示，适合静音录屏。

开始时自然坐姿看向前方，等待「Pitch 已归零」。换人或调整座椅后使用「重新归零」，OpenCV 窗口可按 `c`。标定需连续约 3 秒的稳定姿态，暂停后继续沿用已完成的基线。低头使用相对 Pitch，单项分数最多 3 分。

### OpenCV 窗口与采集工具

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
| PITCH_DOWN_THRESHOLD | 见 `config.py` | 低头相对个人正视基线的角度阈值 |
| PERCLOS_THRESHOLD | 0.20 | 30s 窗口闭眼比例报警 |
| 疲劳分级 | <2 / <5 / <9 / ≥9 | L0 正常 / L1 轻度 / L2 中度 / L3 重度 |
| 眨眼疲劳源 | ≥25 次/分 或 平均时长 ≥0.4s 或 慢眨眼 ≥2 次/分 | 疲劳早期信号 |

头姿参考坐标取自 [MediaPipe 标准人脸模型](https://github.com/google-ai-edge/mediapipe/blob/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model.obj) 的上半脸顶点，转换为 OpenCV 坐标系。[Apache-2.0 许可](licenses/MediaPipe-Apache-2.0.txt)

## 与 AEB 端对接

DMS 输出 `疲劳等级 (L0~L3)` 作为 AEB 决策的输入之一（疲劳调制）。
详见 `项目进度记录/2026-08-01 AEB算法快速验证方案.md` 与 `2026-08-01 AEB算法学习化升级方案.md`。

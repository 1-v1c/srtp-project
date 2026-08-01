---
title: CARLA 仿真中 YOLO 障碍物检测的展示与评测方案
project: SRTP项目—人车协同安全系统
date: 2026-08-01
tags: [srtp, carla, yolo, 感知, 评测, 方案]
---

# CARLA 仿真中 YOLO 障碍物检测的展示与评测方案

> 面向 AEB/感知端队友的方案参考：如何在 CARLA 中展示训练好的 YOLO 模型效果，并进行可视化与数据化评测。

---

## 一、背景与目标

YOLO 障碍物检测模型训练完成后，需要回答两个问题：

1. **展示**：怎么把模型识别效果直观地呈现给队友/导师？（实时画面、对比视频、网页看板）
2. **评测**：怎么用数据证明模型能力？（mAP、漏检率、距离分层表现……）

CARLA 仿真的核心优势：**拥有像素级真值（GT）**，评测精度和可信度超过实车数据。

---

## 二、核心概念：像素级真值 GT

### 2.1 什么是 GT

GT（Ground Truth）= 标准答案。评测时 YOLO 预测框与 GT 框做 IoU 匹配，IoU ≥ 0.5 判为正确检出。**mAP / Precision / Recall 全部建立在 GT 之上，GT 不准则一切指标失去意义。**

### 2.2 "像素级"的含义

| 维度 | 实车数据（人工标注） | CARLA（自动生成） |
|:---|:---|:---|
| 框边缘精度 | ±1~3 像素（人眼误差） | ≈0（数学投影） |
| 漏标 | 会漏（尤其远处小目标） | 不漏（遍历全部 actor） |
| 类别错误 | 会错 | 不可能（引擎直接给出） |
| 标注速度 | 30~60 秒/张 | 实时、无限量 |
| 距离/天气元数据 | 需额外测量 | 直接读取 |

### 2.3 CARLA 获取 GT 的方式

- **车辆/行人**：`actor.bounding_box` + `get_world_vertices(transform)` → 8 个 3D 顶点投影到 2D，取 min/max 得像素级 2D 框
- **红绿灯/标志牌**：`world.get_level_bbs(carla.CityObjectLabel.TrafficLight)`（0.9.14+）
- **遮挡过滤（关键坑）**：3D 投影会产出"ghost box"（被遮挡物体的框），需用 semantic segmentation 相机验证框内目标类像素占比 ≥10% 才保留（论文方案：Precision 0.968 / Recall 0.926）

---

## 三、总体架构

```
CARLA 仿真世界
  │ 挂 RGB 相机传感器 → 每帧图像
  ▼
YOLO 推理（ultralytics）
  │ 输出 boxes + classes + conf
  ▼
├─ 支线 ①：实时可视化展示（OpenCV / 网页前端 / 仿真内 3D）
└─ 支线 ②：离线指标评测（标准指标 + 分层指标 + AEB 专项）
```

---

## 四、可视化展示方案（三级递进）

### 层级 1：OpenCV 窗口直显（最快落地，开发/调试用）

- YOLO 红框 + 真值蓝框同屏叠加，错检漏检一目了然
- `cv2.VideoWriter` 直接录对比视频
- 工作量：2~3 天

### 层级 2：网页前端流式展示（演示/汇报用，推荐）

- CARLA 端推理 → WebSocket / MJPEG 推流 → 浏览器 canvas 绘制
- 优势：
  - 左右分屏（原始 vs 检测 / YOLO vs GT）
  - 实时指标栏（检出数、类别计数、平均置信度、FPS）
  - 场景切换按钮（Town 地图 / 天气一键切换对比）
  - 队友导师浏览器打开即看，无需安装 CARLA
- 工作量：3~5 天

### 层级 3：仿真内 3D 可视化（进阶）

- `world.debug.draw_box()` 在仿真世界里画 3D 真值框，第三人称视角观察
- 或 carla-ros-bridge + RViz（ROS 标准可视化生态，配置成本高）

---

## 五、数据化指标评测

### 5.1 标准检测指标（对比基线）

```
mAP@0.5    mAP@0.5:0.95    Precision    Recall    F1
```

用 ultralytics 自带 `val.py` 即可，自动输出 PR 曲线和混淆矩阵，零开发成本。

### 5.2 CARLA 独有：条件分层评测

| 维度 | 做法 | 意义 |
|:---|:---|:---|
| 按距离分层 | 标注时记录物体距离：<25m / 25~50m / >50m | 远距离小目标掉点一目了然 |
| 按天气光照 | 同一路线跑晴天/雨天/雾天/夜晚 | 测泛化能力 |
| 按类别 | 车/行人/自行车/红绿灯分别出指标 | 定位薄弱类别 |
| 按遮挡 | 半遮挡物体召回率 | 复杂场景可靠性 |

### 5.3 AEB 场景专项指标（对接决策层）

- **漏检率 / 误报率**：漏检 = 事故源头，比 mAP 更重要
- **检出距离**：目标首次被检出的距离（直接影响制动距离）
- **TTC 内检出率**：AEB 触发时间预算（如 TTC 1.5s）内是否检到
- **时间一致性**：同一目标连续帧是否稳定检出（闪烁检测是实车部署大忌）

### 5.4 评测流程

```
① 录制：脚本控制车辆沿固定路线跑 N 遍（切换天气/场景），
   同步保存 RGB 帧 + YOLO 输出 + GT 框（含距离/遮挡元数据）→ 可复现
② 离线评测：ultralytics val.py（标准指标）+ 自写脚本（分层/AEB 专项）
③ 产出：mAP 对比表 + PR 曲线 + 混淆矩阵 + 分层表 + demo 视频
```

---

## 六、落地路线建议

| 阶段 | 内容 | 工作量 |
|:---:|:---|:---:|
| 1 | OpenCV 直显 + GT 叠加（红框 vs 蓝框），录对比视频 | 2~3 天 |
| 2 | CARLA 数据采集脚本（自动标注转 YOLO 格式）→ val.py 出 mAP | 2~3 天 |
| 3 | 分层评测（距离/天气/类别）+ AEB 专项（漏检率、检出距离） | 2~3 天 |
| 4 | 网页前端流式展示 | 3~5 天 |

---

## 七、开源参考

| 仓库 | 用途 |
|:---|:---|
| `RyangDiaz/carla-yolo-dataset-generator` | CARLA 0.9.15 + YOLOv8：数据采集 → 训练 → 推理可视化 → 视频，全套（https://github.com/RyangDiaz/carla-yolo-dataset-generator） |
| `AveesLab/CarFree` | 自动 GT 生成（车辆/行人），KITTI/COCO/YOLO 格式（2022 论文）（https://github.com/AveesLab/CarFree） |
| `MukhlasAdib/CARLA-2DBBox` | 2D bbox 生成工具（https://github.com/MukhlasAdib/CARLA-2DBBox） |

---

## 八、结论速记

- **展示**：YOLO 框 + 真值框叠加（OpenCV 起步 → 网页看板收尾）
- **评测**：像素级真值 + 分层指标（距离/天气/类别）+ AEB 专项（漏检率/检出距离）
- **一句话**：CARLA 的评测数据比实车更可靠（无标注误差、自带距离真值），是答辩时的硬数据来源

---

## 相关链接

- CARLA 官方 bbox 教程：[tuto_G_bounding_boxes](https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/)
- CARLA 语义分割相机：[Sensors reference](https://carla.readthedocs.io/en/0.9.16/ref_sensors/)


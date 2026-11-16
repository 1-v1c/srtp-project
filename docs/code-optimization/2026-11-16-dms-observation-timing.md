# DMS 观测有效性与 PERCLOS 计时优化

## 基本信息

- **修改日期** 2026-11-16（北京时间）
- **模块** DMS 疲劳检测、CalibTool 桌面工具包内的运行入口
- **涉及文件** `fatigue_detection/src/` 和工具包 `desktop/src/` 的 `main.py`、`features.py`、`fatigue.py`、`config.py`，以及 `fatigue_detection/tests/test_observation_timing.py`

## 改动与改善

- 统一人脸、头姿、眼部观测有效性。首帧无人脸、关键点不足或头姿解算失败时继续运行。CSV 追加 `face_valid`、`pose_valid`、`eye_valid`、`perclos_valid_sec`。
- 先求头姿，再判断 EAR 能否使用。侧脸、丢脸及超过 0.25 s 的帧间隔会切断闭眼计时，避免把中断前后的闭眼合成一次微睡眠。
- 连续闭眼、PERCLOS 和评分共用单调时钟。PERCLOS 以 30 s 窗内实际有效观测时长为分母，未达到 0.3 s 的短眨眼豁免。丢帧间隔只推动评分衰减。等间隔离线回放可继续用原来的名义帧率接口。

## 验证结果

- 主程序和桌面入口各在本地、团队基线上通过 14 项既有测试和 9 项新增离线用例。本地实验 B 的六场景可重放，基线断言通过。
- 同样 10 s 的闭眼，15 FPS 与 30 FPS 下均得到 33.3% PERCLOS。满窗后的 0.2 s 短眨眼贡献为 0，模拟侧脸低 EAR 和丢脸重获均未触发假微睡眠。有效正脸连续闭眼 2 s 仍可达到 L3。
- **验证范围** 模拟帧经过实际主循环，另有比较不同帧率的时间窗测试。实机摄像头和 CARLA 联调待验证。

## 代码与交付状态

- **代码** 本地工程已修复，团队仓库的同范围实现位于 `codex/dms-p0-observation` 审阅分支，待合入 `main)。[提交 3da14a2](https://github.com/1-v1c/srtp-project/commit/3da14a2fb598f26e18b9b07962e58dfb9870f9b5)

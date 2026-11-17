# AEB 单工况仿真与纵向时序图

## 基本信息

- **修改日期** 2026-11-17（北京时间）
- **模块** AEB 软件在环仿真与输出
- **涉及文件** `aeb/simulate_single_case.py`、`aeb/simulate_staged_brake.py`、`aeb/tests/test_simulate_single_case.py`、`aeb/README.md`

## 改动与改善

- 新增 `python -m aeb.simulate_single_case` 入口。每次命令运行一个纵向工况、一种控制器和一个疲劳等级。车速、净距、前车的制动区间、执行器延迟与能力可从命令行设置。
- 共用既有的 50 Hz 控制、200 Hz 车辆积分和减速度执行器，使用保险杠净距对接触判定。批量场景的前车制动区间沿用 1.2–3.7 s 默认值。
- 输出一行汇总、一份逐帧轨迹和一张四行共用时间轴的 PNG。图内依次绘制指令加速度、实际加速度、自车实际车速与最小净距所属的距离曲线。

## 验证结果

- `python -m unittest discover -s aeb/tests -v` 通过 22 项测试。新增覆盖前车制动区间配置、单工况 CSV 列及接触标记、非有限输入。
- 默认设置为自车 60 km/h、前车 40 km/h、初始净距 22 m，前车在 1.2–3.7 s 减速 4 m/s²。V3 L0 仿真持续 30 s，无接触，最小净距 1.084 m。汇总一行，逐帧 6001 行。实开制动 1.35 s，峰值减速度 8.0 m/s²。
- 原批量仿真重跑后，指标汇总 CSV、逐帧 CSV 和对照 PNG 的 SHA-256 与改动之前一致。
- **验证范围** 纵向软件在环、现有车辆执行器模型与本地输出。

## 代码与交付状态

- **代码** `codex/aeb-single-case-simulation` 提交至 `main` 的合并请求。[单工况入口实现](https://github.com/1-v1c/srtp-project/commit/e827a0a870f389a7697ad9f0ab8bb0090d7bec47)

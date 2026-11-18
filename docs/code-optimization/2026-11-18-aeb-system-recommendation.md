# AEB 单工况图展示系统自身建议加速度

## 基本信息

- **修改日期** 2026-11-18（北京时间）
- **模块** AEB 仿真日志、单工况绘图
- **涉及文件** `aeb/simulate_staged_brake.py`、`aeb/simulate_single_case.py`、`aeb/tests/test_simulate_single_case.py`、`aeb/README.md`

## 改动与改善

- 原单工况图第一行展示人车仲裁后的最终执行指令，无法独立观察系统建议。改为记录 `box.aeb_decel` 并绘制其负值，更新中文和英文标题、图例；驾驶员请求仍用虚线对照。
- V3 系统建议已经过控制器 jerk 整形，尚未与驾驶员请求仲裁，区别于整形前的 `brake_reference`；建议基于本次闭环实际状态计算。
- 单工况轨迹新增 `aeb_accel_mps2`，保留最终执行指令 `command_accel_mps2` 和所有原有列。共享批量轨迹在末尾新增正值减速度形式的 `aeb_decel_mps2`。正常、接触和终止帧均携带新字段。
- 保留原有仲裁及车辆执行路径，README 同步更新。

## 验证结果

- `python -m unittest discover -s aeb/tests`：31 项测试通过。新增覆盖无目标时人工加速/减速与零系统建议的分离，以及 V3 系统建议的 jerk 整形和与最终制动指令的区别。
- 默认 V3 L0 示例：人工加速 1 m/s²（0–1 s）、制动 2 m/s²（2–6 s），输出 6001 条轨迹，最小净距 2.203 m，30 s 内无接触。逐帧所有原有列和汇总 CSV 与修改前完全一致；初始系统建议为 0，驾驶员和最终执行请求为 +1 m/s²。已检查更新后的四行图。
- 批量仿真完整运行：75283 条轨迹，新增列与表头对齐；所有原有轨迹列、汇总 CSV 和批量 PNG 与修改前一致。
- **验证范围** 简化纵向车辆模型，包括 V2、V3、人工加速/减速、接触及正常结束记录；未做 CARLA 或实车验证。

## 代码与交付状态

- **代码** 交付分支 `codex/aeb-system-recommendation`；[功能代码提交](https://github.com/1-v1c/srtp-project/commit/d2a05f05b9f0e4f5833939e8acd44a209870830e)。本记录随分支申请合入 `main`，最终合入状态以仓库 PR 为准。

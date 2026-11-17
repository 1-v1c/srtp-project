# AEB 入口收敛到 aeb 包

## 基本信息

- **修改日期** 2026-11-17（北京时间）
- **模块** AEB 联调、仿真和测试入口
- **涉及文件** `aeb/README.md` 及四个仓库外层转发文件

## 改动与改善

- 检查了审阅分支中 42 个源码和自动化文件，AEB 代码与测试都从 `aeb/` 导入，仓库中没有依赖顶层旧路径的调用方。
- 删除旧的 `brake_algorithm.py`、`data_box.py`、`tests/test_brake_algorithm.py` 和 `scripts/simulate_staged_brake.py` 转发文件。联调导入和执行命令统一写在 [aeb/README.md](../../aeb/README.md)。

## 验证结果

- `python -m unittest discover -s aeb/tests` 和 `python -m unittest discover` 分别通过 19 项。包内模块自测正常运行。
- 使用包内入口重跑仿真，指标 CSV、逐帧 CSV 和曲线 PNG 的 SHA-256 与整理前一致。
- **验证范围** 共享分支中的调用、Python 回归与纵向软件在环。

## 代码与交付状态

- **代码** `codex/aeb-staged-braking` 审阅分支。[入口清理提交](https://github.com/1-v1c/srtp-project/commit/8aa56b76085a8c735227e1d59fd9bf184ac626dd)

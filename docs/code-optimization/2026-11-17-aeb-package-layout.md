# AEB 代码目录整理

## 基本信息

- **修改日期** 2026-11-17（北京时间）
- **模块** AEB 制动决策、车辆数据接口与回归验证
- **涉及文件** `aeb/`、仓库根目录兼容入口、原脚本和测试入口

## 改动与改善

- 将 V3 制动入口、纵向状态机、数据盒、对比仿真及回归测试归入 [aeb](../../aeb/)。在目录中增设 [README](../../aeb/README.md)，说明代码职责、制动接口及运行方式。
- 根目录的 `brake_algorithm.py`、`data_box.py` 和原仿真、测试路径保留转发入口，供已有的联调导入和复跑命令调用。

## 验证结果

- `python -m unittest discover -s aeb/tests -v`、`python -m unittest discover -s tests` 和 `python -m unittest discover` 各通过 19 项测试。
- 旧根模块与包内模块导入到同一个入口函数和数据盒类。旧命令与新模块自测的输出一致，原仿真脚本支持相同的命令行参数。
- 整理后重跑 V2/V3 软件在环对比。汇总 CSV、逐帧 CSV 和对比 PNG 的 SHA-256 均与整理前一致。
- **验证范围** 目录与入口迁移、Python 回归及现有纵向软件在环。

## 代码与交付状态

- **代码** `codex/aeb-staged-braking` 审阅分支，待合入 `main)。[目录整理提交](https://github.com/1-v1c/srtp-project/commit/3fe02249031ea349c589c41cd6f5bf690b1ecad6)

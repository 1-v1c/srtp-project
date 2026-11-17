# AEB 制动控制

仿真或联调每帧更新 `DataBox`，调用 `update_brake_result(box, dt_sec=...)`，然后读取 `box.a_final` 交给执行器。疲劳等级与制动等级各有独立字段。减速度指令为正值，车辆纵向加速度在制动时为负值。

| 文件 | 功能 |
|---|---|
| `data_box.py` | 定义输入和输出字段。输入包括疲劳等级与有效性、目标中心位移和保险杠净距、车速、驾驶员操作、执行器参数。输出包括预警、制动级别、制动阶段和最终减速度。 |
| `brake_algorithm.py` | AEB 逐帧入口。校验输入和目标，计算 TTC，选择纵向分阶段控制或兼容策略，保留驾驶员更强的制动请求，并把结果写回数据盒。 |
| `longitudinal_brake.py` | V3 纵向控制器。预测不同制动强度下的最小保险杠净距，根据风险选择轻制动、中度制动或紧急制动，执行 jerk 限制及释放滞回。 |
| `simulate_staged_brake.py` | 用同一简化车辆和执行器模型对比 V2 与 V3，输出逐帧时间序列、指标汇总，选择 `--plot` 时还生成加速度与净距对照图。 |
| `tests/test_brake_algorithm.py` | 回归测试。覆盖接口、低速补制动、无效或切换目标、净距预测、紧急升级、释放斜坡及纵向闭环场景。 |
| `__init__.py`、`tests/__init__.py` | 声明 AEB 包及其测试子包，供模块导入和标准测试发现。 |

## 从仓库根目录运行

```powershell
python -m unittest discover -s aeb/tests -v
python -m aeb.brake_algorithm
python -X utf8 -m aeb.simulate_staged_brake --output-dir artifacts --plot
```

绘图需要 Matplotlib。只需要 CSV 时可去掉 `--plot`。

## 联调导入

在仓库根目录执行或把仓库安装路径加入 Python 模块搜索路径。联调模块统一使用包路径导入。

```python
from aeb.brake_algorithm import update_brake_result
from aeb.data_box import DataBox
```

代码优化记录集中在 [docs/code-optimization](../docs/code-optimization/)。

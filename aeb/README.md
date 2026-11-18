# AEB 制动控制

仿真或联调每帧更新 `DataBox`，调用 `update_brake_result(box, dt_sec=...)`，然后读取 `box.a_final` 交给执行器。疲劳等级与制动等级各有独立字段。减速度指令为正值，车辆纵向加速度在制动时为负值。

| 文件 | 功能 |
|---|---|
| `data_box.py` | 定义输入和输出字段。输入包括疲劳等级与有效性、目标中心位移和保险杠净距、车速、驾驶员操作、执行器参数。输出包括预警、制动级别、制动阶段和最终减速度。 |
| `brake_algorithm.py` | AEB 逐帧入口。校验输入和目标，计算 TTC，选择纵向分阶段控制或兼容策略，保留驾驶员更强的制动请求，并把结果写回数据盒。 |
| `longitudinal_brake.py` | V3 纵向控制器。预测不同制动强度下的最小保险杠净距，根据风险选择轻制动、中度制动或紧急制动，执行 jerk 限制及释放滞回。 |
| `simulate_staged_brake.py` | 用同一简化车辆和执行器模型对比 V2 与 V3，输出逐帧时间序列、指标汇总，选择 `--plot` 时还生成加速度与净距对照图。 |
| `simulate_single_case.py` | 每次指定一个纵向工况、一种制动策略和一个疲劳等级，输出两个 CSV 与一张四行共用时间轴的曲线图。 |
| `tests/test_brake_algorithm.py` | 回归测试。覆盖接口、低速补制动、无效或切换目标、净距预测、紧急升级、释放斜坡及纵向闭环场景。 |
| `__init__.py`、`tests/__init__.py` | 声明 AEB 包及其测试子包，供模块导入和标准测试发现。 |

## 从仓库根目录运行

```powershell
python -m unittest discover -s aeb/tests -v
python -m aeb.brake_algorithm
python -X utf8 -m aeb.simulate_staged_brake --output-dir artifacts --plot
```

绘图需要 Matplotlib。只需要 CSV 时可去掉 `--plot`。

## 单工况仿真

先安装 `matplotlib`。单工况命令默认运行 V3 L0，自车 60 km/h、前车 40 km/h、初始净距 22 m，前车在 1.2–3.7 s 内以 4 m/s² 减速。每次仿真持续至几何接触或最长 30 s。

```powershell
python -m pip install matplotlib
python -X utf8 -m aeb.simulate_single_case --output-dir artifacts_single
```

下面的完整命令覆盖当前单工况入口的全部可调参数。在仓库根目录整行复制运行，再按需要修改参数后的值。

```powershell
python -X utf8 -m aeb.simulate_single_case --controller v3 --fatigue 0 --ego-kmh 60 --lead-kmh 40 --gap-m 22 --lead-brake-mps2 4 --lead-brake-start-s 1.2 --lead-brake-end-s 3.7 --driver-brake-mps2 2 --driver-brake-start-s 2 --driver-brake-end-s 6 --driver-accel-mps2 1 --driver-accel-start-s 0 --driver-accel-end-s 1 --delay-s 0.15 --capacity-mps2 8 --actuator-jerk-mps3 30 --output-dir artifacts_custom
```

| 参数 | 示例值 | 含义 |
|---|---|---|
| `--controller` | `v3` | 算法版本，可选 `v2`、`v3` |
| `--fatigue` | `0` | 疲劳等级，可选 `0`～`3` |
| `--ego-kmh` | `60` | 自车初速，km/h |
| `--lead-kmh` | `40` | 前车初速，km/h |
| `--gap-m` | `22` | 初始保险杠净距，m |
| `--lead-brake-mps2` | `4` | 前车减速强度，m/s²，填正数，`0` 表示不刹车 |
| `--lead-brake-start-s` | `1.2` | 前车开始刹车的时间，s |
| `--lead-brake-end-s` | `3.7` | 前车结束刹车的时间，s |
| `--driver-brake-mps2` | `2` | 自车驾驶员制动请求，m/s²，填正数，`0` 表示不踩刹车 |
| `--driver-brake-start-s` | `2` | 自车驾驶员开始踩刹车的时间，s |
| `--driver-brake-end-s` | `6` | 自车驾驶员松开刹车的时间，s |
| `--driver-accel-mps2` | `1` | 自车驾驶员加速强度，m/s²，填正数，默认 0 |
| `--driver-accel-start-s` | `0` | 自车驾驶员开始加速的时间，s |
| `--driver-accel-end-s` | `1` | 自车驾驶员结束加速的时间，s |
| `--delay-s` | `0.15` | 车辆制动执行延迟，s |
| `--capacity-mps2` | `8` | 车辆模型的最大减速度，m/s² |
| `--actuator-jerk-mps3` | `30` | 执行器减速度变化率上限，m/s³ |
| `--output-dir` | `artifacts_custom` | 保存两份 CSV 和曲线图的目录 |

前车和驾驶员制动区间均为开始时刻包含、结束时刻不包含。驾驶员时间满足 `0 ≤ 开始 < 结束 ≤ 30` 秒。

当前仿真总时长为 30 秒，控制周期 20 ms，车辆积分周期 5 ms，这三个值在代码中设置。`--actuator-jerk-mps3` 调节执行器能力，V3 内部各阶段的 jerk 参数由 `BrakeTuning` 设置。每次换工况可以更换输出目录，以保留各次结果。

- `aeb-single-summary.csv` 只有一条工况结果，含输入参数、制动时刻、峰值、最小净距、接触与否和结束时是否仍在接近。
- `aeb-single-traces.csv` 同步记录加速度指令、模型实际加速度、自车与前车车速、保险杠净距和制动状态。
- `aeb-single-curves.png` 自上而下给出指令加速度、实际加速度、自车实际车速、保险杠净距。输出目录相同时，同名文件会被新一轮结果覆盖。

测试对照可分别指定 `--controller v2` 和 `--controller v3`，并为各次运行指定不同的 `--output-dir`。

## 联调导入

在仓库根目录执行或把仓库安装路径加入 Python 模块搜索路径。联调模块统一使用包路径导入。

```python
from aeb.brake_algorithm import update_brake_result
from aeb.data_box import DataBox
```

代码优化记录集中在 [docs/code-optimization](../docs/code-optimization/)。

## 驾驶员与 AEB 同时制动

单工况仿真可设置驾驶员的正值减速度请求、开始和结束时间。例如从第 2 秒到第 6 秒请求 2 m/s² 制动。

```powershell
python -X utf8 -m aeb.simulate_single_case --driver-brake-mps2 2 --driver-brake-start-s 2 --driver-brake-end-s 6 --output-dir artifacts_driver_brake
```

默认驾驶员请求为 0。时间区间左闭右开，按 20 ms 控制周期采样；结束时间默认 30 s。驾驶员请求和 AEB 请求由选定的控制器仲裁，V3 纵向取两者较大的减速度并受车辆能力限制，V2 使用原有权重逻辑。最终请求经过车辆模型的制动延迟和速率限制执行。驾驶员请求强度为踏板效果的简化量。

汇总 CSV 保存这三个参数，逐帧 CSV 的 `driver_request_mps2` 保存该帧已采样的驾驶员正值请求。四行图第一行的实线表示仲裁后指令，虚线为驾驶员纵向请求（加速正、制动负）。前车制动仍由 `--lead-brake-mps2` 等参数配置。

## 驾驶员加速与减速

加速和制动各自指定正值强度及时间区间。下面的例子中，驾驶员先在 0–1 s 请求加速 1 m/s²，再在 2–6 s 请求制动 2 m/s²。两种操作可以在同一轮仿真中发生。

```powershell
python -X utf8 -m aeb.simulate_single_case --driver-accel-mps2 1 --driver-accel-start-s 0 --driver-accel-end-s 1 --driver-brake-mps2 2 --driver-brake-start-s 2 --driver-brake-end-s 6 --output-dir artifacts_driver_operation
```

- 驾驶员加速请求传给算法时使用负的 `a_driver`，制动请求使用正的 `a_driver`。若两个人工输入时段重叠，先取驾驶员制动。
- AEB 或驾驶员产生正制动指令时优先执行制动；制动指令为零时执行驾驶员加速。最终纵向请求统一经过仿真执行延迟和变化率限制，加速请求为车辆纵向加速度的简化输入。
- CSV 保留 `driver_request_mps2` 表示驾驶员原始制动强度，新增 `driver_accel_request_mps2` 表示原始加速强度，`driver_longitudinal_request_mps2` 表示处理人工重叠后的纵向请求（加速为正，制动为负）。
- `command_accel_mps2` 是仲裁后的纵向加速度指令。图中第一行虚线为驾驶员纵向请求，实线为仲裁后的指令；第二行实际加速度还受执行延迟和建立速率影响。

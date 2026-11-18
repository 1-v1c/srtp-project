"""Reproduce longitudinal A/B simulations for the staged braking controller."""
import argparse
from collections import deque
import csv
from dataclasses import dataclass
from pathlib import Path

from .brake_algorithm import update_brake_result
from .data_box import DataBox


CONTROL_DT = 0.02
PHYSICS_DT = 0.005
DURATION = 30.0
STEP_RATIO = round(CONTROL_DT / PHYSICS_DT)
BRAKE_EPS = 0.1


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    gap: float
    ego_speed: float
    lead_speed: float
    lead_brake: float = 0.0
    target_valid: bool = True
    delay: float = 0.15
    capacity: float = 8.0
    actuator_jerk: float = 30.0
    lead_brake_start_s: float = 1.2
    lead_brake_end_s: float = 3.7
    driver_brake_mps2: float = 0.0
    driver_brake_start_s: float = 0.0
    driver_brake_end_s: float = DURATION

    driver_accel_mps2: float = 0.0
    driver_accel_start_s: float = 0.0
    driver_accel_end_s: float = DURATION

    def driver_accel(self, t):
        return self.driver_accel_mps2 if self.driver_accel_start_s <= t < self.driver_accel_end_s else 0.0

    def driver_operation(self, t):
        # DataBox uses positive braking and negative acceleration. Brake takes priority.
        brake = self.driver_brake(t)
        return brake if brake > 0 else -self.driver_accel(t)

    def driver_brake(self, t):
        return self.driver_brake_mps2 if self.driver_brake_start_s <= t < self.driver_brake_end_s else 0.0

    def lead_accel(self, t):
        return -self.lead_brake if self.lead_brake_start_s <= t < self.lead_brake_end_s else 0.0


SCENARIOS = (
    Scenario("CCRs", "静止前车", 45.0, 50 / 3.6, 0.0),
    Scenario("CCRb", "前车急刹", 36.0, 22.0, 16.0, lead_brake=5.0),
    Scenario("creep", "低速接近", 1.0, 0.4, 0.0, delay=0.30, actuator_jerk=15),
    Scenario("adjacent", "相邻车道慢车", 15.0, 50 / 3.6, 20 / 3.6, target_valid=False),
    Scenario("CCRb-pressure", "急刹执行器压力", 36.0, 22.0, 16.0,
             lead_brake=5.0, delay=0.30, capacity=5.0, actuator_jerk=15.0),
)


def simulate(case, staged, fatigue):
    box = DataBox()
    box.fatigue_level = fatigue
    box.brake_response_delay_sec = case.delay
    box.available_decel_mps2 = case.capacity
    box.actuator_jerk_mps3 = case.actuator_jerk
    box.target_valid = case.target_valid

    gap, ego, lead = case.gap, case.ego_speed, case.lead_speed
    pending = deque([0.0] * round(case.delay / PHYSICS_DT))
    applied_decel = 0.0
    actual_accel = 0.0
    longitudinal_request = 0.0
    records = []
    for i in range(round(DURATION / PHYSICS_DT)):
        t = i * PHYSICS_DT
        lead_accel = case.lead_accel(t) if lead > 0 else 0.0
        if i % STEP_RATIO == 0:
            # Both policies receive the same center distance and measured net gap.
            box.v_ego, box.v_target = ego, lead
            box.ego_accel_mps2 = actual_accel
            box.a_target = lead_accel
            box.dx = gap + 4.5
            box.dy = 0.0 if case.target_valid else 3.5
            box.d = (box.dx ** 2 + box.dy ** 2) ** 0.5
            box.bumper_gap_m = gap if case.target_valid else None
            box.a_driver = case.driver_operation(t)
            update_brake_result(box, dt_sec=CONTROL_DT, staged=staged)
            longitudinal_request = box.a_final if box.a_final > 0 or box.a_driver >= 0 else box.a_driver

        # Use one signed longitudinal actuator. Positive means braking, negative means acceleration.
        pending.append(min(case.capacity, longitudinal_request))
        target_decel = pending.popleft()
        increment = case.actuator_jerk * PHYSICS_DT
        applied_decel += min(increment, max(-increment, target_decel - applied_decel))
        next_ego = max(0.0, ego - applied_decel * PHYSICS_DT)
        next_lead = max(0.0, lead + lead_accel * PHYSICS_DT)
        actual_accel = (next_ego - ego) / PHYSICS_DT
        next_gap = gap + ((lead + next_lead) - (ego + next_ego)) * PHYSICS_DT / 2
        records.append((t, gap, ego, lead, longitudinal_request, actual_accel, box.brake_reference,
                        box.brake_phase, box.warning, False, box.aeb_decel))
        if case.target_valid and next_gap <= 1e-9:
            fraction = min(1.0, gap / (gap - next_gap))
            records.append((t + fraction * PHYSICS_DT, 0.0,
                            ego + (next_ego - ego) * fraction,
                            lead + (next_lead - lead) * fraction,
                            longitudinal_request, actual_accel, box.brake_reference,
                            box.brake_phase, box.warning, True, box.aeb_decel))
            break
        gap, ego, lead = next_gap, next_ego, next_lead
    else:
        records.append((DURATION, gap, ego, lead, longitudinal_request, actual_accel,
                        box.brake_reference, box.brake_phase, box.warning, False, box.aeb_decel))
    return records


def metrics(case, records):
    last = records[-1]
    command_active = [row[4] >= BRAKE_EPS for row in records]
    first_cmd = next((row[0] for row, active in zip(records, command_active) if active), None)
    first_actual = next((row[0] for row in records if row[5] <= -BRAKE_EPS), None)
    jerk = (abs((right[5] - left[5]) / PHYSICS_DT)
            for left, right in zip(records, records[1:])
            if left[2] > 0.05 and right[2] > 0.05 and not right[9])
    return {
        "switches": int(command_active[0]) + sum(a != b for a, b in zip(command_active, command_active[1:])),
        "first_cmd_s": first_cmd, "first_actual_s": first_actual,
        "peak_decel": -min(row[5] for row in records),
        "peak_jerk": max(jerk, default=0.0),
        "gap_min": min(row[1] for row in records) if case.target_valid else None,
        "contact_s": last[0] if last[9] else None,
        "impact_relative_kmh": (last[2] - last[3]) * 3.6 if last[9] else None,
        "final_closing_mps": last[2] - last[3] if case.target_valid else None,
    }


def plot_comparison(runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, fontManager
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    for font in ("C:/Windows/Fonts/msyh.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
        if Path(font).is_file():
            fontManager.addfont(font)
            plt.rcParams["font.family"] = FontProperties(fname=font).get_name()
            break
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#bcc9d0",
        "axes.labelcolor": "#243f51",
        "xtick.color": "#3e5260",
        "ytick.color": "#3e5260",
        "text.color": "#172f41",
    })
    baseline_color = "#6a7079"
    staged_color = "#00677E"
    contact_color = "#A3403D"
    fig, axes = plt.subplots(3, 3, figsize=(14.15, 9.0), sharex="col")
    layouts = (
        ("CCRs", "静止前车", "50 km/h · 净距 45 m", 7.5, -6.5),
        ("CCRb", "前车急刹", "79.2 / 57.6 km/h · 净距 36 m", 9.3, -8.5),
        ("creep", "低速接近", "1.44 km/h · 净距 1.0 m", 1.7, -1.8),
    )
    for col, (key, title, setup, end, ymin) in enumerate(layouts):
        column = axes[:, col]
        old = runs[key, "baseline", 0]
        staged = runs[key, "staged", 0]
        column[0].set_title(f"{title}\n{setup}", loc="left", pad=9, fontsize=12, linespacing=1.6)
        for ax in column:
            ax.set_xlim(0, end)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.grid(which="major", color="#dfe7eb", lw=0.75)
            ax.set_axisbelow(True)
            if key == "CCRb":
                ax.axvspan(1.2, 3.7, color="#c1cbd2", alpha=0.22, linewidth=0, zorder=0)
        for row, (index, sign) in enumerate(((5, 1), (4, -1), (1, 1))):
            ax = column[row]
            for trace, color, style, width, z in (
                (old, baseline_color, (0, (4, 2)), 1.65, 2),
                (staged, staged_color, "-", 2.15, 3),
            ):
                xs = [frame[0] for frame in trace]
                ys = [sign * frame[index] for frame in trace]
                if row == 1:
                    ax.step(xs, ys, where="post", color=color, ls=style, lw=width, zorder=z)
                else:
                    ax.plot(xs, ys, color=color, ls=style, lw=width, zorder=z)
            if row < 2:
                ax.set_ylim(ymin, 0.36 if key != "creep" else 0.18)
                if row == 0 and key in ("CCRs", "creep"):
                    stopped = next(frame for frame in staged if frame[2] == 0.0)
                    ax.plot(stopped[0], 0, marker="o", color=staged_color, ms=3.3)
                    ax.annotate("停稳", (stopped[0], 0), xytext=(6, -14),
                                textcoords="offset points", color=staged_color, fontsize=9)
            else:
                ax.set_ylim(bottom=-0.04 if key == "creep" else -2.0)
                if old[-1][9]:
                    ax.plot(old[-1][0], 0.0, "x", color=contact_color, zorder=4, ms=7, mew=1.7)
                    ax.annotate("基线接触", xy=(old[-1][0], 0), xytext=(-10, 12),
                                textcoords="offset points", ha="right", color=contact_color, fontsize=9)
        column[-1].set_xlabel("仿真时间 (s)")

    axes[0, 0].set_ylabel("实际加速度 (m/s²)")
    axes[1, 0].set_ylabel("指令加速度 (m/s²)")
    axes[2, 0].set_ylabel("保险杠净距 (m)")
    fig.text(0.085, 0.950, "AEB 分阶段制动的加速度对照", fontsize=24, weight="bold")
    fig.text(0.085, 0.914, "同一车辆模型与输入 · 疲劳 L0 · 50 Hz 控制 / 200 Hz 车辆积分",
             fontsize=11.5, color="#425b69")
    fig.legend(handles=(
        Line2D([], [], color=baseline_color, ls=(0, (4, 2)), lw=1.65, label="原净距版"),
        Line2D([], [], color=staged_color, lw=2.15, label="新版分阶段制动"),
        Line2D([], [], color=contact_color, ls="none", marker="x", label="发生接触"),
    ), loc="upper left", bbox_to_anchor=(0.077, 0.895), ncol=3, frameon=False)
    fig.subplots_adjust(left=0.085, right=0.975, top=0.803, bottom=0.112, wspace=0.22, hspace=0.16)
    fig.text(0.085, 0.031, "中列阴影表示前车减速。停稳后速度钳为零，实际加速度随之归零。最小净距来自完整 30 s 日志。",
             fontsize=10.5, color="#425b69")
    fig.savefig(path, dpi=185, facecolor="white")
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = {}
    summary = []
    for case in SCENARIOS:
        for staged, level in ((False, 0), (True, 0), (True, 3)):
            policy = "staged" if staged else "baseline"
            trace = simulate(case, staged, level)
            runs[case.key, policy, level] = trace
            summary.append({"scenario": case.key, "policy": policy, "fatigue": level,
                            **metrics(case, trace)})

    with (args.output_dir / "aeb-staged-summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (args.output_dir / "aeb-staged-traces.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(("scenario", "policy", "fatigue", "time_s", "bumper_gap_m",
                         "ego_mps", "lead_mps", "cmd_decel_mps2", "actual_accel_mps2",
                         "ref_decel_mps2", "phase", "warning", "contact", "aeb_decel_mps2"))
        for (case, policy, level), trace in runs.items():
            writer.writerows((case, policy, level, *(
                round(x, 6) if isinstance(x, float) else x for x in row)) for row in trace)
    if args.plot:
        plot_comparison(runs, args.output_dir / "aeb-staged-comparison.png")
    for row in summary:
        print(" ".join(f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}"
                       for key, value in row.items()))


if __name__ == "__main__":
    main()

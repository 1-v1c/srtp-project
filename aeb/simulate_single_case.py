"""Run one longitudinal AEB case with one controller and one fatigue level."""
import argparse
import csv
import math
from pathlib import Path

from .simulate_staged_brake import CONTROL_DT, DURATION, PHYSICS_DT, Scenario, metrics, simulate


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--controller", choices=("v2", "v3"), default="v3")
    parser.add_argument("--fatigue", type=int, choices=range(4), default=0)
    parser.add_argument("--ego-kmh", type=float, default=60.0)
    parser.add_argument("--lead-kmh", type=float, default=40.0)
    parser.add_argument("--gap-m", type=float, default=22.0)
    parser.add_argument("--lead-brake-mps2", type=float, default=4.0)
    parser.add_argument("--lead-brake-start-s", type=float, default=1.2)
    parser.add_argument("--lead-brake-end-s", type=float, default=3.7)
    parser.add_argument("--delay-s", type=float, default=0.15)
    parser.add_argument("--capacity-mps2", type=float, default=8.0)
    parser.add_argument("--actuator-jerk-mps3", type=float, default=30.0)
    parser.add_argument("--driver-brake-mps2", type=float, default=0.0,
                        help="Driver brake request in m/s², positive for braking")
    parser.add_argument("--driver-brake-start-s", type=float, default=0.0)
    parser.add_argument("--driver-brake-end-s", type=float, default=DURATION)
    args = parser.parse_args(argv)

    for name, value in vars(args).items():
        if isinstance(value, float) and not math.isfinite(value):
            parser.error(f"--{name.replace('_', '-')} must be finite")
    if min(args.ego_kmh, args.lead_kmh, args.lead_brake_mps2) < 0 or args.gap_m <= 0:
        parser.error("speeds and lead braking must be non-negative, and gap must be positive")
    if not 0 <= args.lead_brake_start_s < args.lead_brake_end_s:
        parser.error("lead braking must start at or after zero and end after the start")
    if not 0 <= args.delay_s <= DURATION or min(args.capacity_mps2, args.actuator_jerk_mps3) <= 0:
        parser.error(f"delay must be between 0 and {DURATION:g} s; capacity and jerk must be positive")
    if args.driver_brake_mps2 < 0:
        parser.error("driver braking must be non-negative")
    if not 0 <= args.driver_brake_start_s < args.driver_brake_end_s <= DURATION:
        parser.error(f"driver braking must satisfy 0 <= start < end <= {DURATION:g} s")
    return args


def trace_rows(records, case=None):
    for t, gap, ego, lead, command, actual, reference, phase, warning, contact in records:
        # Terminal/contact snapshots retain the request latched at the last control tick.
        control_step = min(math.floor(t / CONTROL_DT + 1e-9), round(DURATION / CONTROL_DT) - 1)
        driver_request = case.driver_brake(control_step * CONTROL_DT) if case else 0.0
        yield {
            "time_s": t,
            "driver_request_mps2": driver_request,
            "command_accel_mps2": -command,
            "actual_accel_mps2": actual,
            "ego_speed_kmh": ego * 3.6,
            "bumper_gap_m": gap,
            "lead_speed_kmh": lead * 3.6,
            "reference_accel_mps2": -reference,
            "phase": phase,
            "warning": warning,
            "contact": contact,
        }


def render_plot(path, rows, args, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, fontManager

    font_options = (
        "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    )
    font = next((f for f in font_options if Path(f).is_file()), None)
    if font:
        fontManager.addfont(font)
        plt.rcParams["font.family"] = FontProperties(fname=font).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 12})
    titles = ("指令加速度", "实际加速度", "实际车速", "保险杠净距") if font else (
        "Commanded acceleration", "Actual acceleration", "Ego speed", "Bumper-to-bumper gap")
    series = ("command_accel_mps2", "actual_accel_mps2", "ego_speed_kmh", "bumper_gap_m")
    units = ("m/s²", "m/s²", "km/h", "m")
    colors = ("#356482", "#006d68", "#32607c", "#9b562c")
    times = [r["time_s"] for r in rows]

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.subplots_adjust(left=0.11, right=0.965, top=0.855, bottom=0.098, hspace=0.45)
    heading = "AEB 单工况闭环仿真" if font else "Single-case closed-loop AEB simulation"
    fig.text(0.11, 0.959, heading, fontsize=25, weight="bold", color="#133343")
    labels = ("自车", "前车", "初始净距", "制动延迟") if font else (
        "Ego", "Lead", "Initial gap", "Brake delay")
    fig.text(0.11, 0.924,
             f"{args.controller.upper()} / L{args.fatigue}   ·   "
             f"{labels[0]} {args.ego_kmh:g} km/h   ·   {labels[1]} {args.lead_kmh:g} km/h   ·   "
             f"{labels[2]} {args.gap_m:g} m", color="#385968")
    fig.text(0.11, 0.894,
             f"{labels[3]} {args.delay_s:g} s   ·   "
             f"50 Hz {'控制' if font else 'controller'} / 200 Hz {'车辆积分' if font else 'vehicle'}",
             fontsize=11, color="#385968")

    for index, (ax, title, key, unit, color) in enumerate(zip(axes, titles, series, units, colors)):
        values = [r[key] for r in rows]
        if index == 0:
            ax.step(times, values, where="post", color=color, lw=2)
        else:
            ax.plot(times, values, color=color, lw=2)
        if index == 0 and args.driver_brake_mps2 > 0:
            ax.step(times, [-r["driver_request_mps2"] for r in rows], where="post",
                    color="#a26a20", ls="--", lw=1.6,
                    label="驾驶员制动请求" if font else "Driver brake request")
            ax.lines[0].set_label("仲裁后指令" if font else "Arbitrated command")
            ax.legend(loc="lower right", frameon=False, fontsize=10)
        if args.lead_brake_mps2 > 0 and args.lead_kmh > 0:
            stop = args.lead_brake_start_s + args.lead_kmh / 3.6 / args.lead_brake_mps2
            end = min(args.lead_brake_end_s, stop, times[-1])
            if end > args.lead_brake_start_s:
                ax.axvspan(args.lead_brake_start_s, end, color="#dcebed", alpha=0.65, zorder=0)
        ax.set_title(title, loc="left", color="#133343", fontsize=14, weight="bold", pad=7)
        ax.set_ylabel(unit)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="both", color="#e1eaee", lw=0.75)
        ax.set_axisbelow(True)

    axes[-1].set_xlim(0, times[-1])
    axes[-1].set_xlabel("仿真时间 (s)" if font else "Simulation time (s)")
    closest = min(rows, key=lambda r: r["bumper_gap_m"])
    gap = closest["bumper_gap_m"]
    gap_label = "接触" if summary["contact_s"] is not None else ("最小净距" if font else "Minimum gap")
    if not font and summary["contact_s"] is not None:
        gap_label = "Contact"
    near_right = closest["time_s"] > 0.7 * times[-1]
    axes[-1].plot(closest["time_s"], gap, "o", ms=6, color=colors[-1], zorder=4)
    axes[-1].annotate(f"{gap_label} {gap:.2f} m", (closest["time_s"], gap),
                      textcoords="offset points", xytext=(-12 if near_right else 14, 20),
                      ha="right" if near_right else "left", color=colors[-1], fontsize=11,
                      arrowprops={"arrowstyle": "-", "color": colors[-1], "lw": 0.8},
                      bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
    fig.text(0.11, 0.031,
             "浅色背景标示前车制动区间。加速度为纵向量，制动时为负。"
             if font else "Shading shows lead braking. Longitudinal acceleration is negative under braking.",
             fontsize=10, color="#385968")
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    case = Scenario(
        "single", "Single case", args.gap_m, args.ego_kmh / 3.6, args.lead_kmh / 3.6,
        lead_brake=args.lead_brake_mps2, delay=args.delay_s, capacity=args.capacity_mps2,
        actuator_jerk=args.actuator_jerk_mps3, lead_brake_start_s=args.lead_brake_start_s,
        lead_brake_end_s=args.lead_brake_end_s,
        driver_brake_mps2=args.driver_brake_mps2,
        driver_brake_start_s=args.driver_brake_start_s,
        driver_brake_end_s=args.driver_brake_end_s,
    )
    records = simulate(case, staged=args.controller == "v3", fatigue=args.fatigue)
    result = metrics(case, records)
    outcome = "contact" if result["contact_s"] is not None else (
        "still_closing" if result["final_closing_mps"] > 1e-6 else "no_contact")
    summary = {
        **{k: v for k, v in vars(args).items() if k != "output_dir"},
        "control_dt_s": CONTROL_DT,
        "physics_dt_s": PHYSICS_DT,
        "end_time_s": records[-1][0],
        "outcome": outcome,
        **result,
    }
    rows = list(trace_rows(records, case))

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    render_plot(output / "aeb-single-curves.png", rows, args, summary)
    for name, data in (("aeb-single-summary.csv", [summary]), ("aeb-single-traces.csv", rows)):
        with (output / name).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(f"{args.controller.upper()} L{args.fatigue}: {outcome}; "
          f"min gap {result['gap_min']:.3f} m; {len(rows)} trace rows")
    print(f"Outputs: {output.resolve()}")


if __name__ == "__main__":
    main()

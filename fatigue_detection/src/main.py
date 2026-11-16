"""
疲劳检测系统 — 主程序入口
Phase 1: 摄像头 + Face Mesh 人脸关键点
Phase 2: + EAR 眼部长宽比实时显示
Phase 3: + MAR 嘴部长宽比 + 哈欠计时
Phase 4: + HeadPose 头部姿态
Phase 5: + PERCLOS 时间窗口统计（眨眼豁免）
Phase 6: + 疲劳评分融合（L0~L3 状态机）+ 报警 + 运行日志
Phase 7: + 眨眼动力学特征（疲劳早期信号）

按 'q' 退出 | 按 's' 截图 | 按 'r' 重置疲劳评分
"""

import argparse
from contextlib import ExitStack
import os
import csv
import time
import cv2

from camera import Camera
from detector import FaceMeshDetector
from features import (
    calculate_ear, calculate_mar,
    calculate_head_pose, draw_head_axes,
    LEFT_EYE_INDICES, RIGHT_EYE_INDICES,
    EARSmoother, MARSmoother, HeadPoseSmoother, PitchBaseline, PERCLOSCounter, BlinkAnalyzer,
)
from fatigue import FatigueScorer
from config import (
    EAR_THRESHOLD, EAR_CONSECUTIVE_FRAMES, FPS_TARGET, MAX_OBSERVATION_GAP_SEC,
    MAR_THRESHOLD, MAR_YAWN_DURATION_SEC, MAR_YAWN_GRACE_SEC,
    PITCH_DOWN_THRESHOLD, HEAD_DOWN_DURATION_SEC, YAW_DEVIATION_THRESHOLD,
    YAW_DEVIATION_DURATION_SEC, YAW_EAR_GATE,
    PITCH_BASELINE_DURATION_SEC, PITCH_BASELINE_MAX_SPREAD_DEG,
    PITCH_BASELINE_MAX_YAW_DEG, PITCH_MAX_ROLL_DEG,
    SHOW_FACE_MESH, SHOW_EYE_POINTS, SHOW_MOUTH_POINTS, SHOW_HEAD_POSE_AXIS,
    PERCLOS_WINDOW_SEC, PERCLOS_THRESHOLD, BLINK_EXEMPT_SEC,
    SLOW_BLINK_SEC, BLINK_WINDOW_SEC,
    BLINK_RATE_THRESHOLD, AVG_BLINK_DUR_THRESHOLD, SLOW_BLINK_RATE_THRESHOLD,
    MICRO_SLEEP_SEC, ALERT_COOLDOWN_SEC, LOG_DIR, SCREENSHOT_DIR,
)

# 报警蜂鸣（仅 Windows 可用，非 Windows 自动跳过）
try:
    import winsound
    _HAS_SOUND = True
except ImportError:
    _HAS_SOUND = False

# 等级 → 显示颜色 (BGR)
LEVEL_COLORS = {
    0: None,
    1: (0, 200, 255),    # 黄
    2: (0, 140, 255),    # 橙
    3: (0, 0, 255),      # 红
}


def main(*, dashboard=None, camera_id=None):
    print("打开摄像头...")
    with ExitStack() as resources:
        cam = Camera() if camera_id is None else Camera(camera_id=camera_id)
        resources.callback(cam.release)
        opened = cam.open()
        if not opened and dashboard is not None and camera_id is None and cam.camera_id != 0:
            cam.release()
            cam = Camera(camera_id=0)
            resources.callback(cam.release)
            opened = cam.open()
        if not opened:
            message = "无法打开摄像头。请检查设备编号或关闭正在占用摄像头的应用。"
            print(f"[ERROR] {message}")
            if dashboard is not None:
                dashboard.status("error", message)
            return

        print(f"   分辨率: {cam.resolution[0]}x{cam.resolution[1]}")
        detector = FaceMeshDetector()
        resources.callback(detector.close)
        if dashboard is None:
            resources.callback(cv2.destroyAllWindows)
        run_session(cam, detector, resources, dashboard)


def run_session(cam, detector, resources, dashboard):
    # 平滑器
    ear_smoother = EARSmoother(window_size=5)
    mar_smoother = MARSmoother(window_size=5)
    head_pose_smoother = HeadPoseSmoother(window_size=5)
    pitch_baseline = PitchBaseline(PITCH_BASELINE_DURATION_SEC, PITCH_BASELINE_MAX_SPREAD_DEG,
                                   MAX_OBSERVATION_GAP_SEC)
    perclos_counter = PERCLOSCounter(
        window_sec=PERCLOS_WINDOW_SEC,
        ear_threshold=EAR_THRESHOLD,
        max_gap_sec=MAX_OBSERVATION_GAP_SEC,
        blink_exempt_sec=BLINK_EXEMPT_SEC,   # 眨眼豁免（<0.3s 不计入闭眼）
    )

    # 眨眼动力学分析器（P0 升级：疲劳早期信号）
    blink_analyzer = BlinkAnalyzer(
        ear_threshold=EAR_THRESHOLD,
        window_sec=BLINK_WINDOW_SEC,
        slow_blink_sec=SLOW_BLINK_SEC,
        long_closure_sec=MICRO_SLEEP_SEC,   # ≥2s 视为长闭眼，不计眨眼
    )

    # 疲劳评分器（模块 07）
    scorer = FatigueScorer()
    distraction = None

    # 运行日志（每秒快照写入 CSV）
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"run_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    log_file = resources.enter_context(open(log_path, "w", newline="", encoding="utf-8-sig"))
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "t", "score", "level", "ear_avg", "mar", "perclos",
        "pitch", "yaw", "roll",
        "blink_rate", "blink_dur",
        "src_eye_close", "src_micro_sleep", "src_perclos",
        "src_yawn", "src_yawn_freq", "src_head_down", "src_yaw_dev",
        "src_blink_rate", "src_blink_dur",
        "face_valid", "pose_valid", "eye_valid", "perclos_valid_sec",
        "pitch_baseline", "pitch_relative",
    ])
    print(f"[LOG] 运行日志: {log_path}")

    # 哈欠计时器
    yawn_start_time = None       # 开始张嘴的时间戳
    yawn_count = 0               # 本段内哈欠次数
    yawn_in_progress = False     # 当前是否在"已确认的哈欠"中
    yawn_gap_start = None        # 宽容缓冲：MAR 跌回阈值以下的时间点

    # EAR 持续闭眼计数器
    eye_closed_start = None

    # 状态计时（持续低/偏头）
    head_down_start = None       # 开始低头的时间戳
    yaw_dev_start = None         # 开始偏头的时间戳

    # 报警状态
    last_alert_level = 0         # 上一次报警等级（用于蜂鸣触发）
    last_alert_time = 0.0        # 上一次报警时间（冷却）
    last_log_time = float("-inf")          # 上一次日志写入时间
    previous_frame_time = None
    _t0 = time.monotonic()            # 程序启动时间（日志时间基准）

    print("[OK] 疲劳评分器就绪。按 r 重置评分，q 退出\n")

    show_face_mesh = SHOW_FACE_MESH
    score_revision = 0
    # ── 主循环 ──────────────────────────────
    while dashboard is None or not dashboard.stop.is_set():
        resumed = False
        if dashboard is not None and dashboard.paused.is_set():
            perclos_counter.update(None, time.monotonic())
            if not resume_camera(cam, dashboard):
                break
            last_log_time = float("-inf")
            resumed = True

        if dashboard is not None:
            for action, enabled in dashboard.pending_actions():
                if action == "reset":
                    scorer.reset()
                    score_revision += 1
                elif action == "calibrate":
                    pitch_baseline.reset()
                    head_down_start = None
                    scorer.clear_head_down()
                elif action == "mesh":
                    show_face_mesh = enabled
                elif action == "distraction" and distraction is not None:
                    distraction.reset()
                    distraction.enabled = enabled
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.monotonic()
        dt = 0.0 if previous_frame_time is None else now - previous_frame_time
        previous_frame_time = now
        frame_gap = resumed or dt > MAX_OBSERVATION_GAP_SEC
        if frame_gap:
            pitch_baseline.interrupt()
            eye_closed_start = head_down_start = yaw_dev_start = None
            yawn_start_time = yawn_gap_start = None
            yawn_in_progress = False
            ear_smoother.reset()
            mar_smoother.reset()
            head_pose_smoother.reset()
            blink_analyzer.update(None, now)

        # 检测
        results = detector.detect(frame)

        # 画绿网
        if show_face_mesh:
            detector.draw_mesh(frame, results)

        # ── 特征提取 ──────────────────────────
        ear_left = ear_right = ear_avg = mar_val = 0.0
        pitch = yaw = roll = 0.0
        landmarks = detector.get_landmarks(results) if detector.has_face(results) else []
        face_found = len(landmarks) >= 478
        pose = calculate_head_pose(landmarks, w, h, invalid_as_none=True) if face_found else None
        pose_valid = pose is not None
        eye_valid = False

        if pose_valid:
            pitch, yaw, roll = head_pose_smoother.update(*pose)
            eye_valid = abs(pose[1]) <= YAW_EAR_GATE and abs(yaw) <= YAW_EAR_GATE
        else:
            head_pose_smoother.reset()

        if face_found:
            mar_val = mar_smoother.update(calculate_mar(landmarks))
        else:
            mar_smoother.reset()

        if eye_valid:
            ear_left_raw = calculate_ear(landmarks, LEFT_EYE_INDICES)
            ear_right_raw = calculate_ear(landmarks, RIGHT_EYE_INDICES)
            ear_left, ear_right, ear_avg = ear_smoother.update(ear_left_raw, ear_right_raw)
            blink_analyzer.update((ear_left_raw + ear_right_raw) / 2, now)
        else:
            ear_smoother.reset()
            blink_analyzer.update(None, now)

        perclos_counter.update(ear_avg if eye_valid else None, now)
        perclos_value, perclos_valid_sec = perclos_counter.get_stats()
        if eye_valid and ear_avg < EAR_THRESHOLD:
            if eye_closed_start is None:
                eye_closed_start = now
        else:
            eye_closed_start = None
        eye_closed_sec = now - eye_closed_start if eye_closed_start is not None else 0.0

        # ── 哈欠检测状态机 ────────────────────
        mar_over_threshold = face_found and mar_val > MAR_THRESHOLD
        if not face_found:
            yawn_start_time = yawn_gap_start = None
            yawn_in_progress = False

        yawn_elapsed = 0.0
        in_gap = False           # 是否处于宽容缓冲中
        yawn_just_completed = False  # 本帧刚确认一次新哈欠（事件标志）

        if mar_over_threshold:
            # 回到阈值以上 → 清除缓冲标记
            yawn_gap_start = None
            if yawn_start_time is None:
                yawn_start_time = now
            yawn_elapsed = now - yawn_start_time

            if yawn_elapsed >= MAR_YAWN_DURATION_SEC and not yawn_in_progress:
                yawn_count += 1
                yawn_in_progress = True
                yawn_just_completed = True
                print(f"[YAWN] 检测到哈欠 #{yawn_count}！（MAR={mar_val:.3f}, 持续{yawn_elapsed:.1f}s）")
        else:
            # MAR 低于阈值 → 启动宽容缓冲，不立刻重置
            if yawn_start_time is not None:
                if yawn_gap_start is None:
                    yawn_gap_start = now  # 开始计时间隙

                gap_duration = now - yawn_gap_start
                if gap_duration > MAR_YAWN_GRACE_SEC:
                    # 缓冲超时 → 确认中断，重置
                    yawn_start_time = None
                    yawn_gap_start = None
                    yawn_elapsed = 0.0
                    yawn_in_progress = False
                else:
                    # 还在缓冲期内 → 计时继续累积，假装没中断
                    yawn_elapsed = now - yawn_start_time
                    in_gap = True

        # ── 状态计时（低头 / 偏头）────────────
        pitch_usable = eye_valid and abs(pose[2]) <= PITCH_MAX_ROLL_DEG
        baseline_sample = pose[0] if pitch_usable and abs(pose[1]) <= PITCH_BASELINE_MAX_YAW_DEG else None
        pitch_baseline.update(baseline_sample, now)
        neutral_pitch = pitch_baseline.value
        effective_pitch = pitch - neutral_pitch if pitch_usable and neutral_pitch is not None else None
        pitch_down = effective_pitch is not None and effective_pitch > PITCH_DOWN_THRESHOLD

        if pitch_down:
            head_down_start = head_down_start if head_down_start is not None else now
        else:
            head_down_start = None
        head_down = (
            head_down_start is not None
            and (now - head_down_start) >= HEAD_DOWN_DURATION_SEC
        )

        if pose_valid and abs(yaw) > YAW_DEVIATION_THRESHOLD:
            yaw_dev_start = yaw_dev_start if yaw_dev_start is not None else now
        else:
            yaw_dev_start = None
        yaw_dev = (
            yaw_dev_start is not None
            and (now - yaw_dev_start) >= YAW_DEVIATION_DURATION_SEC
        )

        # ── 疲劳评分器更新（模块 07）──────────
        perclos_over = eye_valid and perclos_valid_sec > 0 and perclos_value > PERCLOS_THRESHOLD

        # 眨眼统计（疲劳早期信号）
        blink_stats = blink_analyzer.get_stats(now)
        blink_rate_over = eye_valid and blink_stats["blink_rate"] >= BLINK_RATE_THRESHOLD
        blink_dur_over = eye_valid and (
            blink_stats["avg_duration"] >= AVG_BLINK_DUR_THRESHOLD
            or blink_stats["slow_blink_rate"] >= SLOW_BLINK_RATE_THRESHOLD
        )

        scorer.update(dt, {
            "eye_closed_sec": eye_closed_sec,
            "yawn_just_completed": yawn_just_completed,
            "perclos_over": perclos_over,
            "head_down": head_down,
            "yaw_dev": yaw_dev,
            "blink_rate_over": blink_rate_over,
            "blink_dur_over": blink_dur_over,
            "face_valid": face_found,
        }, now=now, observed_dt=0.0 if frame_gap else dt)
        fatigue_score = scorer.get_score()
        fatigue_level, fatigue_label = scorer.get_level()
        active_sources = scorer.get_active_sources()

        # 报警蜂鸣（等级提升或处于 L2/L3 且冷却结束）
        if fatigue_level >= 2 and fatigue_level > last_alert_level:
            last_alert_level = fatigue_level
            last_alert_time = now
            if dashboard is None and _HAS_SOUND:
                winsound.Beep(1500 if fatigue_level == 2 else 2200, 400)
                print(f"[ALERT] {fatigue_label}！分数={fatigue_score:.1f} 来源={[s[0] for s in active_sources]}")
        elif fatigue_level < 2:
            last_alert_level = fatigue_level
        elif fatigue_level >= 2 and now - last_alert_time >= ALERT_COOLDOWN_SEC:
            last_alert_time = now
            if dashboard is None and _HAS_SOUND:
                winsound.Beep(1000, 200)

        # 运行日志：每秒快照一行
        if now - last_log_time >= 1.0:
            last_log_time = now
            src = {n: 0.0 for n in scorer._sources}
            for n, v in active_sources:
                src[n] = v
            log_writer.writerow([
                f"{now - _t0:.1f}", f"{fatigue_score:.2f}", fatigue_level,
                f"{ear_avg:.4f}", f"{mar_val:.4f}", f"{perclos_value:.3f}",
                f"{pitch:.1f}", f"{yaw:.1f}", f"{roll:.1f}",
                f"{blink_stats['blink_rate']:.1f}", f"{blink_stats['avg_duration']:.3f}",
                f"{src['eye_close']:.2f}", f"{src['micro_sleep']:.2f}", f"{src['perclos']:.2f}",
                f"{src['yawn']:.2f}", f"{src['yawn_freq']:.2f}", f"{src['head_down']:.2f}", f"{src['yaw_dev']:.2f}",
                f"{src['blink_rate']:.2f}", f"{src['blink_dur']:.2f}",
                int(face_found), int(pose_valid), int(eye_valid), f"{perclos_valid_sec:.3f}",
                f"{neutral_pitch:.1f}" if neutral_pitch is not None else "",
                f"{effective_pitch:.1f}" if effective_pitch is not None else "",
            ])
            log_file.flush()

        if dashboard is not None:
            dashboard.publish(frame, {
                "elapsed": round(now - _t0, 3),
                "fps": float(cam.fps),
                "camera": cam.camera_id,
                "resolution": [w, h],
                "level": fatigue_level,
                "score": float(fatigue_score),
                "revision": score_revision,
                "sources": {name: float(value) for name, value in active_sources},
                "face_valid": bool(face_found),
                "pose_valid": bool(pose_valid),
                "eye_valid": bool(eye_valid),
                "ear": float(ear_avg) if eye_valid else None,
                "ear_left": float(ear_left) if eye_valid else None,
                "ear_right": float(ear_right) if eye_valid else None,
                "mar": float(mar_val) if face_found else None,
                "perclos": float(perclos_value) if eye_valid and perclos_valid_sec > 0 else None,
                "perclos_valid_sec": float(perclos_valid_sec),
                "perclos_threshold": PERCLOS_THRESHOLD,
                "perclos_window": PERCLOS_WINDOW_SEC,
                "eye_closed_sec": float(eye_closed_sec) if eye_valid else None,
                "pitch": float(pitch) if pose_valid else None,
                "pitch_baseline": neutral_pitch,
                "pitch_relative": effective_pitch,
                "yaw": float(yaw) if pose_valid else None,
                "roll": float(roll) if pose_valid else None,
                "blink_rate": float(blink_stats["blink_rate"]) if eye_valid else None,
                "blink_ms": 1000 * float(blink_stats["avg_duration"]) if eye_valid else None,
                "yawns": yawn_count,
                "mesh": show_face_mesh,
                "distraction": {"available": False, "enabled": False, "result": None},
            })
            continue

        # ── 画面显示 ──────────────────────────
        y_offset = 30

        # FPS
        cv2.putText(frame, f"FPS: {cam.fps:.1f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 28

        # 人脸状态
        if face_found:
            cv2.putText(frame, "Face: DETECTED", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Face: NOT FOUND", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        y_offset += 28

        # ── EAR 显示 ──────────────────────────
        if eye_valid:
            color_left  = (0, 0, 255) if ear_left  < EAR_THRESHOLD else (0, 255, 0)
            color_right = (0, 0, 255) if ear_right < EAR_THRESHOLD else (0, 255, 0)

            cv2.putText(frame, f"EAR L: {ear_left:.3f}  R: {ear_right:.3f}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            y_offset += 22

            # 闭眼警告
            if eye_closed_sec >= EAR_CONSECUTIVE_FRAMES / FPS_TARGET:
                cv2.putText(frame, "EYES CLOSED!", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_left, 2)
                y_offset += 22
        y_offset += 6

        # ── MAR 显示 ──────────────────────────
        if face_found:
            mar_color = (0, 0, 255) if mar_over_threshold else (0, 255, 0)

            # 嘴巴状态标签
            if yawn_in_progress:
                mouth_state = "YAWNING!"
            elif in_gap:
                mouth_state = "GAP..."     # 宽容缓冲中
            elif mar_over_threshold:
                mouth_state = "OPEN"
            else:
                mouth_state = "CLOSED"

            cv2.putText(frame, f"MAR: {mar_val:.3f}  Mouth: {mouth_state}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, mar_color, 2)
            y_offset += 22

            # 哈欠计时
            if mar_over_threshold:
                timer_color = (0, 0, 255) if yawn_elapsed >= MAR_YAWN_DURATION_SEC else (0, 255, 255)
                cv2.putText(frame, f"Yawn Timer: {yawn_elapsed:.1f}s / {MAR_YAWN_DURATION_SEC}s", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, timer_color, 2)
                y_offset += 20

            # 哈欠计数
            if yawn_count > 0:
                cv2.putText(frame, f"Yawns: {yawn_count}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        y_offset += 6

        # ── PERCLOS 显示 ───────────────────────
        if eye_valid and perclos_valid_sec > 0:
            perclos_color = (0, 0, 255) if perclos_over else (0, 255, 255)
            cv2.putText(frame, f"PERCLOS: {perclos_value:.2f}  ({PERCLOS_WINDOW_SEC}s)", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, perclos_color, 2)
            y_offset += 22

            if perclos_over:
                cv2.putText(frame, "PERCLOS ALERT!", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y_offset += 22

        # ── 眨眼统计显示（P0：疲劳早期信号）────
        if eye_valid:
            blink_color = (0, 0, 255) if (blink_rate_over or blink_dur_over) else (0, 255, 255)
            cv2.putText(
                frame,
                f"Blink: {blink_stats['blink_rate']:.0f}/min dur {blink_stats['avg_duration']*1000:.0f}ms",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blink_color, 2)
            y_offset += 22

        # ── 疲劳评分显示 ───────────────────────
        score_color = LEVEL_COLORS[fatigue_level] or (255, 255, 255)
        cv2.putText(frame, f"Fatigue: {fatigue_score:.1f}  {fatigue_label}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, score_color, 2)
        y_offset += 22

        if active_sources:
            src_text = "+".join(f"{n}:{v:.1f}" for n, v in active_sources[:4])
            cv2.putText(frame, f"  src: {src_text}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # ── HeadPose 显示 ──────────────────────
        if pose_valid:
            y_offset += 6

            # 角度数值
            pitch_color = (0, 0, 255) if pitch_down else (0, 255, 255)
            yaw_color   = (0, 0, 255) if abs(yaw) > YAW_DEVIATION_THRESHOLD else (0, 255, 255)

            pitch_relative_text = f"{effective_pitch:+.1f}" if effective_pitch is not None else "--"
            cv2.putText(frame, f"Pitch: {pitch:.1f} ({pitch_relative_text})  Yaw: {yaw:.1f}  Roll: {roll:.1f}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pitch_color, 2)
            y_offset += 22

            # 低头警告（只限 effective_pitch 正值过大，仰头不警告）
            if pitch_down:
                cv2.putText(frame, "HEAD DOWN!", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y_offset += 22

            # 3D 坐标轴（画在鼻尖位置）
            if SHOW_HEAD_POSE_AXIS:
                draw_head_axes(frame, landmarks, w, h)

        # 嘴部标记点（3组上下唇，6个点）
            if SHOW_MOUTH_POINTS and len(landmarks) >= 478:
                from features import MOUTH_VERTICAL_PAIRS, MOUTH_CORNER_LEFT, MOUTH_CORNER_RIGHT
                for up, lo in MOUTH_VERTICAL_PAIRS:
                    for idx in (up, lo):
                        x, y = landmarks[idx]
                        px, py = int(x * w), int(y * h)
                        cv2.circle(frame, (px, py), 2, (255, 255, 0), -1)
                # 嘴角两点用青色
                for idx in (MOUTH_CORNER_LEFT, MOUTH_CORNER_RIGHT):
                    x, y = landmarks[idx]
                    px, py = int(x * w), int(y * h)
                    cv2.circle(frame, (px, py), 3, (255, 255, 0), -1)

        # 眼角标记点
        if SHOW_EYE_POINTS and eye_valid:
            for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
                x, y = landmarks[idx]
                px, py = int(x * w), int(y * h)
                cv2.circle(frame, (px, py), 2, (0, 255, 255), -1)

        # ── 顶部报警横幅（等级 ≥ L1 时显示）────
        if fatigue_level >= 1:
            banner_color = LEVEL_COLORS[fatigue_level]
            flash = (fatigue_level == 3 and int(now * 2) % 2 == 0)
            if flash:
                banner_color = (255, 255, 255)  # L3 红白闪烁
            cv2.rectangle(frame, (0, 0), (w, 36), banner_color, -1)
            banner_text = f"{fatigue_label}  Score:{fatigue_score:.1f}"
            if active_sources:
                banner_text += "  |  " + " ".join(n for n, _ in active_sources[:4])
            cv2.putText(frame, banner_text, (w // 2 - 220, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.imshow("Fatigue Detection", frame)

        # ── 按键 ──────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[EXIT] 退出")
            break
        elif key == ord('s'):
            os.makedirs(SCREENSHOT_DIR, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(SCREENSHOT_DIR, f"MAR_{timestamp}.png")
            cv2.imwrite(filename, frame)
            print(f"[SHOT] 截图: {filename}")
        elif key == ord('r'):
            scorer.reset()
            print("[RESET] 疲劳评分已重置")
        elif key == ord('c'):
            pitch_baseline.reset()
            head_down_start = None
            scorer.clear_head_down()
            print("[CALIBRATION] 请自然坐姿看向前方，重新采集 Pitch 基线。")

    if dashboard is not None:
        if dashboard.stop.is_set():
            dashboard.status("stopped", "监测已停止。重新启动程序即可继续。")
        else:
            dashboard.status("error", "摄像头画面中断。请检查连接后重启程序。")
    print(f"[OK] 运行日志已保存: {log_path}")


def resume_camera(cam, dashboard):
    cam.release()
    message = "监测已暂停，点击「继续监测」接着记录。"
    while not dashboard.stop.is_set():
        dashboard.status("paused", message)
        if not dashboard.wait_for_resume():
            break
        dashboard.status("resuming", "正在重新打开摄像头")
        if cam.open():
            return not dashboard.stop.is_set()
        cam.release()
        dashboard.paused.set()
        message = "摄像头暂不可用。关闭占用它的应用后，点击「继续监测」重试。"
    return False


def cli():
    parser = argparse.ArgumentParser(description="DMS 疲劳检测")
    parser.add_argument("--web", action="store_true", help="打开本机浏览器仪表盘")
    parser.add_argument("--camera", type=int, help="指定摄像头设备编号")
    parser.add_argument("--port", type=int, default=8765, help="浏览器仪表盘端口")
    parser.add_argument("--no-open", action="store_true", help="运行仪表盘但不自动打开浏览器")
    args = parser.parse_args()
    if args.web:
        from web_dashboard import run_dashboard
        run_dashboard(main, args.port, args.camera, not args.no_open)
    else:
        main(camera_id=args.camera)


if __name__ == "__main__":
    cli()

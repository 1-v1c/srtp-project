"""
疲劳检测系统 — 主程序入口
Phase 1: 摄像头 + Face Mesh 人脸关键点
Phase 2: + EAR 眼部长宽比实时显示
Phase 3: + MAR 嘴部长宽比 + 哈欠计时

按 'q' 退出 | 按 's' 截图
"""

import os
import cv2
import time

from camera import Camera
from detector import FaceMeshDetector
from features import (
    calculate_ear, calculate_mar,
    LEFT_EYE_INDICES, RIGHT_EYE_INDICES,
    EARSmoother, MARSmoother,
)
from config import (
    EAR_THRESHOLD, EAR_CONSECUTIVE_FRAMES,
    MAR_THRESHOLD, MAR_YAWN_DURATION_SEC, MAR_YAWN_GRACE_SEC,
    SHOW_FACE_MESH,
    SHOW_EYE_POINTS, SHOW_MOUTH_POINTS,
)


def main():
    # ── 初始化 ──────────────────────────────
    print("📷 打开摄像头...")
    cam = Camera()
    if not cam.open():
        print("❌ 无法打开摄像头！")
        return
    print(f"   分辨率: {cam.resolution[0]}x{cam.resolution[1]}")

    print("🧠 加载 MediaPipe Face Mesh...")
    detector = FaceMeshDetector()
    print("   就绪！\n")

    # 平滑器
    ear_smoother = EARSmoother(window_size=5)
    mar_smoother = MARSmoother(window_size=5)

    # 哈欠计时器
    yawn_start_time = None       # 开始张嘴的时间戳
    yawn_count = 0               # 本段内哈欠次数
    yawn_in_progress = False     # 当前是否在"已确认的哈欠"中
    yawn_gap_start = None        # 宽容缓冲：MAR 跌回阈值以下的时间点

    # EAR 持续闭眼计数器
    eye_closed_counter = 0

    # ── 主循环 ──────────────────────────────
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # 检测
        results = detector.detect(frame)

        # 画绿网
        if SHOW_FACE_MESH:
            detector.draw_mesh(frame, results)

        # ── 特征提取 ──────────────────────────
        ear_left = ear_right = ear_avg = 0.0
        mar_val = 0.0
        face_found = detector.has_face(results)
        landmarks = []

        if face_found:
            landmarks = detector.get_landmarks(results)

            if len(landmarks) >= 478:
                # --- EAR ---
                ear_left = calculate_ear(landmarks, LEFT_EYE_INDICES)
                ear_right = calculate_ear(landmarks, RIGHT_EYE_INDICES)
                ear_left, ear_right, ear_avg = ear_smoother.update(ear_left, ear_right)

                # --- 持续闭眼计数 ---
                if ear_avg < EAR_THRESHOLD:
                    eye_closed_counter += 1
                else:
                    eye_closed_counter = 0

                # --- MAR ---
                mar_val = calculate_mar(landmarks)
                mar_val = mar_smoother.update(mar_val)

        # ── 哈欠检测状态机 ────────────────────
        fps = cam.fps if cam.fps > 0 else 30
        mar_over_threshold = mar_val > MAR_THRESHOLD

        yawn_elapsed = 0.0
        in_gap = False  # 是否处于宽容缓冲中

        if mar_over_threshold:
            # 回到阈值以上 → 清除缓冲标记
            yawn_gap_start = None
            if yawn_start_time is None:
                yawn_start_time = time.time()
            yawn_elapsed = time.time() - yawn_start_time

            if yawn_elapsed >= MAR_YAWN_DURATION_SEC and not yawn_in_progress:
                yawn_count += 1
                yawn_in_progress = True
                print(f"🥱 检测到哈欠 #{yawn_count}！（MAR={mar_val:.3f}, 持续{yawn_elapsed:.1f}s）")
        else:
            # MAR 低于阈值 → 启动宽容缓冲，不立刻重置
            if yawn_start_time is not None:
                if yawn_gap_start is None:
                    yawn_gap_start = time.time()  # 开始计时间隙

                gap_duration = time.time() - yawn_gap_start
                if gap_duration > MAR_YAWN_GRACE_SEC:
                    # 缓冲超时 → 确认中断，重置
                    yawn_start_time = None
                    yawn_gap_start = None
                    yawn_elapsed = 0.0
                    yawn_in_progress = False
                else:
                    # 还在缓冲期内 → 计时继续累积，假装没中断
                    yawn_elapsed = time.time() - yawn_start_time
                    in_gap = True

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
        if face_found:
            color_left  = (0, 0, 255) if ear_left  < EAR_THRESHOLD else (0, 255, 0)
            color_right = (0, 0, 255) if ear_right < EAR_THRESHOLD else (0, 255, 0)

            cv2.putText(frame, f"EAR L: {ear_left:.3f}  R: {ear_right:.3f}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            y_offset += 22

            # 闭眼警告
            if eye_closed_counter >= EAR_CONSECUTIVE_FRAMES:
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
        if SHOW_EYE_POINTS and face_found and len(landmarks) >= 478:
            for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
                x, y = landmarks[idx]
                px, py = int(x * w), int(y * h)
                cv2.circle(frame, (px, py), 2, (0, 255, 255), -1)

        cv2.imshow("Fatigue Detection", frame)

        # ── 按键 ──────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("👋 退出")
            break
        elif key == ord('s'):
            os.makedirs("data/screenshots", exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"data/screenshots/MAR_{timestamp}.png"
            cv2.imwrite(filename, frame)
            print(f"📸 截图: {filename}")

    # ── 清理 ────────────────────────────────
    cam.release()
    detector.close()
    cv2.destroyAllWindows()
    print("✅ 已释放所有资源")


if __name__ == "__main__":
    main()

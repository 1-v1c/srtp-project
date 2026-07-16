"""
CalibTool — 疲劳检测特征标定采集器
鼠标点选标签 · 空格录制 · 自动统计 · CSV 导出
覆盖 EAR / MAR / HeadPose 全部标定流程
"""

import cv2
import numpy as np
import time
import os
from collections import deque

from PIL import Image, ImageDraw, ImageFont

from camera import Camera
from detector import FaceMeshDetector
from features import (
    calculate_ear, calculate_mar, calculate_head_pose,
    LEFT_EYE_INDICES, RIGHT_EYE_INDICES,
    MOUTH_VERTICAL_PAIRS, MOUTH_CORNER_LEFT, MOUTH_CORNER_RIGHT,
    EARSmoother, MARSmoother, HeadPoseSmoother,
)
from logger import CalibLogger
from config import (
    EAR_THRESHOLD, MAR_THRESHOLD,
    SHOW_FACE_MESH, SHOW_EYE_POINTS, SHOW_MOUTH_POINTS,
    FRAME_WIDTH, FRAME_HEIGHT,
)

# ══════════════════════════════════════════════════════════════
# 设计系统
# ══════════════════════════════════════════════════════════════

# ── 色彩 (BGR) ─────────────────────────────────────────────
#            BGR       RGB hex     用途
CLR_BG      = (30, 28, 28)    # #1C1C1E  深色背景
CLR_SURFACE = (44, 44, 46)    # #2C2C2E  卡片 / 按钮底
CLR_BORDER  = (58, 56, 56)    # #38383A  分隔线 / 边框
CLR_PRIMARY = (255, 122, 0)   # #007AFF  选中/激活 (iOS 蓝)
CLR_SUCCESS = (89, 199, 52)   # #34C759  EAR 正常 / 导出 (iOS 绿)
CLR_DANGER  = (48, 59, 255)   # #FF3B30  录制 / 删除 (iOS 红)
CLR_WARNING = (0, 149, 255)   # #FF9500  MAR (iOS 橙)
CLR_TEXT    = (235, 235, 240) # #F0F0F0  主文字 (off-white, 非纯白)
CLR_TEXT_DIM= (156, 156, 163) # #9C9CA3  次要 / 提示文字

# ── 布局 (全部基于 8px 网格) ────────────────────────────────
TOP_BAR_H    = 64
BOTTOM_BAR_H = 56
SIDE_PANEL_W = 272
CORNER_R     = 8
BTN_H        = 32
BTN_GAP      = 8
PAD          = 16          # 基础内边距
PAD_S        = 8           # 小间距

# ── 排版层级 (4 级) ────────────────────────────────────────
SCALE_TITLE   = 0.60   # 面板标题
SCALE_BODY    = 0.50   # FPS / 按钮 / 标签
SCALE_CAPTION = 0.42   # 次级信息 / 卡片数据
SCALE_SMALL   = 0.35   # 提示 / 时间戳

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD  = cv2.FONT_HERSHEY_DUPLEX

# ── 阶段 & 标签 ────────────────────────────────────────────
PHASES = {
    "EAR":  ["正常睁眼", "眯眼", "完全闭眼", "快速眨眼", "侧脸睁眼"],
    "MAR":  ["正常闭嘴", "微笑", "正常说话", "张嘴惊讶", "大笑", "真打哈欠"],
    "HEAD": ["正对屏幕", "低头", "抬头", "左转", "右转", "左倾", "右倾"],
}

PHASE_METRICS = {
    "EAR":  ["EAR_L", "EAR_R", "EAR_avg"],
    "MAR":  ["MAR"],
    "HEAD": ["Pitch", "Yaw", "Roll"],
}


# ══════════════════════════════════════════════════════════════
# 绘图原语
# ══════════════════════════════════════════════════════════════

# ── PIL 中文渲染（cv2.putText 不支持 CJK）─────────────────

_CN_FONT_CACHE = {}

def _get_cn_font(size: int) -> ImageFont.FreeTypeFont:
    if size in _CN_FONT_CACHE:
        return _CN_FONT_CACHE[size]
    for fp in [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]:
        if os.path.exists(fp):
            font = ImageFont.truetype(fp, size)
            _CN_FONT_CACHE[size] = font
            return font
    return ImageFont.load_default()

def _is_ascii(text: str) -> bool:
    return all(ord(c) < 128 for c in text)

def _render_pil_text(text: str, size: int, color_bgr: tuple
                     ) -> tuple[np.ndarray | None, int, int]:
    font = _get_cn_font(size)
    color_rgba = (color_bgr[2], color_bgr[1], color_bgr[0], 255)
    dummy = Image.new('RGBA', (1, 1))
    draw_dummy = ImageDraw.Draw(dummy)
    bbox = draw_dummy.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if tw <= 0 or th <= 0:
        return None, 0, 0
    pil_img = Image.new('RGBA', (tw + 4, th + 4), (0, 0, 0, 0))
    draw = ImageDraw.Draw(pil_img)
    draw.text((2 - bbox[0], 2 - bbox[1]), text, font=font, fill=color_rgba)
    return np.array(pil_img), tw, th

def _paste_cn(img: np.ndarray, rgba: np.ndarray, px: int, py: int):
    h, w = rgba.shape[:2]
    x1, y1 = max(0, px), max(0, py)
    x2, y2 = min(img.shape[1], px + w), min(img.shape[0], py + h)
    ox, oy = x1 - px, y1 - py
    if x2 <= x1 or y2 <= y1:
        return
    alpha = rgba[oy:oy + (y2 - y1), ox:ox + (x2 - x1), 3] / 255.0
    for c in range(3):
        roi = img[y1:y2, x1:x2, c]
        src = rgba[oy:oy + (y2 - y1), ox:ox + (x2 - x1), c]
        img[y1:y2, x1:x2, c] = (alpha * src + (1 - alpha) * roi).astype(np.uint8)

# ── 文字绘制 ──────────────────────────────────────────────

def _text_size(text, scale=SCALE_BODY, thickness=1, font=FONT):
    """测量文本尺寸。中文自动走 PIL，ASCII 走 cv2"""
    if not _is_ascii(text):
        pil_size = max(12, int(scale * 32))
        _, tw, th = _render_pil_text(text, pil_size, CLR_TEXT)
        return tw, th
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    return w, h

def _draw_text(img, text, x, y, scale=SCALE_BODY, color=CLR_TEXT, thickness=2,
               font=FONT, align="left"):
    """统一文字绘制：ASCII 走 cv2，其余走 PIL"""
    x, y = int(x), int(y)
    if not _is_ascii(text):
        pil_size = max(12, int(scale * 32))
        rgba, tw, th = _render_pil_text(text, pil_size, color)
        if rgba is not None:
            px = x - tw // 2 if align == "center" else x
            py = y - th
            _paste_cn(img, rgba, px, py + 4)
        return
    if align == "center":
        tw, _ = _text_size(text, scale, thickness, font)
        x -= tw // 2
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

# ── 圆角矩形 ──────────────────────────────────────────────

def _rounded_rect(img, x, y, w, h, r, color, alpha=1.0):
    """绘制填充圆角矩形（支持半透明），自动裁剪到画面边界"""
    x, y, w, h, r = int(x), int(y), int(w), int(h), int(r)
    if w <= 0 or h <= 0:
        return
    # 裁剪到图像边界
    H_img, W_img = img.shape[:2]
    if x >= W_img or y >= H_img:
        return
    if x + w > W_img:
        w = W_img - x
    if y + h > H_img:
        h = H_img - y
    if w <= 0 or h <= 0:
        return
    r = min(r, w // 2, h // 2)

    # 构建圆角掩码
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (r, 0), (w - r - 1, h - 1), 255, -1)
    cv2.rectangle(mask, (0, r), (w - 1, h - r - 1), 255, -1)
    cv2.circle(mask, (r, r), r, 255, -1)
    cv2.circle(mask, (w - r - 1, r), r, 255, -1)
    cv2.circle(mask, (r, h - r - 1), r, 255, -1)
    cv2.circle(mask, (w - r - 1, h - r - 1), r, 255, -1)

    roi = img[y:y + h, x:x + w]
    if alpha >= 1.0:
        roi[mask > 0] = color
    else:
        blended = roi.copy()
        blended[mask > 0] = color
        cv2.addWeighted(blended, alpha, roi, 1 - alpha, 0, roi)

def _rounded_outline(img, x, y, w, h, r, color, thickness=1):
    """绘制圆角边框，自动裁剪到画面边界"""
    x, y, w, h, r = int(x), int(y), int(w), int(h), int(r)
    if w <= 0 or h <= 0:
        return
    # 裁剪到图像边界
    H_img, W_img = img.shape[:2]
    if x >= W_img or y >= H_img:
        return
    if x + w > W_img:
        w = W_img - x
    if y + h > H_img:
        h = H_img - y
    if w <= 0 or h <= 0:
        return
    r = min(r, w // 2, h // 2)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (r, 0), (w - r - 1, h - 1), 255, -1)
    cv2.rectangle(mask, (0, r), (w - 1, h - r - 1), 255, -1)
    cv2.circle(mask, (r, r), r, 255, -1)
    cv2.circle(mask, (w - r - 1, r), r, 255, -1)
    cv2.circle(mask, (r, h - r - 1), r, 255, -1)
    cv2.circle(mask, (w - r - 1, h - r - 1), r, 255, -1)

    kernel = np.ones((thickness, thickness), dtype=np.uint8)
    inner = cv2.erode(mask, kernel)
    border_mask = mask - inner

    roi = img[y:y + h, x:x + w]
    roi[border_mask > 0] = color

# ── 毛玻璃面板 ────────────────────────────────────────────

def _frosted_panel(img, x, y, w, h, alpha=0.88):
    """半透明深色面板"""
    x, y, w, h = int(x), int(y), int(w), int(h)
    if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
        return
    roi = img[y:y + h, x:x + w]
    overlay = np.full_like(roi, CLR_BG)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)

# ── 按钮 ─────────────────────────────────────────────────

def _draw_btn(img, x, y, w, h, text, selected=False, accent=CLR_PRIMARY):
    """圆角按钮：选中=实心蓝底白字，常态=半透明灰底"""
    r = CORNER_R
    if selected:
        _rounded_rect(img, x, y, w, h, r, accent)
        _draw_text(img, text, x + w // 2, y + h // 2 + 5,
                   SCALE_BODY, (255, 255, 255), 1, align="center")
    else:
        _rounded_rect(img, x, y, w, h, r, CLR_SURFACE, alpha=0.92)
        _rounded_outline(img, x, y, w, h, r, (100, 98, 98), thickness=1)
        _draw_text(img, text, x + w // 2, y + h // 2 + 5,
                   SCALE_BODY, CLR_TEXT_DIM, 1, align="center")

# ── 记录卡片 ──────────────────────────────────────────────

def _draw_card(img, x, y, w, rec, selected=False):
    """侧面板记录卡片"""
    h = 46
    r = CORNER_R // 2

    # 底色 + 选中边框
    if selected:
        _rounded_rect(img, x, y, w, h, r, CLR_SURFACE)
        _rounded_outline(img, x, y, w, h, r, CLR_PRIMARY, thickness=2)
    else:
        _rounded_rect(img, x, y, w, h, r, CLR_SURFACE, alpha=0.6)

    # 左侧色条 (phase 标识)
    phase_colors = {"EAR": CLR_SUCCESS, "MAR": CLR_WARNING, "HEAD": CLR_PRIMARY}
    bar_c = phase_colors.get(rec["phase"], CLR_TEXT_DIM)
    _rounded_rect(img, x + PAD_S // 2, y + PAD_S, 4, h - PAD_S * 2, 2, bar_c)

    # 标签名（主文字）
    _draw_text(img, rec['label'], x + PAD + PAD_S, y + PAD,
               SCALE_BODY, CLR_TEXT, 1)
    # 时长（次要）
    _draw_text(img, f"{rec['duration']:.1f}s", x + PAD + PAD_S, y + PAD + 18,
               SCALE_SMALL, CLR_TEXT_DIM, 1)
    # 右侧数值摘要
    val = rec.get("summary", "")
    if val:
        tw, _ = _text_size(val, SCALE_SMALL, 1)
        _draw_text(img, val, x + w - tw - PAD, y + PAD + 10,
                   SCALE_SMALL, CLR_TEXT_DIM, 1)
    # 选中删除标记
    if selected:
        _draw_text(img, "×", x + w - PAD - 4, y + PAD,
                   SCALE_BODY, CLR_DANGER, 2, align="center")


# ══════════════════════════════════════════════════════════════
# CalibTool
# ══════════════════════════════════════════════════════════════

class CalibTool:
    def __init__(self):
        # ── 硬件 ──
        self.cam = Camera()
        self.detector = FaceMeshDetector()
        self.ear_smoother = EARSmoother(5)
        self.mar_smoother = MARSmoother(5)
        self.head_pose_smoother = HeadPoseSmoother(5)

        # ── UI 状态 ──
        self.phase = "EAR"
        self.labels = PHASES[self.phase]
        self.selected_label_idx = 0
        self.recording = False
        self.record_start = 0.0
        self.record_elapsed = 0.0
        self.record_frames: list[dict] = []
        self.records: list[dict] = []
        self.selected_record_idx = -1
        self._card_rects: list[tuple] = []
        self._pending_delete = False  # Backspace 首次按提示，再按才删
        self._pending_quit = False    # Q 首次按提示，再按才退出
        self._scroll_offset = 0       # 记录列表滚动位置
        self._max_side_cards = 5      # 侧面板最大可见卡片（_draw_side_panel 中更新）

        # ── 日志 ──
        _proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.logger = CalibLogger(log_dir=os.path.join(_proj_dir, "data", "calibration"))
        self.logger.open_session()

        # ── 尺寸 ──
        self.cam_w, self.cam_h = FRAME_WIDTH, FRAME_HEIGHT
        self.canvas_w = self.cam_w + SIDE_PANEL_W
        self.canvas_h = self.cam_h

        # ── 按钮热区缓存 ──
        self._phase_btns = []
        self._label_btns = []
        self._rec_btn = None
        self._export_btn = None

        # ── 当前帧数据 ──
        self.ear_left = self.ear_right = self.ear_avg = 0.0
        self.mar_val = 0.0
        self.pitch = self.yaw = self.roll = 0.0
        self.face_found = False
        self.fps_val = 0.0

    # ══════════════════════════════════════════════════════
    # 运行
    # ══════════════════════════════════════════════════════

    def run(self):
        if not self.cam.open():
            print("[ERROR] 无法打开摄像头")
            return
        print(f" {self.cam_w}×{self.cam_h}  |  鼠标点选标签  |   录制  |  Q 退出")
        cv2.namedWindow("CalibTool", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("CalibTool", self._on_mouse)

        try:
            while True:
                ret, frame_raw = self.cam.read()
                if not ret:
                    break
                frame_raw = cv2.flip(frame_raw, 1)
                self._process_frame(frame_raw)
                canvas = self._build_canvas(frame_raw)
                cv2.imshow("CalibTool", canvas)

                key = cv2.waitKey(1) & 0xFF

                # 非 Q 键 → 重置退出确认
                if key != ord('q'):
                    self._pending_quit = False

                if key == ord('q'):
                    if self.records and not self._pending_quit:
                        self._pending_quit = True
                        print(f"Press Q again to quit ({len(self.records)} records will be auto-exported)")
                    else:
                        if self.records:
                            self._export_csv()
                        break
                elif key == ord(' '):
                    self._toggle_recording()
                elif key == ord('s'):
                    self._screenshot(canvas)
                elif key == 9:          # Tab
                    self.selected_label_idx = (
                        (self.selected_label_idx + 1) % len(self.labels))
                elif key == ord('b'):    # 上一个标签
                    self.selected_label_idx = (
                        (self.selected_label_idx - 1) % len(self.labels))
                elif key == ord('1'):
                    self._switch_phase("EAR")
                elif key == ord('2'):
                    self._switch_phase("MAR")
                elif key == ord('3'):
                    self._switch_phase("HEAD")
                elif key == ord('e'):
                    self._export_csv()
                elif key in (255, 8, 127):  # Delete / Backspace
                    self._delete_selected_record()
                elif key == ord('h'):
                    print("=== Keyboard Shortcuts ===")
                    print("  1/2/3  - Switch phase (EAR/MAR/HEAD)")
                    print("  Space  - Start/Stop recording")
                    print("  Tab    - Next label")
                    print("  B      - Previous label")
                    print("  E      - Export CSV")
                    print("  S      - Screenshot")
                    print("  Q      - Quit (press twice to confirm)")
                    print("  H      - This help")
        finally:
            self.cam.release()
            self.detector.close()
            self.logger.close()
            cv2.destroyAllWindows()
            print("[OK] 已释放资源")

    # ══════════════════════════════════════════════════════
    # 帧处理
    # ══════════════════════════════════════════════════════

    def _process_frame(self, frame):
        results = self.detector.detect(frame)
        self.face_found = self.detector.has_face(results)
        self.fps_val = self.cam.fps

        if SHOW_FACE_MESH:
            self.detector.draw_mesh(frame, results)

        self.ear_left = self.ear_right = self.ear_avg = 0.0
        self.mar_val = 0.0
        self.pitch = self.yaw = self.roll = 0.0

        if self.face_found:
            lm = self.detector.get_landmarks(results)
            if len(lm) >= 478:
                el = calculate_ear(lm, LEFT_EYE_INDICES)
                er = calculate_ear(lm, RIGHT_EYE_INDICES)
                self.ear_left, self.ear_right, self.ear_avg = \
                    self.ear_smoother.update(el, er)
                self.mar_val = self.mar_smoother.update(calculate_mar(lm))
                # HeadPose
                p, y, r = calculate_head_pose(lm, self.cam_w, self.cam_h)
                self.pitch, self.yaw, self.roll = \
                    self.head_pose_smoother.update(p, y, r)

        if self.recording:
            self.record_elapsed = time.time() - self.record_start
            self.record_frames.append({
                "t": self.record_elapsed,
                "ear_l": self.ear_left,
                "ear_r": self.ear_right,
                "ear_avg": self.ear_avg,
                "mar": self.mar_val,
                "pitch": self.pitch,
                "yaw": self.yaw,
                "roll": self.roll,
            })

    # ══════════════════════════════════════════════════════
    # 画布组装
    # ══════════════════════════════════════════════════════

    def _build_canvas(self, frame):
        H, W = self.canvas_h, self.canvas_w
        base = np.full((H, W, 3), CLR_BG, dtype=np.uint8)
        base[0:H, 0:self.cam_w] = frame

        self._draw_top_bar(base)           # 只覆盖 0..cam_w
        self._draw_side_panel(base)         # 覆盖 cam_w..canvas_w
        self._draw_phase_buttons(base)      # 在侧面板之上绘制按钮，不被遮挡
        self._draw_bottom_bar(base)

        if self.recording:
            self._draw_recording_overlay(base)

        return base

    # ══════════════════════════════════════════════════════
    # 顶部栏
    # ══════════════════════════════════════════════════════

    def _draw_top_bar(self, img):
        _frosted_panel(img, 0, 0, self.cam_w, TOP_BAR_H, alpha=0.94)
        # 底部分隔线
        _rounded_rect(img, 0, TOP_BAR_H - 1, self.cam_w, 1, 0, CLR_BORDER)
        py = PAD

        # 第一行：状态
        fps_txt = f"FPS {self.fps_val:.0f}"
        face_txt = "● Face" if self.face_found else "○ No Face"
        face_c = CLR_SUCCESS if self.face_found else CLR_TEXT_DIM

        _draw_text(img, fps_txt, PAD, py + 14, SCALE_BODY, CLR_TEXT)
        _draw_text(img, face_txt, 80, py + 14, SCALE_CAPTION, face_c)

        # 第二行：实时数据
        if self.face_found:
            if self.phase == "HEAD":
                _draw_text(img, f"Pitch {self.pitch:.1f}  Yaw {self.yaw:.1f}  Roll {self.roll:.1f}",
                           PAD, py + 32, SCALE_CAPTION, CLR_PRIMARY, 1)
            else:
                _draw_text(img, f"EAR {self.ear_avg:.3f}", PAD, py + 32,
                           SCALE_CAPTION, CLR_SUCCESS, 1)
                _draw_text(img, f"MAR {self.mar_val:.3f}", PAD + 120, py + 32,
                           SCALE_CAPTION, CLR_WARNING, 1)
        else:
            _draw_text(img, "等待人脸…", PAD, py + 32, SCALE_SMALL, CLR_TEXT_DIM)

    # ══════════════════════════════════════════════════════
    # 右上角阶段切换按钮（在侧面板之上绘制，不被磨砂遮挡）
    # ══════════════════════════════════════════════════════

    def _draw_phase_buttons(self, img):
        """侧面板上层绘制 HEAD / MAR / EAR 三个按钮"""
        self._phase_btns.clear()
        px = self.canvas_w - PAD
        for ph in reversed(["HEAD", "MAR", "EAR"]):
            tw, _ = _text_size(ph, SCALE_BODY, 1)
            bw = tw + 24
            px -= bw
            sel = (ph == self.phase)
            _draw_btn(img, px, PAD, bw, BTN_H, ph, sel)
            self._phase_btns.append((px, PAD, bw, BTN_H, ph))
            px -= BTN_GAP  # 按钮间距

    # ══════════════════════════════════════════════════════
    # 底部栏
    # ══════════════════════════════════════════════════════

    def _draw_bottom_bar(self, img):
        by = self.canvas_h - BOTTOM_BAR_H
        _frosted_panel(img, 0, by, self.canvas_w, BOTTOM_BAR_H)
        _rounded_rect(img, 0, by, self.canvas_w, 1, 0, CLR_BORDER)
        by_c = by + (BOTTOM_BAR_H - BTN_H) // 2

        # ── 录制按钮（右侧固定，独立醒目） ──
        REC_W = 88
        REC_GAP = 24
        rec_x = self.cam_w - PAD - REC_W

        if self.recording:
            # 录制中：红底白字，醒目
            _rounded_rect(img, rec_x, by_c, REC_W, BTN_H, CORNER_R, CLR_DANGER)
            _draw_text(img, "[STOP] 停止", rec_x + REC_W // 2, by_c + BTN_H // 2 + 5,
                       SCALE_BODY, (255, 255, 255), 1, align="center")
        else:
            # 空闲：灰底灰字，低调
            _rounded_rect(img, rec_x, by_c, REC_W, BTN_H, CORNER_R,
                          CLR_SURFACE, alpha=0.5)
            _rounded_outline(img, rec_x, by_c, REC_W, BTN_H, CORNER_R, CLR_BORDER)
            _draw_text(img, "[REC] 录制", rec_x + REC_W // 2, by_c + BTN_H // 2 + 5,
                       SCALE_BODY, CLR_TEXT_DIM, 1, align="center")
        self._rec_btn = (rec_x, by_c, REC_W, BTN_H)

        # ── 标签按钮（左侧，宽度自适应） ──
        n = len(self.labels)
        avail = rec_x - PAD - REC_GAP
        btn_w = min(100, max(60, (avail - (n - 1) * BTN_GAP) // n))

        self._label_btns.clear()
        lx = PAD
        for i, label in enumerate(self.labels):
            sel = (i == self.selected_label_idx) and not self.recording
            _draw_btn(img, lx, by_c, btn_w, BTN_H, label, sel)
            self._label_btns.append((lx, by_c, btn_w, BTN_H, i))
            lx += btn_w + BTN_GAP

        # ── 导出按钮（侧面板区域） ──
        exp_x = self.cam_w + PAD
        _draw_btn(img, exp_x, by_c, 56, BTN_H, "导出", accent=CLR_SUCCESS)
        self._export_btn = (exp_x, by_c, 56, BTN_H)

    # ══════════════════════════════════════════════════════
    # 侧面板
    # ══════════════════════════════════════════════════════

    def _draw_side_panel(self, img):
        sx = self.cam_w
        _frosted_panel(img, sx, 0, SIDE_PANEL_W, self.canvas_h, alpha=0.92)
        # 左侧分隔线
        _rounded_rect(img, sx, 0, 1, self.canvas_h, 0, CLR_BORDER)

        # 标题行
        _draw_text(img, "采集记录", sx + PAD, TOP_BAR_H + PAD,
                   SCALE_TITLE, CLR_TEXT, 2, font=FONT_BOLD)
        n = len(self.records)
        _draw_text(img, f"({n})", sx + PAD + 90, TOP_BAR_H + PAD,
                   SCALE_CAPTION, CLR_TEXT_DIM)

        # 分隔线
        sep_y = TOP_BAR_H + PAD + 12
        _rounded_rect(img, sx + PAD, sep_y, SIDE_PANEL_W - PAD * 2, 1, 0, CLR_BORDER)

        # 记录卡片列表（带滚轮翻页）
        card_h = 48
        hint_y = self.canvas_h - BOTTOM_BAR_H - PAD
        avail_h = hint_y - (sep_y + PAD_S) - PAD_S * 2
        max_visible = max(1, int(avail_h // (card_h + PAD_S)))
        self._max_side_cards = max_visible
        total = len(self.records)

        # 计算可见区间（最新记录在底部，滚轮向上看旧记录）
        if total > max_visible:
            self._scroll_offset = min(self._scroll_offset, total - max_visible)
            start = total - max_visible - self._scroll_offset
            visible = self.records[start:start + max_visible]
        else:
            self._scroll_offset = 0
            visible = self.records[-max_visible:]

        self._card_rects.clear()
        for i, rec in enumerate(reversed(visible)):
            real_idx = total - 1 - i - self._scroll_offset
            cy = sep_y + PAD_S + i * (card_h + PAD_S)
            selected = (real_idx == self.selected_record_idx)
            _draw_card(img, sx + PAD, cy, SIDE_PANEL_W - PAD * 2, rec, selected)
            self._card_rects.append(
                (sx + PAD, cy, SIDE_PANEL_W - PAD * 2, card_h, real_idx))

        # 滚动指示器
        if total > max_visible:
            indicator = ""
            if self._scroll_offset < total - max_visible:
                indicator += f"^ older({self._scroll_offset}) "
            if self._scroll_offset > 0:
                indicator += "v newer"
            if indicator:
                _draw_text(img, indicator.strip(), sx + PAD, sep_y - 4,
                           SCALE_SMALL, CLR_WARNING, 1)

        # 底部提示（hint_y 已在上面计算）
        if self._pending_delete:
            hint = "再次 Backspace 确认删除"
        elif self.selected_record_idx >= 0:
            hint = "Backspace 删除选中"
        else:
            hint = "点击记录选中"
        _draw_text(img, hint, sx + PAD, hint_y, SCALE_SMALL, CLR_TEXT_DIM)

    # ══════════════════════════════════════════════════════
    # 录制叠加层
    # ══════════════════════════════════════════════════════

    def _draw_recording_overlay(self, img):
        # 右上角指示器
        rx, ry = self.cam_w - 130, TOP_BAR_H + PAD + 20
        if int(time.time() * 3) % 2:
            cv2.circle(img, (rx, ry), 6, CLR_DANGER, -1)
        _draw_text(img, f"录制中 {self.record_elapsed:.1f}s", rx + 16, ry + 6,
                   SCALE_BODY, CLR_DANGER)

        # 中央 pill badge：标签名
        label = self.labels[self.selected_label_idx]
        tw, th = _text_size(label, SCALE_TITLE, 2)
        badge_w = tw + PAD * 3
        badge_h = th + PAD * 2
        badge_x = (self.cam_w - badge_w) // 2
        badge_y = (self.canvas_h - BOTTOM_BAR_H - badge_h) // 2 + 20

        _rounded_rect(img, badge_x, badge_y, badge_w, badge_h, CORNER_R,
                      (0, 0, 0), alpha=0.7)
        _rounded_outline(img, badge_x, badge_y, badge_w, badge_h, CORNER_R,
                         CLR_DANGER, thickness=2)
        _draw_text(img, label, self.cam_w // 2, badge_y + badge_h // 2 + 6,
                   SCALE_TITLE, CLR_DANGER, 2, align="center")

    # ══════════════════════════════════════════════════════
    # 鼠标
    # ══════════════════════════════════════════════════════

    def _on_mouse(self, event, x, y, flags, param):
        # 滚轮翻页（侧面板记录列表）
        if event == cv2.EVENT_MOUSEWHEEL:
            max_scroll = max(0, len(self.records) - self._max_side_cards)
            if flags > 0:
                self._scroll_offset = max(0, self._scroll_offset - 1)
            else:
                self._scroll_offset = min(max_scroll, self._scroll_offset + 1)
            return

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # 阶段按钮
        for bx, by, bw, bh, ph in self._phase_btns:
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self._switch_phase(ph)
                return

        # 标签按钮
        for bx, by, bw, bh, idx in self._label_btns:
            if bx <= x <= bx + bw and by <= y <= by + bh and not self.recording:
                self.selected_label_idx = idx
                return

        # 录制按钮
        if self._rec_btn:
            bx, by, bw, bh = self._rec_btn
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self._toggle_recording()
                return

        # 导出按钮
        if self._export_btn:
            bx, by, bw, bh = self._export_btn
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self._export_csv()
                return

        # 记录卡片（选中/取消）
        for bx, by, bw, bh, idx in self._card_rects:
            if bx <= x <= bx + bw and by <= y <= by + bh:
                if self.selected_record_idx != idx:
                    self._pending_delete = False  # 切换卡片时重置确认
                self.selected_record_idx = -1 if self.selected_record_idx == idx else idx
                return

    # ══════════════════════════════════════════════════════
    # 操作
    # ══════════════════════════════════════════════════════

    def _switch_phase(self, ph):
        if ph == self.phase:
            return
        if self.recording:
            print("[WARN] Switching phase while recording - recording stopped")
            self._stop_recording()
        self.phase = ph
        self.labels = PHASES[ph]
        self.selected_label_idx = 0
        self._pending_delete = False

    def _toggle_recording(self):
        if self.recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        self.recording = True
        self.record_start = time.time()
        self.record_elapsed = 0.0
        self.record_frames.clear()
        print(f"[REC] 开始录制 [{self.labels[self.selected_label_idx]}] …")

    def _stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        elapsed = time.time() - self.record_start
        n = len(self.record_frames)

        if n < 5 or elapsed < 0.5:
            print("[WARN] 录制太短，已丢弃")
            self.record_frames.clear()
            return

        label = self.labels[self.selected_label_idx]
        print(f"[STOP] 停止录制 [{label}]  {elapsed:.1f}s  {n}帧")
        try:
            self._compute_and_log(label, elapsed, n)
        except Exception as e:
            print(f"[ERROR] 数据处理异常: {e}")
            import traceback
            traceback.print_exc()
        self.record_frames.clear()

    def _compute_and_log(self, label, duration, n_frames):
        frames = self.record_frames

        def _stats(key):
            vals = [f[key] for f in frames if key in f]
            if not vals:
                return 0, 0, 0
            return min(vals), max(vals), sum(vals) / len(vals)

        if self.phase == "EAR":
            metrics = {}
            for mk in ["ear_l", "ear_r", "ear_avg"]:
                mn, mx, av = _stats(mk)
                name = {"ear_l": "EAR_L", "ear_r": "EAR_R", "ear_avg": "EAR_avg"}[mk]
                metrics[name] = (mn, mx, av)
                print(f"  {name:8s}  min={mn:.4f}  max={mx:.4f}  avg={av:.4f}")
        elif self.phase == "MAR":
            mn, mx, av = _stats("mar")
            metrics = {"MAR": (mn, mx, av)}
            print(f"  {'MAR':8s}  min={mn:.4f}  max={mx:.4f}  avg={av:.4f}")
        else:  # HEAD
            pitch_stats = _stats("pitch")
            yaw_stats = _stats("yaw")
            roll_stats = _stats("roll")
            metrics = {
                "Pitch": pitch_stats,
                "Yaw": yaw_stats,
                "Roll": roll_stats,
            }
            print(f"  Pitch  min={pitch_stats[0]:.1f}  max={pitch_stats[1]:.1f}  avg={pitch_stats[2]:.1f}")
            print(f"  Yaw    min={yaw_stats[0]:.1f}  max={yaw_stats[1]:.1f}  avg={yaw_stats[2]:.1f}")
            print(f"  Roll   min={roll_stats[0]:.1f}  max={roll_stats[1]:.1f}  avg={roll_stats[2]:.1f}")

        first_key = list(metrics.keys())[0]
        summary = f"{metrics[first_key][2]:.3f}"

        self.records.append({
            "phase": self.phase,
            "label": label,
            "duration": duration,
            "frames": n_frames,
            "summary": summary,
            "_metrics": metrics,
        })

    def _export_csv(self):
        if not self.records:
            print("[WARN] 暂无记录可导出")
            return
        for rec in self.records:
            metrics = rec.get("_metrics", {})
            if metrics:
                self.logger.write_record(
                    rec["phase"], rec["label"],
                    rec["duration"], rec["frames"], metrics)
        filepath = self.logger._filepath
        print(f"[EXPORT] 已导出 {len(self.records)} 条 → {filepath}")

    def _delete_selected_record(self):
        if self.selected_record_idx < 0 or \
           self.selected_record_idx >= len(self.records):
            return
        if not self._pending_delete:
            self._pending_delete = True
            rec = self.records[self.selected_record_idx]
            print(f"  Press Backspace again to DELETE [{rec['phase']}] {rec['label']}")
            return
        rec = self.records.pop(self.selected_record_idx)
        print(f"[DEL] Deleted [{rec['phase']}] {rec['label']} ({rec['duration']:.1f}s)")
        self.selected_record_idx = -1
        self._pending_delete = False

    def _screenshot(self, canvas):
        os.makedirs("data/screenshots", exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"data/screenshots/calib_{ts}.png"
        cv2.imwrite(fname, canvas)
        print(f"[SHOT] {fname}")


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    CalibTool().run()

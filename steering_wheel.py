import cv2
import mediapipe as mp
USE_TASKS = False
mp_tasks = None
try:
    mp_hands = mp.solutions.hands
except Exception:
    # mediapipe 'solutions' API not available; try the 'tasks' API
    try:
        from mediapipe.tasks.python import vision as mp_tasks_vision
        from mediapipe.tasks.python.vision.core import image as mp_tasks_image
        from mediapipe.tasks.python.core import base_options as mp_tasks_base_options
        mp_tasks = mp_tasks_vision
        USE_TASKS = True
    except Exception:
        mp_hands = None
import numpy as np
import math
import time
import platform
from pynput.keyboard import Key, Controller
import os
import argparse
import struct

CAMERA_INDEX       = 0
DEAD_ZONE_DEG      = 12
RELEASE_ZONE_DEG   = 6
SOFT_ZONE_DEG      = 25
FLIP_CAMERA        = True
SHOW_ANGLE         = True
MIN_DETECTION_CONF = 0.5
MIN_TRACKING_CONF  = 0.5
GRACE_FRAMES       = 8
CALIBRATION_SECONDS = 3.0
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


def parse_args():
    p = argparse.ArgumentParser(description="Virtual Steering Wheel")
    p.add_argument("--model", default=MODEL_PATH, help="Path to hand_landmarker.task model")
    p.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Camera index (integer)")
    p.add_argument("--no-flip", dest="flip", action="store_false", help="Disable horizontal flip of camera")
    p.add_argument("--test-image", dest="test_image", default=None, help="Run detection on a single image file and exit")
    p.add_argument("--debug", dest="debug", action="store_true", help="Print debug detection info")
    p.set_defaults(flip=FLIP_CAMERA)
    return p.parse_args()

CLR_WHEEL        = (220, 110, 40)
CLR_STEERING     = (60, 150, 245)
CLR_LEFT         = (243, 140, 40)
CLR_RIGHT        = (75, 210, 95)
CLR_NEUTRAL      = (240, 240, 240)
CLR_TEXT         = (240, 245, 255)
CLR_ACCENT       = (100, 170, 255)
CLR_HUD_BG       = (16, 20, 28)
CLR_PANEL_BG     = (24, 30, 40)
CLR_PANEL_BORDER = (72, 88, 110)
CLR_HAND_L       = (243, 140, 40)
CLR_HAND_R       = (75, 210, 95)

keyboard   = Controller()
if not USE_TASKS:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
else:
    mp_drawing = mp_tasks.drawing_utils


class SteeringController:
    SENSITIVITY_MAP = {
        1: ("Low", 20.0, 35.0),
        2: ("Medium", 12.0, 25.0),
        3: ("High", 6.0, 18.0),
    }

    ANGLE_SMOOTHING_ALPHA = 0.18

    def __init__(self):
        self.keys_held     = {Key.left: False, Key.right: False, Key.up: False}
        self.angle_history = []
        self.HISTORY_LEN   = 3
        self.display_angle_deg = 0.0
        self.sensitivity_level = 2
        self.sensitivity_label, self.dead_zone_deg, self.soft_zone_deg = self.SENSITIVITY_MAP[self.sensitivity_level]
        self.release_zone_deg = RELEASE_ZONE_DEG
        self.calibration_offset_deg = 0.0
        self.calibrated = False

    def set_sensitivity(self, level: int):
        if level in self.SENSITIVITY_MAP:
            self.sensitivity_level = level
            self.sensitivity_label, self.dead_zone_deg, self.soft_zone_deg = self.SENSITIVITY_MAP[level]
            print(f"[INFO] Sensitivity set to {self.sensitivity_label} ({self.sensitivity_level})")

    def _press(self, key):
        if not self.keys_held[key]:
            keyboard.press(key)
            self.keys_held[key] = True

    def _release(self, key):
        if self.keys_held[key]:
            keyboard.release(key)
            self.keys_held[key] = False

    def release_all(self):
        for key in list(self.keys_held.keys()):
            try:
                keyboard.release(key)
            except Exception:
                pass
            self.keys_held[key] = False
        self.angle_history.clear()

    def smooth_angle(self, raw_angle: float) -> float:
        self.angle_history.append(raw_angle)
        if len(self.angle_history) > self.HISTORY_LEN:
            self.angle_history.pop(0)
        return float(np.mean(self.angle_history))

    def smooth_display_angle(self, target_angle: float) -> float:
        self.display_angle_deg = (
            self.display_angle_deg * (1.0 - self.ANGLE_SMOOTHING_ALPHA)
            + target_angle * self.ANGLE_SMOOTHING_ALPHA
        )
        return self.display_angle_deg

    def update(self, left_wrist, right_wrist):
        dx = right_wrist[0] - left_wrist[0]
        dy = right_wrist[1] - left_wrist[1]

        raw_angle_rad = math.atan2(dy, dx)
        raw_angle_deg = math.degrees(raw_angle_rad)
        angle = self.smooth_angle(raw_angle_deg)

        direction = "STRAIGHT"
        centered_angle = angle - (self.calibration_offset_deg if self.calibrated else 0.0)
        if centered_angle < -self.dead_zone_deg:
            direction = "LEFT"
        elif centered_angle > self.dead_zone_deg:
            direction = "RIGHT"
        elif self.keys_held[Key.left] and centered_angle > -self.release_zone_deg:
            direction = "STRAIGHT"
        elif self.keys_held[Key.right] and centered_angle < self.release_zone_deg:
            direction = "STRAIGHT"

        strength = 0.0
        if direction == "LEFT":
            strength = min(1.0, (abs(centered_angle) - self.dead_zone_deg) / (self.soft_zone_deg - self.dead_zone_deg))
            self._press(Key.left)
            self._release(Key.right)
        elif direction == "RIGHT":
            strength = min(1.0, (abs(centered_angle) - self.dead_zone_deg) / (self.soft_zone_deg - self.dead_zone_deg))
            self._press(Key.right)
            self._release(Key.left)
        else:
            self._release(Key.left)
            self._release(Key.right)

        return angle, direction, strength


def draw_steering_wheel(frame, center, angle_deg, direction, strength):
    h, w = frame.shape[:2]
    radius = int(min(w, h) * 0.16)
    cx, cy = center

    wheel_base = (40, 44, 52)
    rim_color = (96, 104, 120)
    spoke_color = (180, 190, 205)
    hub_color = (32, 36, 44)
    inner_ring = CLR_PANEL_BG
    accent = CLR_ACCENT if direction == "STRAIGHT" else (CLR_LEFT if direction == "LEFT" else CLR_RIGHT)

    cv2.circle(frame, (cx, cy), radius + 18, wheel_base, -1)
    cv2.circle(frame, (cx, cy), radius + 14, CLR_PANEL_BORDER, 2)
    cv2.ellipse(frame, (cx, cy), (radius + 12, radius + 4), 0, 0, 360, (96, 104, 120), -1)
    cv2.circle(frame, (cx, cy), radius + 4, inner_ring, -1)
    cv2.circle(frame, (cx, cy), radius, rim_color, 24)

    spoke_angles = [-90, 0, 90]
    for angle in spoke_angles:
        spoke_angle = math.radians(-angle_deg + angle)
        start_x = int(cx + (radius - 22) * math.cos(spoke_angle))
        start_y = int(cy + (radius - 22) * math.sin(spoke_angle))
        end_x = int(cx + (radius * 0.42) * math.cos(spoke_angle))
        end_y = int(cy + (radius * 0.42) * math.sin(spoke_angle))
        cv2.line(frame, (start_x, start_y), (end_x, end_y), spoke_color, 18)
        cv2.line(frame, (cx, cy), (end_x, end_y), spoke_color, 4)

    cv2.circle(frame, (cx, cy), int(radius * 0.30), hub_color, -1)
    cv2.circle(frame, (cx, cy), int(radius * 0.20), accent, -1)
    cv2.circle(frame, (cx, cy), int(radius * 0.10), CLR_TEXT, -1)

    indicator_angle = math.radians(-angle_deg + 90)
    px = int(cx + (radius - 28) * math.cos(indicator_angle))
    py = int(cy + (radius - 28) * math.sin(indicator_angle))
    cv2.circle(frame, (px, py), 10, accent, -1)
    cv2.circle(frame, (px, py), 4, CLR_TEXT, -1)

    for offset in [12, 22]:
        cv2.ellipse(frame, (cx, cy), (radius - offset, radius - offset), 0, 0, 360, (140, 150, 170), 1)

    label = "STEER"
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.58, 2)
    cv2.putText(frame, label, (cx - label_size[0] // 2, cy + radius // 2), cv2.FONT_HERSHEY_DUPLEX, 0.58, CLR_NEUTRAL, 2, cv2.LINE_AA)


def draw_hud(frame, angle, direction, strength, both_hands_visible, fps, sensitivity_label="Medium"):
    h, w = frame.shape[:2]

    top_panel_h = 60
    bottom_panel_h = 130
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, top_panel_h), CLR_HUD_BG, -1)
    cv2.rectangle(overlay, (0, h - bottom_panel_h), (w, h), CLR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.74, frame, 0.26, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    direction_text = "STRAIGHT"
    direction_color = CLR_STEERING
    if direction == "LEFT":
        direction_text = "LEFT"
        direction_color = CLR_LEFT
    elif direction == "RIGHT":
        direction_text = "RIGHT"
        direction_color = CLR_RIGHT

    header_y = 36
    cv2.putText(frame, "VIRTUAL STEERING", (22, header_y), font, 0.64, CLR_TEXT, 2, cv2.LINE_AA)
    cv2.putText(frame, "Simulator-style controls", (22, header_y + 20), font, 0.40, CLR_ACCENT, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS {fps:.0f}", (w - 156, header_y + 4), font, 0.44, CLR_NEUTRAL, 1, cv2.LINE_AA)

    status_text = "🟢 Hands Detected" if both_hands_visible else "🔴 Waiting for Hands"
    status_color = CLR_RIGHT if both_hands_visible else CLR_LEFT
    status_size, _ = cv2.getTextSize(status_text, font, 0.50, 1)
    status_w = status_size[0] + 26
    status_h = 30
    status_x = 22
    status_y = top_panel_h - 34
    cv2.rectangle(frame, (status_x, status_y), (status_x + status_w, status_y + status_h), CLR_PANEL_BG, -1)
    cv2.rectangle(frame, (status_x, status_y), (status_x + status_w, status_y + status_h), CLR_PANEL_BORDER, 1)
    cv2.putText(frame, status_text, (status_x + 12, status_y + 20), font, 0.50, status_color, 1, cv2.LINE_AA)

    if fps < 15.0:
        fps_warning = "LOW FPS"
        warn_size, _ = cv2.getTextSize(fps_warning, font, 0.44, 1)
        warn_w = warn_size[0] + 22
        warn_x = (w - warn_w) // 2
        warn_y = top_panel_h - 34
        cv2.rectangle(frame, (warn_x, warn_y), (warn_x + warn_w, warn_y + 26), CLR_PANEL_BG, -1)
        cv2.rectangle(frame, (warn_x, warn_y), (warn_x + warn_w, warn_y + 26), CLR_LEFT, 1)
        cv2.putText(frame, fps_warning, (warn_x + 10, warn_y + 18), font, 0.44, CLR_LEFT, 1, cv2.LINE_AA)

    tracking_text = "🟢 Tracking Active" if both_hands_visible else "🔴 Tracking Lost"
    tracking_color = CLR_RIGHT if both_hands_visible else CLR_LEFT
    track_size, _ = cv2.getTextSize(tracking_text, font, 0.44, 1)
    track_w = track_size[0] + 24
    track_h = 26
    track_x = w - track_w - 22
    track_y = top_panel_h - 34
    cv2.rectangle(frame, (track_x, track_y), (track_x + track_w, track_y + track_h), CLR_PANEL_BG, -1)
    cv2.rectangle(frame, (track_x, track_y), (track_x + track_w, track_y + track_h), CLR_PANEL_BORDER, 1)
    cv2.putText(frame, tracking_text, (track_x + 12, track_y + 18), font, 0.44, tracking_color, 1, cv2.LINE_AA)

    panel_w = 380
    panel_h = 118
    panel_x = (w - panel_w) // 2
    panel_y = h - panel_h - 18
    glass = frame.copy()
    cv2.rectangle(glass, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (40, 48, 64), -1)
    cv2.addWeighted(glass, 0.58, frame, 0.42, 0, frame)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), CLR_PANEL_BORDER, 1)
    cv2.rectangle(frame, (panel_x + 10, panel_y + 10), (panel_x + panel_w - 10, panel_y + panel_h - 10), (50, 70, 95), 1)

    left_x = panel_x + 22
    right_x = panel_x + panel_w - 22
    row_y = panel_y + 32
    cv2.putText(frame, "FPS", (left_x, row_y), font, 0.42, CLR_NEUTRAL, 1, cv2.LINE_AA)
    fps_text = f"{fps:.0f}"
    fps_size, _ = cv2.getTextSize(fps_text, font, 0.52, 1)
    cv2.putText(frame, fps_text, (right_x - fps_size[0], row_y), font, 0.52, CLR_TEXT, 1, cv2.LINE_AA)

    row_y += 24
    cv2.putText(frame, "Angle", (left_x, row_y), font, 0.42, CLR_NEUTRAL, 1, cv2.LINE_AA)
    angle_text = f"{angle:+.0f}°"
    angle_size, _ = cv2.getTextSize(angle_text, font, 0.52, 1)
    cv2.putText(frame, angle_text, (right_x - angle_size[0], row_y), font, 0.52, CLR_TEXT, 1, cv2.LINE_AA)

    row_y += 24
    cv2.putText(frame, "Direction", (left_x, row_y), font, 0.42, CLR_NEUTRAL, 1, cv2.LINE_AA)
    dir_size, _ = cv2.getTextSize(direction_text, font, 0.52, 1)
    cv2.putText(frame, direction_text, (right_x - dir_size[0], row_y), font, 0.52, direction_color, 1, cv2.LINE_AA)

    row_y += 24
    cv2.putText(frame, "Sensitivity", (left_x, row_y), font, 0.42, CLR_NEUTRAL, 1, cv2.LINE_AA)
    sens_text = sensitivity_label.capitalize()
    sens_size, _ = cv2.getTextSize(sens_text, font, 0.52, 1)
    cv2.putText(frame, sens_text, (right_x - sens_size[0], row_y), font, 0.52, CLR_TEXT, 1, cv2.LINE_AA)

    gauge_w = int(w * 0.40)
    gauge_h = 18
    gauge_x = (w - gauge_w) // 2
    gauge_y = panel_y - 28
    cv2.rectangle(frame, (gauge_x, gauge_y), (gauge_x + gauge_w, gauge_y + gauge_h), CLR_PANEL_BG, -1)
    cv2.rectangle(frame, (gauge_x, gauge_y), (gauge_x + gauge_w, gauge_y + gauge_h), CLR_PANEL_BORDER, 1)
    center_x = gauge_x + gauge_w // 2
    cv2.line(frame, (center_x, gauge_y), (center_x, gauge_y + gauge_h), CLR_PANEL_BORDER, 1)

    normalized = max(min(angle / 40.0, 1.0), -1.0)
    fill_width = int((gauge_w // 2 - 10) * abs(normalized))
    if normalized > 0:
        cv2.rectangle(frame, (center_x + 4, gauge_y + 4), (center_x + 4 + fill_width, gauge_y + gauge_h - 4), CLR_RIGHT, -1)
    elif normalized < 0:
        cv2.rectangle(frame, (center_x - 4 - fill_width, gauge_y + 4), (center_x - 4, gauge_y + gauge_h - 4), CLR_LEFT, -1)

    cv2.putText(frame, "STEERING", (gauge_x + 10, gauge_y - 8), font, 0.42, CLR_NEUTRAL, 1, cv2.LINE_AA)
    cv2.putText(frame, angle_text, (center_x - 28, gauge_y + gauge_h + 28), font, 0.60, CLR_TEXT, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{int(strength * 100)}%", (center_x + 32, gauge_y + gauge_h + 28), font, 0.50, CLR_NEUTRAL, 1, cv2.LINE_AA)

    cv2.putText(frame, "H = Help   Q = Quit   1/2/3 = Sensitivity", (22, h - 18), font, 0.72, CLR_NEUTRAL, 1, cv2.LINE_AA)


def draw_help_overlay(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    help_w = int(w * 0.48)
    help_h = int(h * 0.36)
    help_x = (w - help_w) // 2
    help_y = (h - help_h) // 2
    cv2.rectangle(overlay, (help_x, help_y), (help_x + help_w, help_y + help_h), CLR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)

    cv2.rectangle(frame, (help_x, help_y), (help_x + help_w, help_y + 48), CLR_ACCENT, -1)
    cv2.rectangle(frame, (help_x, help_y), (help_x + help_w, help_y + help_h), CLR_PANEL_BORDER, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    title_y = help_y + 34
    cv2.putText(frame, "CONTROLS", (help_x + 24, title_y), font, 0.95, CLR_TEXT, 2, cv2.LINE_AA)
    cv2.putText(frame, "Q  Quit", (help_x + 24, title_y + 50), font, 0.70, CLR_NEUTRAL, 1, cv2.LINE_AA)
    cv2.putText(frame, "H  Toggle Help", (help_x + 24, title_y + 86), font, 0.70, CLR_NEUTRAL, 1, cv2.LINE_AA)
    cv2.putText(frame, "1/2/3  Sensitivity", (help_x + 24, title_y + 122), font, 0.70, CLR_NEUTRAL, 1, cv2.LINE_AA)
    cv2.putText(frame, "Hands visible = Input active", (help_x + 24, title_y + 158), font, 0.62, CLR_NEUTRAL, 1, cv2.LINE_AA)
    cv2.putText(frame, "Keep your wrists centered and stable.", (help_x + 24, title_y + 188), font, 0.62, CLR_NEUTRAL, 1, cv2.LINE_AA)


def draw_hand_connection(frame, lw, rw):
    lx, ly = lw
    rx, ry = rw
    cv2.line(frame, (lx, ly), (rx, ry), (30, 100, 200), 8)
    cv2.line(frame, (lx, ly), (rx, ry), CLR_ACCENT, 2)
    cv2.circle(frame, (lx, ly), 10, CLR_HAND_L, -1)
    cv2.circle(frame, (rx, ry), 10, CLR_HAND_R, -1)
    cv2.circle(frame, (lx, ly), 13, CLR_HAND_L, 2)
    cv2.circle(frame, (rx, ry), 13, CLR_HAND_R, 2)
    mx = (lx + rx) // 2
    my = (ly + ry) // 2
    cv2.circle(frame, (mx, my), 7, CLR_WHEEL, -1)


def show_error_window(lines, window_name="ERROR"):
    width = 420
    height = 220
    bg_color = (15, 15, 40)
    text_color = (220, 220, 220)
    accent_color = (0, 150, 255)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = bg_color

    title_font = cv2.FONT_HERSHEY_SIMPLEX
    body_font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, window_name, (20, 40), title_font, 1.0, accent_color, 2, cv2.LINE_AA)

    y = 80
    for line in lines:
        cv2.putText(canvas, line, (20, y), body_font, 0.7, text_color, 1, cv2.LINE_AA)
        y += 35

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    cv2.imshow(window_name, canvas)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def set_opencv_window_icon(window_name, icon_path):
    if platform.system() != "Windows":
        return
    try:
        import ctypes
        hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
        if hwnd == 0:
            return
        LR_LOADFROMFILE = 0x00000010
        IMAGE_ICON = 1
        ICON_SMALL = 0
        ICON_BIG = 1
        hicon = ctypes.windll.user32.LoadImageW(0, icon_path, IMAGE_ICON, 0, 0, LR_LOADFROMFILE)
        if hicon:
            ctypes.windll.user32.SendMessageW(hwnd, 0x0080, ICON_SMALL, hicon)
            ctypes.windll.user32.SendMessageW(hwnd, 0x0080, ICON_BIG, hicon)
    except Exception:
        pass


def ensure_app_icon():
    icon_path = os.path.join(os.path.dirname(__file__), "steering_icon.ico")
    if os.path.exists(icon_path):
        return icon_path

    size = 64
    img = np.zeros((size, size, 4), dtype=np.uint8)
    center = size // 2
    cv2.circle(img, (center, center), 28, (18, 24, 34, 255), -1)
    cv2.circle(img, (center, center), 22, (40, 100, 175, 255), -1)
    cv2.circle(img, (center, center), 16, (72, 170, 235, 255), -1)
    cv2.circle(img, (center, center), 9, (245, 245, 245, 255), -1)

    for angle in [0, 72, 144, 216, 288]:
        rad = math.radians(angle)
        x1 = int(center + 10 * math.cos(rad))
        y1 = int(center + 10 * math.sin(rad))
        x2 = int(center + 18 * math.cos(rad))
        y2 = int(center + 18 * math.sin(rad))
        cv2.line(img, (x1, y1), (x2, y2), (245, 245, 245, 255), 4)
        cv2.circle(img, (x2, y2), 3, (255, 255, 255, 255), -1)

    cv2.ellipse(img, (center, center), (18, 18), 45, 0, 180, (255, 255, 255, 120), 3)

    bmp_size = size * size * 4
    mask_row_bytes = ((size + 31) // 32) * 4
    mask_size = mask_row_bytes * size
    image_size = 40 + bmp_size + mask_size

    with open(icon_path, "wb") as f:
        f.write(struct.pack("<HHH", 0, 1, 1))
        f.write(struct.pack("<BBBBHHII", size, size, 0, 0, 1, 32, image_size, 22))
        f.write(struct.pack("<IIIHHIIIIII", 40, size, size * 2, 1, 32, 0, bmp_size + mask_size, 0, 0, 0, 0))
        for row in range(size - 1, -1, -1):
            f.write(img[row].tobytes())
        f.write(b"\x00" * mask_size)

    return icon_path


def show_ready_transition(status="Ready ✓", subtitle="Tracking active"):
    width = 560
    height = 280
    base = np.zeros((height, width, 3), dtype=np.uint8)
    base[:] = CLR_HUD_BG

    panel_x = 24
    panel_y = 22
    panel_w = width - 48
    panel_h = height - 44
    frame = base.copy()
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), CLR_PANEL_BG, -1)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), CLR_PANEL_BORDER, 2)
    title_font = cv2.FONT_HERSHEY_SIMPLEX
    message_font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "✅  Virtual Steering Wheel", (panel_x + 24, panel_y + 44), title_font, 0.75, CLR_TEXT, 2, cv2.LINE_AA)
    cv2.putText(frame, status, (panel_x + 24, panel_y + 88), message_font, 0.90, CLR_RIGHT, 2, cv2.LINE_AA)
    cv2.putText(frame, subtitle, (panel_x + 24, panel_y + 128), message_font, 0.70, CLR_NEUTRAL, 1, cv2.LINE_AA)

    for alpha in np.linspace(0.0, 1.0, 12):
        blended = cv2.addWeighted(frame, alpha, base, 1.0 - alpha, 0)
        cv2.imshow("Calibration", blended)
        cv2.waitKey(35)
    cv2.waitKey(1)


def show_startup_splash(window_name="Virtual Steering Wheel"):
    width = 700
    height = 340
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = CLR_HUD_BG

    panel_x = 30
    panel_y = 28
    panel_w = width - 60
    panel_h = height - 56
    cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), CLR_PANEL_BG, -1)
    cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), CLR_PANEL_BORDER, 2)

    icon_center = (width // 2, panel_y + 112)
    icon_radius = 54
    cv2.circle(canvas, icon_center, icon_radius + 6, CLR_PANEL_BORDER, -1)
    cv2.circle(canvas, icon_center, icon_radius, CLR_WHEEL, -1)
    cv2.circle(canvas, icon_center, icon_radius - 14, CLR_HUD_BG, -1)
    cv2.circle(canvas, icon_center, 18, CLR_ACCENT, -1)
    cv2.circle(canvas, icon_center, 10, CLR_TEXT, -1)

    for angle in [0, 72, 144, 216, 288]:
        rad = math.radians(angle)
        sx = int(icon_center[0] + (icon_radius - 18) * math.cos(rad))
        sy = int(icon_center[1] + (icon_radius - 18) * math.sin(rad))
        ex = int(icon_center[0] + (icon_radius - 8) * math.cos(rad))
        ey = int(icon_center[1] + (icon_radius - 8) * math.sin(rad))
        cv2.line(canvas, (sx, sy), (ex, ey), CLR_TEXT, 4)

    cv2.ellipse(canvas, icon_center, (icon_radius - 12, icon_radius - 12), 45, 0, 180, (255, 255, 255), 2)

    title_font = cv2.FONT_HERSHEY_SIMPLEX
    label_font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "VIRTUAL STEERING WHEEL", (panel_x + 30, panel_y + 220), title_font, 1.0, CLR_TEXT, 2, cv2.LINE_AA)
    cv2.putText(canvas, "Starting calibration and controls...", (panel_x + 30, panel_y + 260), label_font, 0.62, CLR_NEUTRAL, 1, cv2.LINE_AA)

    bar_x = panel_x + 60
    bar_y = panel_y + 284
    bar_w = panel_w - 120
    bar_h = 18
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 50, 68), -1)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), CLR_PANEL_BORDER, 1)

    shadow = np.zeros_like(canvas)
    for alpha in np.linspace(0.0, 1.0, 12):
        blended = cv2.addWeighted(canvas, alpha, shadow, 1.0 - alpha, 0)
        cv2.imshow(window_name, blended)
        cv2.waitKey(30)

    for progress in range(0, 101, 5):
        frame = canvas.copy()
        filled = int((bar_w - 4) * (progress / 100.0))
        if filled > 0:
            cv2.rectangle(frame, (bar_x + 2, bar_y + 2), (bar_x + 2 + filled, bar_y + bar_h - 2), CLR_ACCENT, -1)
        progress_text = f"Initializing... {progress}%"
        text_size, _ = cv2.getTextSize(progress_text, label_font, 0.55, 1)
        text_x = bar_x + (bar_w - text_size[0]) // 2
        cv2.putText(frame, progress_text, (text_x, bar_y - 10), label_font, 0.55, CLR_NEUTRAL, 1, cv2.LINE_AA)
        cv2.imshow(window_name, frame)
        cv2.waitKey(40)

    cv2.waitKey(300)


def show_calibration_screen(countdown, message, progress=0, icon="🚗"):
    width = 560
    height = 280
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = CLR_HUD_BG

    panel_x = 24
    panel_y = 22
    panel_w = width - 48
    panel_h = height - 44
    cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), CLR_PANEL_BG, -1)
    cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), CLR_PANEL_BORDER, 2)

    title_font = cv2.FONT_HERSHEY_SIMPLEX
    message_font = cv2.FONT_HERSHEY_SIMPLEX
    count_font = cv2.FONT_HERSHEY_SIMPLEX
    small_font = cv2.FONT_HERSHEY_SIMPLEX

    title = f"{icon}  Virtual Steering Wheel"
    cv2.putText(canvas, title, (panel_x + 24, panel_y + 44), title_font, 0.75, CLR_TEXT, 2, cv2.LINE_AA)
    cv2.putText(canvas, "Calibrating...", (panel_x + 24, panel_y + 84), message_font, 0.78, CLR_ACCENT, 2, cv2.LINE_AA)

    if countdown:
        cv2.putText(canvas, countdown, (width // 2 - 36, panel_y + 150), count_font, 2.4, CLR_TEXT, 5, cv2.LINE_AA)

    bar_x = panel_x + 24
    bar_y = panel_y + 180
    bar_w = panel_w - 48
    bar_h = 22
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 60, 78), -1)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), CLR_PANEL_BORDER, 1)
    filled_w = int(bar_w * max(min(progress / 100.0, 1.0), 0.0))
    if filled_w > 0:
        cv2.rectangle(canvas, (bar_x + 2, bar_y + 2), (bar_x + 2 + filled_w, bar_y + bar_h - 2), CLR_ACCENT, -1)

    percent_text = f"{int(progress)}%"
    pct_size, _ = cv2.getTextSize(percent_text, small_font, 0.6, 1)
    cv2.putText(canvas, percent_text, (bar_x + bar_w - pct_size[0], bar_y + bar_h + 26), small_font, 0.6, CLR_NEUTRAL, 1, cv2.LINE_AA)
    cv2.putText(canvas, message, (bar_x, bar_y + bar_h + 52), small_font, 0.65, CLR_NEUTRAL, 1, cv2.LINE_AA)

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration", width, height)
    cv2.imshow("Calibration", canvas)
    cv2.waitKey(1)


def get_wrist_pair(results, w, h):
    hand_data = {}
    if not USE_TASKS and getattr(results, 'multi_hand_landmarks', None) and getattr(results, 'multi_handedness', None):
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            wrist = hand_landmarks.landmark[0]
            hand_data[label] = (wrist.x, wrist.y)
    elif USE_TASKS and results and getattr(results, 'hand_landmarks', None):
        lands = results.hand_landmarks
        handedness_list = getattr(results, 'handedness', None) or getattr(results, 'handednesses', None) or getattr(results, 'handedness_list', None)
        for idx, hand in enumerate(lands):
            landmarks = getattr(hand, 'landmarks', None) or hand
            l0x, l0y = getattr(landmarks[0], 'x', None), getattr(landmarks[0], 'y', None)
            label = None
            if handedness_list and len(handedness_list) > idx:
                try:
                    cand = handedness_list[idx]
                    label = getattr(cand, 'classification', None) and cand.classification[0].label
                except Exception:
                    label = None
            if not label:
                label = 'Left' if l0x < 0.5 else 'Right'
            if l0x is not None and l0y is not None:
                hand_data[label] = (l0x, l0y)
    if 'Left' in hand_data and 'Right' in hand_data:
        return hand_data['Left'], hand_data['Right']
    return None


def calibrate_center(cap, hands, hand_detector, tasks_use_video_mode, controller):
    angles = []
    for step in [3, 2, 1]:
        progress = int(((4 - step) / 3.0) * 100)
        show_calibration_screen(f"{step}...", "Hold both hands straight", progress, icon="🚗")

        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(1.0)
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        if not USE_TASKS:
            results = hands.process(rgb)
        else:
            ImageClass = getattr(mp_tasks_image, 'Image', None) or getattr(mp_tasks, 'Image', None) or getattr(mp, 'Image', None)
            ImageFormat = getattr(mp_tasks_image, 'ImageFormat', None) or getattr(mp_tasks, 'ImageFormat', None) or getattr(mp, 'ImageFormat', None)
            if ImageClass is None or ImageFormat is None:
                continue
            mp_image = ImageClass(image_format=ImageFormat.SRGB, data=rgb)
            if tasks_use_video_mode and hasattr(hand_detector, 'detect_for_video'):
                try:
                    results = hand_detector.detect_for_video(mp_image, int(time.time() * 1000))
                except ValueError:
                    results = hand_detector.detect(mp_image)
            else:
                results = hand_detector.detect(mp_image)
        rgb.flags.writeable = True

        pts = get_wrist_pair(results, frame.shape[1], frame.shape[0])
        if pts is not None:
            left, right = pts
            dx = right[0] - left[0]
            dy = right[1] - left[1]
            angles.append(math.degrees(math.atan2(dy, dx)))

        time.sleep(1.0)

    offset = float(np.mean(angles)) if angles else 0.0
    controller.calibration_offset_deg = offset
    controller.calibrated = True
    show_calibration_screen("", "Calibration complete ✓", 100, icon="✅")
    time.sleep(0.6)
    show_ready_transition(status="Calibration Complete ✓", subtitle="Ready to steer")
    time.sleep(0.6)
    cv2.destroyWindow("Calibration")

def main():
    args = parse_args()
    model_path = args.model
    camera_index = args.camera
    flip_camera = args.flip
    debug_mode = getattr(args, 'debug', False)

    print("Loading Camera...")
    if platform.system() == "Darwin":
        backend = cv2.CAP_AVFOUNDATION
    elif platform.system() == "Windows" and hasattr(cv2, 'CAP_DSHOW'):
        backend = cv2.CAP_DSHOW
    else:
        backend = cv2.CAP_ANY

    print("Loading MediaPipe...")
    print(f"[INFO] Using camera backend: {backend}")
    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened() and platform.system() == "Windows" and hasattr(cv2, 'CAP_DSHOW'):
        print("[INFO] Primary backend failed, retrying with CAP_DSHOW")
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        error_lines = [
            "Camera not detected",
            "",
            "Connect webcam and",
            "restart application",
        ]
        show_error_window(error_lines, window_name="ERROR")
        print("[ERROR] Cannot open camera.")
        print("  -> On macOS: System Settings > Privacy & Security > Camera")
        print("  -> On Windows: try a different camera index, e.g. --camera 1")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    controller = SteeringController()

    print("Starting Virtual Steering Wheel...")
    print(f"[INFO] MediaPipe mode: {'tasks' if USE_TASKS else 'solutions'}")
    cv2.namedWindow("Virtual Steering Wheel", cv2.WINDOW_NORMAL)
    icon_path = ensure_app_icon()
    set_opencv_window_icon("Virtual Steering Wheel", icon_path)
    show_startup_splash("Virtual Steering Wheel")
    if not USE_TASKS:
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=MIN_DETECTION_CONF,
            min_tracking_confidence=MIN_TRACKING_CONF,
        )
        hand_detector = None
        tasks_use_video_mode = False
    else:
        # Using mediapipe.tasks. Require a hand_landmarker model file at provided model_path
        if not os.path.exists(model_path):
            print(f"[ERROR] mediapipe 'tasks' API detected but model not found at: {model_path}")
            print("  -> Download the 'hand_landmarker.task' model and place it in the project folder, or pass --model PATH.")
            return
        BaseOptions = getattr(mp_tasks_base_options, 'BaseOptions', None)
        tasks_use_video_mode = False
        image_detector = None
        if BaseOptions is not None:
            base_options = BaseOptions(model_asset_path=model_path)
            options = mp_tasks.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_tasks.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=MIN_DETECTION_CONF,
                min_tracking_confidence=MIN_TRACKING_CONF,
            )
            hand_detector = mp_tasks.HandLandmarker.create_from_options(options)
            tasks_use_video_mode = True
            if args.test_image:
                image_options = mp_tasks.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=mp_tasks.RunningMode.IMAGE,
                    num_hands=2,
                    min_hand_detection_confidence=MIN_DETECTION_CONF,
                    min_tracking_confidence=MIN_TRACKING_CONF,
                )
                image_detector = mp_tasks.HandLandmarker.create_from_options(image_options)
        elif hasattr(mp_tasks.HandLandmarker, 'create_from_model_path'):
            try:
                hand_detector = mp_tasks.HandLandmarker.create_from_model_path(model_path)
                tasks_use_video_mode = False
                if args.test_image:
                    image_detector = mp_tasks.HandLandmarker.create_from_model_path(model_path)
            except Exception as e:
                print(f"[ERROR] Failed to create HandLandmarker from model path: {e}")
                return
        else:
            print('[ERROR] Cannot find BaseOptions in mediapipe.tasks and no create_from_model_path available; incompatible mediapipe build.')
            return
        hands = None
        image_detector = image_detector

    if not args.test_image:
        calibrate_center(cap, hands, hand_detector, tasks_use_video_mode, controller)

    # If a test image is provided, run detection on it once and exit (no camera required)
    if args.test_image:
        if not os.path.exists(args.test_image):
            print(f"[ERROR] Test image not found: {args.test_image}")
            return
        frame = cv2.imread(args.test_image)
        if frame is None:
            print(f"[ERROR] Failed to read test image: {args.test_image}")
            return
        if flip_camera:
            frame = cv2.flip(frame, 1)

        # Run one iteration of the processing loop on the loaded image
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        if not USE_TASKS:
            results = hands.process(rgb)
        else:
            ImageClass = getattr(mp_tasks_image, 'Image', None) or getattr(mp_tasks, 'Image', None) or getattr(mp, 'Image', None)
            ImageFormat = getattr(mp_tasks_image, 'ImageFormat', None) or getattr(mp_tasks, 'ImageFormat', None) or getattr(mp, 'ImageFormat', None)
            if ImageClass is None or ImageFormat is None:
                print('[ERROR] mediapipe Image/ImageFormat not available in this build.')
                return
            mp_image = ImageClass(image_format=ImageFormat.SRGB, data=rgb)
            if not USE_TASKS:
                results = hands.process(rgb)
            else:
                ImageClass = getattr(mp_tasks_image, 'Image', None) or getattr(mp_tasks, 'Image', None) or getattr(mp, 'Image', None)
                ImageFormat = getattr(mp_tasks_image, 'ImageFormat', None) or getattr(mp_tasks, 'ImageFormat', None) or getattr(mp, 'ImageFormat', None)
                if ImageClass is None or ImageFormat is None:
                    print('[ERROR] mediapipe Image/ImageFormat not available in this build.')
                    return
                mp_image = ImageClass(image_format=ImageFormat.SRGB, data=rgb)
                if args.test_image and image_detector is not None:
                    results = image_detector.detect(mp_image)
                elif tasks_use_video_mode and hasattr(hand_detector, 'detect_for_video'):
                    try:
                        results = hand_detector.detect_for_video(mp_image, 0)
                    except ValueError:
                        results = hand_detector.detect(mp_image)
                else:
                    results = hand_detector.detect(mp_image)
        rgb.flags.writeable = True

        if debug_mode:
            hand_count = len(getattr(results, 'multi_hand_landmarks', []) or getattr(results, 'hand_landmarks', []))
            handedness_count = len(getattr(results, 'multi_handedness', []) or getattr(results, 'handedness', []) or getattr(results, 'handednesses', []))
            print(f"[DEBUG] test-image result hand_count={hand_count}, handedness_count={handedness_count}")

        # reuse existing handling by emulating variables used in loop
        both_visible = False
        hand_data = {}

        if not USE_TASKS and getattr(results, 'multi_hand_landmarks', None) and getattr(results, 'multi_handedness', None):
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                try:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, landmark_style, conn_style)
                except Exception:
                    pass
                wrist = hand_landmarks.landmark[0]
                wx    = int(wrist.x * w)
                wy    = int(wrist.y * h)
                hand_data[label] = (wrist.x, wrist.y, wx, wy)

        elif USE_TASKS and results and getattr(results, 'hand_landmarks', None):
            lands = results.hand_landmarks
            handedness_list = getattr(results, 'handedness', None) or getattr(results, 'handednesses', None) or getattr(results, 'handedness_list', None)
            for idx, hand in enumerate(lands):
                l0 = hand.landmarks[0]
                label = None
                if handedness_list and len(handedness_list) > idx:
                    try:
                        cand = handedness_list[idx]
                        label = getattr(cand, 'classification', None) and cand.classification[0].label
                    except Exception:
                        label = None
                if not label:
                    label = 'Left' if l0.x < 0.5 else 'Right'
                for lmk in hand.landmarks:
                    x_px = int(lmk.x * w)
                    y_px = int(lmk.y * h)
                    cv2.circle(frame, (x_px, y_px), 3, CLR_HAND_L if label == 'Left' else CLR_HAND_R, -1)
                wx = int(l0.x * w)
                wy = int(l0.y * h)
                cv2.putText(frame, label, (wx + 8, wy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_TEXT, 1)
                hand_data[label] = (l0.x, l0.y, wx, wy)

        if 'Left' in hand_data and 'Right' in hand_data:
            both_visible = True
            lx_n, ly_n, lx_px, ly_px = hand_data['Left']
            rx_n, ry_n, rx_px, ry_px = hand_data['Right']
            draw_hand_connection(frame, (lx_px, ly_px), (rx_px, ry_px))
            angle, direction, strength = controller.update((lx_n, ly_n), (rx_n, ry_n))
            angle = controller.smooth_display_angle(angle)
        else:
            if debug_mode:
                print(f"[DEBUG] No both-hands steering pair found. Detected labels: {list(hand_data.keys())}")
            angle, direction, strength = 0.0, 'STRAIGHT', 0.0

        # Save annotated output image
        out_path = os.path.join(os.path.dirname(__file__), 'test_output.png')
        draw_hud(frame, angle, direction, strength, both_visible, 0.0, controller.sensitivity_label)
        cv2.imwrite(out_path, frame)
        print(f"Wrote test output to: {out_path}")
        return

    conn_style     = mp_drawing.DrawingSpec(color=(80, 80, 100), thickness=1)
    landmark_style = mp_drawing.DrawingSpec(color=(200, 200, 255), thickness=1, circle_radius=2)

    prev_time    = time.time()
    fps          = 30.0
    angle        = 0.0
    direction    = "STRAIGHT"
    strength     = 0.0
    lost_frames  = 0
    help_overlay = False
    reduced_resolution = False
    frame_counter = 0

    print("=" * 50)
    print("  Virtual Steering Wheel  |  Press Q to quit")
    print("=" * 50)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            if flip_camera:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            proc_frame = frame
            proc_h, proc_w = h, w
            if reduced_resolution or w > 480:
                proc_w = min(480, w)
                if fps < 12.0:
                    proc_w = min(360, w)
                proc_h = int(h * (proc_w / w))
                if proc_w < w:
                    proc_frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)

            rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            if not USE_TASKS:
                results = hands.process(rgb)
            else:
                # convert to mediapipe.tasks Image and run video-mode detection
                ImageClass = getattr(mp_tasks_image, 'Image', None) or getattr(mp_tasks, 'Image', None) or getattr(mp, 'Image', None)
                ImageFormat = getattr(mp_tasks_image, 'ImageFormat', None) or getattr(mp_tasks, 'ImageFormat', None) or getattr(mp, 'ImageFormat', None)
                if ImageClass is None or ImageFormat is None:
                    print('[ERROR] mediapipe Image/ImageFormat not available in this build.')
                    return
                mp_image = ImageClass(image_format=ImageFormat.SRGB, data=rgb)
                if tasks_use_video_mode and hasattr(hand_detector, 'detect_for_video'):
                    try:
                        results = hand_detector.detect_for_video(mp_image, int(time.time() * 1000))
                    except ValueError:
                        results = hand_detector.detect(mp_image)
                else:
                    results = hand_detector.detect(mp_image)
            rgb.flags.writeable = True

            if debug_mode:
                hand_count = len(getattr(results, 'multi_hand_landmarks', []) or getattr(results, 'hand_landmarks', []))
                handedness_count = len(getattr(results, 'multi_handedness', []) or getattr(results, 'handedness', []) or getattr(results, 'handednesses', []))
                print(f"[DEBUG] live result hand_count={hand_count}, handedness_count={handedness_count}")

            both_visible = False
            hand_data = {}

            if not USE_TASKS and getattr(results, 'multi_hand_landmarks', None) and getattr(results, 'multi_handedness', None):
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label
                    try:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, landmark_style, conn_style)
                    except Exception:
                        pass
                    wrist = hand_landmarks.landmark[0]
                    wx    = int(wrist.x * w)
                    wy    = int(wrist.y * h)
                    hand_data[label] = (wrist.x, wrist.y, wx, wy)

            elif USE_TASKS and results and getattr(results, 'hand_landmarks', None):
                lands = results.hand_landmarks
                # Try to read handedness if provided
                handedness_list = getattr(results, 'handedness', None) or getattr(results, 'handednesses', None) or getattr(results, 'handedness_list', None)
                for idx, hand in enumerate(lands):
                    # hand may be an object with .landmarks or a plain list of landmarks
                    landmarks = getattr(hand, 'landmarks', None) or hand

                    def _xy(lm):
                        # lm could be object with .x/.y or a sequence
                        x = getattr(lm, 'x', None)
                        y = getattr(lm, 'y', None)
                        if x is None or y is None:
                            try:
                                x, y = lm[0], lm[1]
                            except Exception:
                                x, y = 0.0, 0.0
                        return x, y

                    l0x, l0y = _xy(landmarks[0])
                    label = None
                    if handedness_list and len(handedness_list) > idx:
                        try:
                            cand = handedness_list[idx]
                            label = getattr(cand, 'classification', None) and cand.classification[0].label
                        except Exception:
                            label = None
                    if not label:
                        label = 'Left' if l0x < 0.5 else 'Right'
                    for lmk in landmarks:
                        lx, ly = _xy(lmk)
                        x_px = int(lx * w)
                        y_px = int(ly * h)
                        cv2.circle(frame, (x_px, y_px), 3, CLR_HAND_L if label == 'Left' else CLR_HAND_R, -1)
                    wx = int(l0x * w)
                    wy = int(l0y * h)
                    # put label text near wrist
                    cv2.putText(frame, label, (wx + 8, wy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_TEXT, 1)
                    hand_data[label] = (l0x, l0y, wx, wy)

            if 'Left' in hand_data and 'Right' in hand_data:
                both_visible = True
                lost_frames  = 0
                lx_n, ly_n, lx_px, ly_px = hand_data['Left']
                rx_n, ry_n, rx_px, ry_px = hand_data['Right']
                draw_hand_connection(frame, (lx_px, ly_px), (rx_px, ry_px))
                angle, direction, strength = controller.update((lx_n, ly_n), (rx_n, ry_n))
                angle = controller.smooth_display_angle(angle)
            else:
                if debug_mode and hand_data:
                    print(f"[DEBUG] Detected hands but missing left/right pair: {list(hand_data.keys())}")
                lost_frames += 1
                if lost_frames >= GRACE_FRAMES:
                    controller.release_all()
                    angle, direction, strength = 0.0, 'STRAIGHT', 0.0

            now       = time.time()
            fps       = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            frame_counter += 1
            if frame_counter % 60 == 0:
                reduced_resolution = fps < 18.0

            draw_steering_wheel(frame, (int(w * 0.76), int(h * 0.48)), angle, direction, strength)
            draw_hud(frame, angle, direction, strength, both_visible, fps, controller.sensitivity_label)
            if help_overlay:
                draw_help_overlay(frame)
            cv2.imshow("Virtual Steering Wheel", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('1'):
                controller.set_sensitivity(1)
            elif key == ord('2'):
                controller.set_sensitivity(2)
            elif key == ord('3'):
                controller.set_sensitivity(3)
            elif key in (ord('h'), ord('H')):
                help_overlay = not help_overlay

            if key in (ord('q'), ord('Q'), 27):
                break

    finally:
        controller.release_all()
        try:
            if 'hands' in locals() and hands is not None:
                hands.close()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Stopped. All keys released.")


if __name__ == "__main__":
    main()

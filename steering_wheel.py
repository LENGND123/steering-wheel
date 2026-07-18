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

CAMERA_INDEX       = 0
DEAD_ZONE_DEG      = 12
RELEASE_ZONE_DEG   = 6
SOFT_ZONE_DEG      = 25
FLIP_CAMERA        = True
SHOW_ANGLE         = True
MIN_DETECTION_CONF = 0.5
MIN_TRACKING_CONF  = 0.5
GRACE_FRAMES       = 8
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

CLR_WHEEL   = (80, 200, 255)
CLR_LEFT    = (60, 120, 255)
CLR_RIGHT   = (50, 220, 140)
CLR_NEUTRAL = (200, 200, 200)
CLR_TEXT    = (255, 255, 255)
CLR_ACCENT  = (0, 180, 255)
CLR_HAND_L  = (255, 130, 60)
CLR_HAND_R  = (60, 230, 130)

keyboard   = Controller()
if not USE_TASKS:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
else:
    mp_drawing = mp_tasks.drawing_utils


class SteeringController:
    def __init__(self):
        self.keys_held     = {Key.left: False, Key.right: False, Key.up: False}
        self.angle_history = []
        self.HISTORY_LEN   = 1

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

    def update(self, left_wrist, right_wrist):
        dx = right_wrist[0] - left_wrist[0]
        dy = right_wrist[1] - left_wrist[1]

        raw_angle_rad = math.atan2(dy, dx)
        raw_angle_deg = math.degrees(raw_angle_rad)
        angle = self.smooth_angle(raw_angle_deg)

        direction = "STRAIGHT"
        if angle < -DEAD_ZONE_DEG:
            direction = "LEFT"
        elif angle > DEAD_ZONE_DEG:
            direction = "RIGHT"
        elif self.keys_held[Key.left] and angle > -RELEASE_ZONE_DEG:
            direction = "STRAIGHT"
        elif self.keys_held[Key.right] and angle < RELEASE_ZONE_DEG:
            direction = "STRAIGHT"

        strength = 0.0
        if direction == "LEFT":
            strength = min(1.0, (abs(angle) - DEAD_ZONE_DEG) / (SOFT_ZONE_DEG - DEAD_ZONE_DEG))
            self._press(Key.left)
            self._release(Key.right)
        elif direction == "RIGHT":
            strength = min(1.0, (abs(angle) - DEAD_ZONE_DEG) / (SOFT_ZONE_DEG - DEAD_ZONE_DEG))
            self._press(Key.right)
            self._release(Key.left)
        else:
            self._release(Key.left)
            self._release(Key.right)

        return angle, direction, strength


def draw_steering_wheel(frame, center, angle_deg, direction, strength):
    h, w = frame.shape[:2]
    radius = int(min(w, h) * 0.10)
    cx, cy = center

    color = CLR_NEUTRAL
    if direction == "LEFT":
        color = CLR_LEFT
    elif direction == "RIGHT":
        color = CLR_RIGHT

    cv2.circle(frame, (cx + 3, cy + 3), radius, (0, 0, 0), 4)
    cv2.circle(frame, (cx, cy), radius, color, 3)

    for sa in [0, 120, 240]:
        rad = math.radians(sa - angle_deg)
        x1 = int(cx + radius * 0.4 * math.cos(rad))
        y1 = int(cy - radius * 0.4 * math.sin(rad))
        x2 = int(cx + radius * 0.95 * math.cos(rad))
        y2 = int(cy - radius * 0.95 * math.sin(rad))
        cv2.line(frame, (x1, y1), (x2, y2), color, 2)

    cv2.circle(frame, (cx, cy), 6, color, -1)

    if direction != "STRAIGHT":
        start_a = -30 if direction == "RIGHT" else 150
        end_a   =  30 if direction == "RIGHT" else 210
        cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_a, end_a, color, 5)


def draw_hud(frame, angle, direction, strength, both_hands_visible, fps):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 130), (w, h), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    bar_w = int(w * 0.5)
    bar_h = 14
    bar_x = (w - bar_w) // 2
    bar_y = h - 90
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 60), -1)

    mid = bar_x + bar_w // 2
    cv2.rectangle(frame, (mid - 2, bar_y - 4), (mid + 2, bar_y + bar_h + 4), (180, 180, 180), -1)

    fill_len = int((bar_w // 2) * strength)
    if direction == "LEFT" and fill_len > 0:
        cv2.rectangle(frame, (mid - fill_len, bar_y), (mid, bar_y + bar_h), CLR_LEFT, -1)
    elif direction == "RIGHT" and fill_len > 0:
        cv2.rectangle(frame, (mid, bar_y), (mid + fill_len, bar_y + bar_h), CLR_RIGHT, -1)

    font      = cv2.FONT_HERSHEY_SIMPLEX
    dir_color = CLR_LEFT if direction == "LEFT" else (CLR_RIGHT if direction == "RIGHT" else CLR_NEUTRAL)
    cv2.putText(frame, "  <- LEFT",  (bar_x, bar_y - 10),           font, 0.45, CLR_LEFT,  1)
    cv2.putText(frame, "RIGHT ->",   (bar_x + bar_w - 80, bar_y - 10), font, 0.45, CLR_RIGHT, 1)
    cv2.putText(frame, direction,    (mid - 30, bar_y + bar_h + 28), font, 0.8,  dir_color, 2)

    if SHOW_ANGLE:
        cv2.putText(frame, f"{angle:+.1f} deg", (bar_x, h - 20), font, 0.55, CLR_TEXT, 1)

    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 90, 30), font, 0.55, CLR_ACCENT, 1)

    status       = "BOTH HANDS DETECTED" if both_hands_visible else "SHOW BOTH HANDS"
    status_color = (60, 220, 60) if both_hands_visible else (0, 80, 255)
    cv2.putText(frame, status, (10, 30), font, 0.55, status_color, 1)

    draw_steering_wheel(frame, (w - 80, h - 80), angle, direction, strength)


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


def main():
    args = parse_args()
    model_path = args.model
    camera_index = args.camera
    flip_camera = args.flip
    debug_mode = getattr(args, 'debug', False)

    if platform.system() == "Darwin":
        backend = cv2.CAP_AVFOUNDATION
    elif platform.system() == "Windows" and hasattr(cv2, 'CAP_DSHOW'):
        backend = cv2.CAP_DSHOW
    else:
        backend = cv2.CAP_ANY

    print(f"[INFO] Using camera backend: {backend}")
    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened() and platform.system() == "Windows" and hasattr(cv2, 'CAP_DSHOW'):
        print("[INFO] Primary backend failed, retrying with CAP_DSHOW")
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        print("  -> On macOS: System Settings > Privacy & Security > Camera")
        print("  -> On Windows: try a different camera index, e.g. --camera 1")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    controller = SteeringController()

    print(f"[INFO] MediaPipe mode: {'tasks' if USE_TASKS else 'solutions'}")
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
        else:
            if debug_mode:
                print(f"[DEBUG] No both-hands steering pair found. Detected labels: {list(hand_data.keys())}")
            angle, direction, strength = 0.0, 'STRAIGHT', 0.0

        # Save annotated output image
        out_path = os.path.join(os.path.dirname(__file__), 'test_output.png')
        draw_hud(frame, angle, direction, strength, both_visible, 0.0)
        cv2.imwrite(out_path, frame)
        print(f"Wrote test output to: {out_path}")
        return

    conn_style     = mp_drawing.DrawingSpec(color=(80, 80, 100), thickness=1)
    landmark_style = mp_drawing.DrawingSpec(color=(200, 200, 255), thickness=1, circle_radius=2)

    prev_time    = time.time()
    angle        = 0.0
    direction    = "STRAIGHT"
    strength     = 0.0
    lost_frames  = 0

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

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

            draw_hud(frame, angle, direction, strength, both_visible, fps)
            cv2.imshow("Virtual Steering Wheel", frame)

            key = cv2.waitKey(1) & 0xFF
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

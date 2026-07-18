# 🎮 Virtual Steering Wheel — MediaPipe + Python

Control any car game using your hands as a steering wheel — no hardware needed. Just your webcam.

---

## How It Works

Hold both fists toward the camera like you're gripping a steering wheel. Tilt your hands to steer.

```
Both hands level  →  Straight (no key pressed)
Tilt LEFT         →  ← LEFT arrow key
Tilt RIGHT        →  → RIGHT arrow key
Remove hands      →  All keys released instantly
```

---

## Requirements

- Python 3.9+
- Webcam

---

## Install Dependencies

```bash
pip install mediapipe opencv-python pynput numpy
```

---

## Run

```bash
python3 steering_wheel.py
```

Press **Q** in the camera window to quit.

### Debug mode

```bash
python3 steering_wheel.py --debug
```

This prints detection state and helps confirm whether the script is seeing hands.

---

## macOS Setup ⚠️

This project was built and tested on **macOS (Apple M2)**. On macOS you must grant camera permission to Terminal before running:

1. Go to **System Settings → Privacy & Security → Camera**
2. Enable access for **Terminal** (or your Python launcher)
3. Run the script again

---

## Windows Setup

The script works on Windows with **one small change** already baked in: it will prefer `cv2.CAP_DSHOW` when available.

If the camera does not open, try a different camera index:

```bash
python3 steering_wheel.py --camera 1
```

**Steps for Windows:**

1. Install Python from https://python.org

2. Install dependencies:
   ```bash
   pip install mediapipe opencv-python pynput numpy
   ```

3. Run:
   ```bash
   python steering_wheel.py
   ```

4. If the camera doesn't open, try changing `CAMERA_INDEX` at the top of `steering_wheel.py`:
   ```python
   CAMERA_INDEX = 0   # try 0, 1, or 2
   ```

> **Note for Windows:** You may see a Windows Security prompt asking if Python can access your camera — click **Allow**.

---

## Config (top of `steering_wheel.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `CAMERA_INDEX` | `0` | `0` = built-in webcam, `1` = external USB camera |
| `DEAD_ZONE_DEG` | `12` | Degrees of tilt to ignore at center (prevents jitter) |
| `FLIP_CAMERA` | `True` | Mirror the feed (selfie view). Set `False` for some external cameras |
| `GRACE_FRAMES` | `8` | Frames to wait before releasing keys when hands disappear |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `[ERROR] Cannot open camera` | Check `CAMERA_INDEX` — try `0`, `1`, `2` |
| `Wrote test output` but no hands are detected | The default sample image may not contain real hands; try a real photo with both hands visible or use the webcam mode |
| Steering stays straight | Ensure both hands appear in the webcam frame with good lighting and visible wrist positions |
| Steering is reversed | Toggle `FLIP_CAMERA = False` in config |
| Keys stuck after removing hands | Hands must be fully out of frame for ~8 frames |
| Low FPS / laggy | Lower camera resolution in the script or close other apps |

---

## Works With Any Game That Uses Arrow Keys

- Google Chrome Dinosaur game
- Trackmania
- TORCS
- Hill Climb Racing (browser)
- Any browser/PC racing game using arrow keys

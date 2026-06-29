#  Project 1: Object Tracker

---

## What We Are Building

<video controls width="100%"><source src="/original/project_1.mp4" type="video/mp4"></video>

---

You have built four exercises. Each one taught one hard thing in isolation. Project 1 is where all four converge into one closed-loop robot: your model detects the target, the phone supplies video, encoder hardware measures motion, and UDP carries commands and telemetry between the laptop and Arduino.
Exercise A trained best.pt YOLO26n detection, class names, confidence, and bounding-box area. Exercise B phone stream MJPEG over WiFi, JPEG markers, OpenCV decoding, stream testing. Exercise C encoder motors Precise movement, ticks to distance, ramping, and calibration. Exercise D UDP WiFi Command packets to the robot and encoder telemetry back to Python. Project 1 YOLO detects ball -> PID computes steering -> UDP sends motor commands -> encoders return -> state machine manages behavior -> Heads-Up Display (HUD) shows it live
All four exercises converging --- the robot detects, steers, and tracks autonomously.

### The Five Components

1. **Object Detection**  
   **What it does:** finds the ball in every camera frame and returns center position plus bounding-box area.  
   **Where you learned it:** Exercise A, especially PA.8 and PA.9.  
   **New here:** none. It is the same YOLO call you used in `detect_stream.py`.

2. **Camera Stream**  
   **What it does:** pulls MJPEG frames from your phone camera.  
   **Where you learned it:** Exercise B, PB.5 and PB.6.  
   **New here:** a background thread so frame reading does not block robot control.

3. **PID Controller**  
   **What it does:** converts error into motor correction continuously.  
   **Where you learned the idea:** Exercise C's distance control in PC.8.  
   **New here:** continuous adjustment instead of a threshold stop.

4. **UDP Communication**  
   **What it does:** sends `MOTOR` commands and receives `ENC` telemetry.  
   **Where you learned it:** Exercise D, PD.2, PD.3, and PD.6.  
   **New here:** none. `Commander` and `Telemetry` wrap Exercise D sockets in classes.

5. **State Machine**  
   **What it does:** manages behavior across `STOPPED`, `SEARCHING`, `ACQUIRING`, and `TRACKING`.  
   **Where you saw the concept:** new in Project 1 --- a hand-designed finite state machine, *not* the learned states of Chapter 5's reinforcement learning.  
   **New here:** states directly choose motor outputs and timeout behavior.

### Pre-flight Checklist
best.pt is in the project folder and detects your target above 0.75 confidence. Phone stream opens in a browser without timing out. Robot drives 30 cm with less than 2 cm error. WASD keyboard control works over UDP. You know your robot's ESP_IP from Exercise D Serial Monitor output. You know your phone stream URL from Exercise B.
!!! warning "Do not skip broken prerequisites"
    If any box is unchecked, finish that exercise before continuing. Project 1 is an integration project; it will not work if its detector, camera stream, motor base, or UDP link is already broken.

### Full Code
This is the complete file for reference. Do not try to read it all now --- it will not make sense yet. As you work through each lesson, come back here to find the exact section being explained. By the end of the project every line will be familiar.
Complete obj_track_adv.pyCopy or download
```python
from shared import PID, MobileVideoStream, Commander, Telemetry, RobotState, ramp
import cv2
import numpy as np
import time
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# Network
ESP_IP         = "YOUR_ESP_IP"        # e.g. "192.168.1.100"
MOBILE_IP      = "YOUR_PHONE_IP"      # e.g. "192.168.1.101"
UDP_CMD_PORT   = 5001
UDP_TELEM_PORT = 5002

# Vision
TARGET_OBJECT = "ball"
TARGET_AREA   = 143_000

# Turn PID
BODY_TURN_KP = 10
BODY_TURN_KI = 0
BODY_TURN_KD = 3

# Distance PID
DIST_KP = 10
DIST_KI = 0
DIST_KD = 5
MAX_DIST_INTEGRAL = 0.8

# Motion limits
MAX_SPEED = 30
MIN_SPEED = 3
MAX_TURN  = 30
MAX_ACCEL = 2

# Dead zones (normalised)
ANGLE_DEAD_ZONE = 0.00
AREA_DEAD_ZONE  = 0.08

# Smoothing / state-machine
SMOOTH_RATE    = 0.04
ACQUIRE_FRAMES = 5
LOST_TIMEOUT   = 1.0


# ──────────────────────────────────────────────────────────────────────────────
#  TRACKER
# ──────────────────────────────────────────────────────────────────────────────

class Tracker:
    """YOLO-based ball tracker with PID control and a 4-state state machine."""

    STATE_COLORS = {
        RobotState.STOPPED:   (100, 100, 100),
        RobotState.SEARCHING: (255, 165,   0),
        RobotState.ACQUIRING: (255, 255,   0),
        RobotState.TRACKING:  (  0, 255,   0),
    }

    def __init__(self, model_path, target):
        print("Loading YOLO model...")
        self.model  = YOLO(model_path)
        self.target = target

        try:
            self.model.to("cuda")
            print("Using CUDA")
        except Exception:
            print("Using CPU")
        self.model.fuse()

        self.pid_turn = PID(BODY_TURN_KP, BODY_TURN_KI, BODY_TURN_KD,
                            output_limits=(-MAX_TURN, MAX_TURN))
        self.pid_dist = PID(DIST_KP, DIST_KI, DIST_KD,
                            max_integral=MAX_DIST_INTEGRAL,
                            output_limits=(-MAX_SPEED, MAX_SPEED))

        self.state          = RobotState.STOPPED
        self.acquire_count  = 0
        self.last_seen_t    = 0.0
        self.last_turn_dir  = 1.0
        self.smoothed_angle = 0.0
        self.speed_limit    = 1.0
        self.cur_left       = 0.0
        self.cur_right      = 0.0

    # ── Detection ────────────────────────────────────────────────────────────

    def _detect(self, frame):
        """Return the bounding box of the largest detected target, or None."""
        results   = self.model(frame, imgsz=320, verbose=False, conf=0.60)
        best      = None
        best_area = 0

        for r in results:
            for box in r.boxes:
                if self.model.names[int(box.cls[0])] != self.target:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best      = (x1, y1, x2, y2)

        return best

    # ── State machine ─────────────────────────────────────────────────────────

    def _update_state(self, bbox_found):
        if bbox_found:
            self.last_seen_t = time.time()

            if self.state in (RobotState.STOPPED, RobotState.SEARCHING):
                self.state         = RobotState.ACQUIRING
                self.acquire_count = 0
                self.speed_limit   = 0.4

            elif self.state == RobotState.ACQUIRING:
                self.acquire_count += 1
                self.speed_limit    = 0.4 + 0.6 * self.acquire_count / ACQUIRE_FRAMES
                if self.acquire_count >= ACQUIRE_FRAMES:
                    self.state       = RobotState.TRACKING
                    self.speed_limit = 1.0

            elif self.state == RobotState.TRACKING:
                self.speed_limit = 1.0

        else:
            elapsed = time.time() - self.last_seen_t
            if elapsed < LOST_TIMEOUT:
                self.state       = RobotState.SEARCHING
                self.speed_limit = 0.3
            else:
                self.state          = RobotState.STOPPED
                self.speed_limit    = 0.0
                self.smoothed_angle = 0.0
                self.pid_turn.reset()
                self.pid_dist.reset()

    # ── Control ──────────────────────────────────────────────────────────────

    def _motor_outputs_for_ball(self, cx, area, frame_width):
        """Compute raw left/right targets when the ball is visible."""
        raw_angle_error = (cx - frame_width / 2) / frame_width

        if abs(raw_angle_error) < ANGLE_DEAD_ZONE:
            raw_angle_error = 0.0
        if raw_angle_error != 0.0:
            self.last_turn_dir = np.sign(raw_angle_error)

        self.smoothed_angle = raw_angle_error
        turn_cmd = self.pid_turn.update(self.smoothed_angle)

        dist_error = (area - TARGET_AREA) / TARGET_AREA
        if abs(dist_error) < AREA_DEAD_ZONE:
            dist_error = 0.0
        self.pid_dist.update(dist_error)

        base_speed = 10
        tgt_left   = base_speed + turn_cmd
        tgt_right  = base_speed - turn_cmd

        # Scale to MAX_SPEED without clipping
        max_raw = max(abs(tgt_left), abs(tgt_right))
        if max_raw > MAX_SPEED:
            scale     = MAX_SPEED / max_raw
            tgt_left  *= scale
            tgt_right *= scale

        return tgt_left, tgt_right, dist_error, turn_cmd, base_speed

    def control(self, frame):
        """Run one control cycle; returns (left_speed, right_speed, debug_dict)."""
        h, w = frame.shape[:2]
        bbox = self._detect(frame)
        self._update_state(bbox is not None)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cx   = (x1 + x2) / 2
            area = (x2 - x1) * (y2 - y1)
            tgt_left, tgt_right, dist_error, turn_cmd, base_speed = \
                self._motor_outputs_for_ball(cx, area, w)

        elif self.state == RobotState.SEARCHING:
            tgt_left   =  10 * self.last_turn_dir * self.speed_limit
            tgt_right  = -10 * self.last_turn_dir * self.speed_limit
            dist_error = turn_cmd = base_speed = 0.0
            area = 0
            cx   = w / 2

        else:  # STOPPED
            tgt_left = tgt_right = 0
            dist_error = turn_cmd = base_speed = 0.0
            area = 0
            cx   = w / 2

        self.cur_left  = ramp(self.cur_left,  tgt_left,  MAX_ACCEL)
        self.cur_right = ramp(self.cur_right, tgt_right, MAX_ACCEL)

        debug = {
            "state":          self.state,
            "bbox":           bbox,
            "smoothed_angle": self.smoothed_angle,
            "turn_cmd":       turn_cmd,
            "dist_error":     dist_error,
            "base_speed":     base_speed,
            "area":           area,
            "cx":             int(cx) if bbox is None else int((bbox[0] + bbox[2]) / 2),
            "speed_limit":    self.speed_limit,
        }
        return round(self.cur_left), round(self.cur_right), debug

    def reset(self):
        self.pid_turn.reset()
        self.pid_dist.reset()
        self.smoothed_angle = 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  VISUALISATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def draw_overlay(frame, debug, left_speed, right_speed, telem, scale_x, scale_y):
    h, w = frame.shape[:2]
    state_color = Tracker.STATE_COLORS.get(debug["state"], (255, 255, 255))

    # Centre line
    cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)

    # Smoothed-angle indicator
    smooth_x = int(w / 2 + debug["smoothed_angle"] * w)
    cv2.line(frame, (smooth_x, 0), (smooth_x, h), (0, 255, 255), 2)
    cv2.putText(frame, "SMOOTH", (smooth_x + 5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Bounding box
    if debug["bbox"] is not None:
        x1, y1, x2, y2 = debug["bbox"]
        x1, y1, x2, y2 = (int(x1 * scale_x), int(y1 * scale_y),
                           int(x2 * scale_x), int(y2 * scale_y))

        if abs(debug["dist_error"]) < AREA_DEAD_ZONE:
            box_color = (0, 255, 0)     # green  — correct distance
        elif debug["dist_error"] > 0:
            box_color = (0, 0, 255)     # red    — too close
        else:
            box_color = (255, 100, 0)   # blue   — too far

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cy = (y1 + y2) // 2
        cv2.circle(frame, (int(debug["cx"] * scale_x), cy), 6, (0, 0, 255), -1)

    # Text HUD
    lines = [
        (f"STATE:  {debug['state']}",                                   state_color),
        (f"SMOOTH: {debug['smoothed_angle']:+.3f}",                     (0, 255, 255)),
        (f"AREA:   {int(debug['area'])} / {TARGET_AREA}",               (255, 255, 255)),
        (f"SPD LIM:{debug['speed_limit']:.0%}",                         (255, 255, 255)),
        (f"L:{int(left_speed):+4d} R:{int(right_speed):+4d}",           (0, 255, 0)),
        (f"ENC L:{telem['left_ticks']:+4d} R:{telem['right_ticks']:+4d}", (255, 165, 0)),
    ]
    for i, (text, color) in enumerate(lines):
        thickness = 2 if i in (0, 4) else 1
        cv2.putText(frame, text, (10, 25 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65 if i in (0, 4) else 0.55,
                    color, thickness)


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    global SMOOTH_RATE

    stream_url = f"http://{MOBILE_IP}:8080/video"
    print(f"Stream: {stream_url}")

    video     = MobileVideoStream(stream_url)
    commander = Commander(ESP_IP, UDP_CMD_PORT)
    telemetry = Telemetry(UDP_TELEM_PORT)

    print("Waiting for camera", end="", flush=True)
    for _ in range(30):
        if video.connected:
            break
        time.sleep(0.5)
        print(".", end="", flush=True)
    print()

    if not video.connected:
        print("Could not connect to camera")
        return

    tracker = Tracker("best.pt", TARGET_OBJECT)
    commander.stop()
    time.sleep(0.5)

    print(f"\nTracking: {TARGET_OBJECT}  |  Target area: {TARGET_AREA}")
    print("Q=Quit  R=Reset  F=Smooth faster  S=Smooth slower")
    print("=" * 60)

    last_fid      = -1
    last_cmd_time = 0.0
    CMD_INTERVAL  = 0.03
    fps_count     = 0
    fps_timer     = time.time()

    while True:
        frame, fid = video.read()
        if frame is None or fid == last_fid:
            time.sleep(0.005)
            continue

        last_fid   = fid
        fps_count += 1
        if fps_count >= 30:
            fps = fps_count / (time.time() - fps_timer)
            print(f"FPS: {fps:.1f}  SMOOTH_RATE: {SMOOTH_RATE:.2f}")
            fps_count = 0
            fps_timer = time.time()

        left_speed, right_speed, debug = tracker.control(frame)
        print(left_speed, right_speed)

        orig_h, orig_w = frame.shape[:2]
        frame = cv2.resize(frame, (640, 480))
        h, w  = frame.shape[:2]

        # Send motor command at a fixed interval
        now = time.time()
        if now - last_cmd_time >= CMD_INTERVAL:
            commander.motors(left_speed, right_speed)
            last_cmd_time = now

        draw_overlay(frame, debug, left_speed, right_speed,
                     telemetry.read(), w / orig_w, h / orig_h)

        cv2.imshow(f"Object Tracker — {TARGET_OBJECT}", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            tracker.reset()
            print("Reset")
        elif key == ord("f"):
            SMOOTH_RATE = min(SMOOTH_RATE + 0.02, 0.5)
            print(f"SMOOTH_RATE → {SMOOTH_RATE:.2f}")
        elif key == ord("s"):
            SMOOTH_RATE = max(SMOOTH_RATE - 0.02, 0.02)
            print(f"SMOOTH_RATE → {SMOOTH_RATE:.2f}")

    print("\nShutting down...")
    commander.stop()
    time.sleep(0.2)
    video.stop()
    telemetry.stop()
    cv2.destroyAllWindows()
    print("Done")


if __name__ == "__main__":
    main()
```
Download this file and place it in your `my-detector` project folder. You will not run it until the final lesson.

---

## Get the Code


---

The code for this project is in the course GitHub repository. Download both files before continuing.
Step 1 — Open the repository. Go to github.com/purwar-lab/ml-for-robotics- . Step 2 — Download the two files below. Save both files to your my-detector project folder from Exercise A. Step 3 — Verify the files are there. Open VS Code and open the my-detector folder. You should see shared.py and obj_track_adv.py in the file explorer on the left. Step 4 — Open the project file and read the configuration block. Open obj_track_adv.py . The very top section after the import lines is the CONFIGURATION block. Every value in this section must be filled in before running. The next lesson walks through each one.

### Download Files
**`shared.py`**
Reusable infrastructure: PID controller, camera stream, UDP commands, telemetry, and state machine constants. Used by both Project 1 and Project 2.
**`obj_track_adv.py`**
Project 1 specific code: YOLO detection, the Tracker class, HUD overlay, and the main loop.
Both files must be in the same folder. `shared.py` is imported by `obj_track_adv.py` automatically when you run it --- you never open or edit `shared.py` directly.
!!! info "Do not run the files yet"
    They will not work with the default placeholder values. Complete every lesson in this project first, then run the project file in the final lesson.

---

## Setting Up


---

If you completed Exercise A, your VS Code setup is already done. Project 1 runs in the same folder and virtual environment you used for your detector.
Open your my-detector folder in VS Code. Activate the same virtual environment you used in Exercise A. On Windows use venv\Scripts\activate . On macOS or Linux use source venv/bin/activate . Copy shared.py and obj_track_adv.py into this folder if they are not already there. Copy your trained best.pt into the same folder as obj_track_adv.py .
!!! tip "Reuse the Exercise A environment"
    In Exercise A you created `my-detector` and installed `ultralytics`, `opencv-python`, and `requests`. Those are the core packages Project 1 needs. You do not need a new folder or a new virtual environment.
!!! warning "Use the same terminal context"
    If `python obj_track_adv.py` cannot find `ultralytics`, you are probably in the wrong folder or the virtual environment is not activated.

---

## Installing Dependencies


---

Project 1 only adds one dependency beyond what you already used in Exercises A and B.

### Already Installed From Earlier Projects
Package Where you used it What Project 1 uses it for ultralytics Exercise A Loads best.pt and runs YOLO detection. opencv-python Exercises A and B Decodes frames, draws the HUD, and shows the video window. requests Exercise B Connects to the phone MJPEG stream.

### The Only New Package
`numpy` may already be installed because `ultralytics` depends on it. Verify first:
Check NumPy
```bash
python -c "import numpy; print(numpy.__version__)"
```
If that prints a version number, you are done. If it throws an import error, install NumPy:
Install NumPy only if needed
```bash
pip install numpy
```

### Full Dependency Check
Create `check_deps.py` in the same folder as `obj_track_adv.py` and run it.
check_deps.py
```python
import cv2
import numpy as np
import requests
import socket
import threading
from ultralytics import YOLO

print("All dependencies present.")
```
Run the check
```bash
python check_deps.py
```
If `All dependencies present.` prints, move on.

---

## Configuration: Tuning Your Robot


---

Project 1 becomes your robot when you fill in the configuration block at the top of `obj_track_adv.py`. Every value comes from a previous project or from one calibration step you perform now.
Constant Source What to enter ESP_IP Exercise D, PD.2 The IP printed by the Arduino Serial Monitor when robot_udp boots. If you lost it, re-upload Exercise D's sketch and read Serial Monitor again. UDP_CMD_PORT = 5001 Exercise D, PD.0 Leave as 5001 . It must match CMD_PORT in the Arduino sketch. UDP_TELEM_PORT = 5002 Exercise D, PD.0 Leave as 5002 . It must match TELEM_PORT in the Arduino sketch. TARGET_OBJECT Exercise A, PA.3 The exact Roboflow class name you trained, such as "ball" . Capitalization and spelling must match. TARGET_AREA Calibrate now The pixel area of the bounding box when the robot is at the distance you want it to hold. PID gains P1.4 Leave defaults for the first run. Tune only after the staged tests work. MAX_SPEED First-run safety Start at 25 , not 30 . Slower is safer for the first real robot test. MAX_ACCEL Exercise C ramping concept Keep at 2 so motor commands change gradually. mobile_ip Exercise B, PB.3 The IP shown in IP Webcam or Simple IP Camera when the stream is active.

### Calibrate TARGET_AREA
Run detect_stream.py from Exercise B. Hold your target object at the distance you want the robot to maintain, usually 40 to 60 cm in front of the phone. Read the area value from your PA.9 debug output. If your script does not print area yet, temporarily add the snippet below. Set TARGET_AREA to the area value at that distance.
Temporary area print
```python
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    area = (x2 - x1) * (y2 - y1)
    print(f"Area at current distance: {int(area)}")
```

### Fill-in Table
Value Your entry Source ESP_IP "____________" Exercise D Serial Monitor TARGET_OBJECT "____________" Exercise A Roboflow class name TARGET_AREA ____________ Calibrated with detect_stream.py mobile_ip "____________" Exercise B phone app stream_url "____________" http://mobile_ip:8080/video or the iPhone app URL
!!! warning "Configuration errors look like code bugs"
    A wrong `TARGET_OBJECT`, stale `ESP_IP`, or badly calibrated `TARGET_AREA` can make correct code behave incorrectly. Fill in this table before changing any PID values.

---

## The Shared Infrastructure


---

Before looking at the Project 1 specific code, it helps to understand what is in `shared.py`. This file is the foundation both projects 1 and 2 are built on. You will study it once here and never need to re-read it in Project 2 --- it will already be familiar.
Complete shared.py
```python
# shared.py
# Reusable infrastructure used by both obj_track_adv.py
# and lane_follower_adv.py.
# Contains: PID, MobileVideoStream, Commander, Telemetry,
# RobotState, ramp()

import cv2
import numpy as np
import time
import socket
import threading
import requests


class PID:
    """Proportional-Integral-Derivative controller with optional output clamping."""

    def __init__(self, kp, ki, kd, max_integral=100.0, output_limits=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral  = max_integral
        self.output_limits = output_limits
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time  = None

    def update(self, error):
        now = time.time()
        dt  = 0.02 if self._prev_time is None else max(now - self._prev_time, 1e-4)
        self._prev_time = now

        self._integral   = np.clip(self._integral + error * dt,
                                   -self.max_integral, self.max_integral)
        derivative       = (error - self._prev_error) / dt
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        if self.output_limits:
            output = np.clip(output, *self.output_limits)
        return output

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time  = None

class MobileVideoStream:
    """Continuously pulls MJPEG frames from a phone camera in a background thread."""

    def __init__(self, url):
        self.url       = url
        self.frame     = None
        self.fid       = 0
        self.connected = False
        self.running   = True
        self._lock     = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        backoff = 1.0
        while self.running:
            try:
                response = requests.get(self.url, stream=True, timeout=10)
                if response.status_code != 200:
                    raise ConnectionError("Non-200 status")

                self.connected = True
                backoff = 1.0
                buf = b""

                for chunk in response.iter_content(1024):
                    buf += chunk
                    start = buf.find(b"\xff\xd8")
                    end   = buf.find(b"\xff\xd9")

                    if start != -1 and end != -1:
                        jpg = buf[start : end + 2]
                        buf = buf[end + 2 :]
                        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                        if img is not None:
                            with self._lock:
                                self.frame = img
                                self.fid  += 1

                    if not self.running:
                        break

            except Exception:
                self.connected = False
                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)

    def read(self):
        with self._lock:
            return self.frame, self.fid

    def stop(self):
        self.running = False

class Commander:
    """Sends UDP motor commands to the ESP."""

    def __init__(self, ip, port):
        self.ip    = ip
        self.port  = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def motors(self, left, right, light="off"):
        msg = f"MOTOR,{int(left)},{int(right)},{light.upper()}"
        self._sock.sendto(msg.encode(), (self.ip, self.port))

    def stop(self):
        self._sock.sendto(b"STOP,OFF", (self.ip, self.port))

class Telemetry:
    """Receives encoder telemetry from the ESP over UDP."""

    def __init__(self, port):
        self.left_ticks  = 0
        self.right_ticks = 0
        self.cmd_left    = 0
        self.cmd_right   = 0
        self.running     = True
        self._lock       = threading.Lock()
        self._sock       = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("", port))
        self._sock.settimeout(1.0)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.running:
            try:
                data, _ = self._sock.recvfrom(128)
                parts   = data.decode().strip().split(",")
                if parts[0] == "ENC" and len(parts) == 5:
                    with self._lock:
                        self.left_ticks  = int(parts[1])
                        self.right_ticks = int(parts[2])
                        self.cmd_left    = int(parts[3])
                        self.cmd_right   = int(parts[4])
            except Exception:
                pass

    def read(self):
        with self._lock:
            return {
                "left_ticks":  self.left_ticks,
                "right_ticks": self.right_ticks,
                "cmd_left":    self.cmd_left,
                "cmd_right":   self.cmd_right,
            }

    def stop(self):
        self.running = False
        self._sock.close()

class RobotState:
    STOPPED   = "STOPPED"
    SEARCHING = "SEARCHING"
    ACQUIRING = "ACQUIRING"
    TRACKING  = "TRACKING"

def ramp(current, target, max_step):
    """Limit how fast a value can change per step."""
    diff = target - current
    if abs(diff) <= max_step:
        return target
    return current + max_step * np.sign(diff)
```

---

## The PID Controller


---

In Exercise C's `driveDistance()` function, the robot stopped when encoder ticks reached a target. That is closed-loop in the sense that it checks the measured result, but it only makes one decision: keep going or stop.
A PID controller makes a continuous adjustment. Every few milliseconds it measures the current error, computes a correction, applies it, and measures again. It never stops adjusting until the error is zero.

### Exercise C Code You Already Wrote
You have already written closed-loop control. In Exercise C, `driveDistance()` measured encoder ticks, compared them to a target, and changed motor behavior when the robot got close.
Exercise C driveDistance()Reference
```cpp
void driveDistance(float targetCm, int speed) {
  long targetTicks = abs(targetCm) / CM_PER_TICK;
  resetEncoders();
  setMotors(speed, speed);

  while (ticksLeft < targetTicks && ticksRight < targetTicks) {
    long remaining = targetTicks - max(ticksLeft, ticksRight);
    if (remaining < 50) {
      int slowSpeed = max(80, speed / 3);
      setMotors(slowSpeed, slowSpeed);
    }
  }

  stopMotors();
}
```
Line Control idea targetTicks The desired result, converted from centimeters into encoder ticks. ticksLeft , ticksRight The measured result from the hardware. remaining The error: how far the robot still needs to travel. if (remaining < 50) A threshold decision: when the error is small, slow down. stopMotors() The final action when the measured value reaches the target.
**Exercise C approach** — `if remaining < 50 ticks: slow down`
`if remaining == 0: stop`
**Project 1 PID approach** — `correction = Kp * error + Ki * integral + Kd * derivative`
`motor_output = base + correction`
The Exercise C version had two output levels: full speed or slow speed. The PID version produces a continuous output, so a large error creates a large correction and a small error creates a small correction.

### The Project 1 PID Class
This class is in `shared.py` --- the version below is identical in both projects.
Project 1 PID classshared.py
```python
class PID:
    def __init__(self, kp, ki, kd, max_integral=100.0, output_limits=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral = max_integral
        self.output_limits = output_limits
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def update(self, error):
        now = time.time()
        dt = 0.02 if self._prev_time is None else max(now - self._prev_time, 1e-4)
        self._prev_time = now

        self._integral = np.clip(
            self._integral + error * dt,
            -self.max_integral,
            self.max_integral,
        )
        derivative = (error - self._prev_error) / dt
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        if self.output_limits:
            output = np.clip(output, *self.output_limits)
        return output

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None
```
self.kp , self.ki , self.kd These are the three tuning knobs. Exercise C used one threshold, 50 ticks. Project 1 separates the response into current error, accumulated error, and rate of change. dt : time since the previous update The integral and derivative terms only make sense when time is measured. The first call assumes 0.02 seconds, about 50 Hz. The max(..., 1e-4) guard prevents division by zero if two updates happen almost instantly. self._integral + error * dt This accumulates error over time. np.clip() prevents windup, the PID version of a robot continuing to push harder after it has been blocked for too long. derivative = (error - self._prev_error) / dt This measures whether the error is shrinking or growing. Exercise C could slow down near the target, but it could not tell whether the robot was already correcting quickly enough. output = kp * error + ki * integral + kd * derivative This is the PID equation translated into code. The result is one continuous correction value, not a binary full-speed-or-slow-speed decision. reset() Exercise C called resetEncoders() at the start of a movement. Project 1 calls reset() when tracking fully stops, so stale integral and derivative memory do not affect a new tracking session.

### Why There Are Two PIDs
This code is in `obj_track_adv.py`.
Two independent controllersobj_track_adv.py
```python
self.pid_turn = PID(
    BODY_TURN_KP,
    BODY_TURN_KI,
    BODY_TURN_KD,
    output_limits=(-MAX_TURN, MAX_TURN),
)
self.pid_dist = PID(
    DIST_KP,
    DIST_KI,
    DIST_KD,
    max_integral=MAX_DIST_INTEGRAL,
    output_limits=(-MAX_SPEED, MAX_SPEED),
)
```
`pid_turn` controls steering. Its error is how far the ball is from the center of the frame, and its output is added to one motor while being subtracted from the other.
`pid_dist` controls distance. Its error is how far the bounding-box area is from `TARGET_AREA`, and its output becomes the forward or reverse base speed.
Combining turn and distance outputsControl logic
```python
tgt_left = base_speed + turn_cmd
tgt_right = base_speed - turn_cmd
```
Concept Exercise C Project 1 PID What is measured encoder ticks bounding box area What is the target targetTicks TARGET_AREA What is the error targetTicks - ticksLeft (area - TARGET_AREA) / TARGET_AREA What happens at zero error stop motors keep adjusting gently around the target Response type binary: go or stop continuous: always adjusting
**P: Proportional**
Current error. Big error produces a big correction. Too much `Kp` creates oscillation.
**I: Integral**
Accumulated old error. It corrects persistent offset but can cause windup.
**D: Derivative**
Error slope. It dampens overshoot by reacting when error changes quickly.

### Integral Windup
In Exercise C, if the robot could not reach the target because of wheel slip, it could keep trying forever. The PID equivalent is integral windup: if the robot cannot close a distance gap, the integral term grows until the robot lunges when the path clears.
`MAX_DIST_INTEGRAL` clamps the accumulated distance error so old failure does not turn into a dangerous future command.

### Tuning Procedure
Phase 1: Angle PID only. Temporarily set MAX_SPEED = 0 . Set BODY_TURN_KP = 5 , BODY_TURN_KI = 0 , and BODY_TURN_KD = 0 . Run the tracker. Increase KP until oscillation is strong, then reduce by about 40 percent. Add KD , starting near 0.3 * KP , to dampen overshoot. Phase 2: Distance PID only. Restore MAX_SPEED = 25 . Hold the ball directly in front at target distance and move it closer and farther. Increase DIST_KP until the robot responds, then reduce by about 30 percent if it oscillates. Add DIST_KD if the approach is jerky. Phase 3: Both together. Run normally. If the robot oscillates left and right while approaching, increase BODY_TURN_KD . If it cannot close distance while turning, reduce BODY_TURN_KP slightly so steering corrections do not cancel forward speed.
!!! tip "Goal of tuning"
    A tuned tracker turns smoothly toward the ball, approaches without lunging, and settles near the calibrated distance without hunting back and forth.

---

## Reading the Camera Stream


---

The `MobileVideoStream` class is `detect_stream.py` from Exercise B rewritten as a class with a background thread. Open both files side by side before reading further.

### Exercise B Code: Sequential Frame Reading
In Exercise B, the stream reader could wait for a frame because the script only needed to detect, draw, and display. Project 1 cannot wait like that because the robot must keep sending motor corrections.
Exercise B detect_stream.py core loopReference
```python
def read_frame(stream_response):
    buf = b""
    for chunk in stream_response.iter_content(1024):
        buf += chunk
        start = buf.find(b"\xff\xd8")
        end = buf.find(b"\xff\xd9")
        if start != -1 and end != -1:
            jpg = buf[start:end + 2]
            buf = buf[end + 2:]
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                return frame
    return None

while True:
    frame = read_frame(stream)       # wait for a frame
    results = model(frame, ...)      # then detect
    annotated = results[0].plot()    # then draw
    cv2.imshow("Detection", annotated)
```

### Project 1 Code: Threaded Frame Reading
`MobileVideoStream` is in `shared.py`.
Project 1 MobileVideoStreamshared.py
```python
class MobileVideoStream:
    def __init__(self, url):
        self.url = url
        self.frame = None
        self.fid = 0
        self.connected = False
        self.running = True
        self._lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        backoff = 1.0
        while self.running:
            try:
                response = requests.get(self.url, stream=True, timeout=10)
                self.connected = True
                backoff = 1.0
                buf = b""

                for chunk in response.iter_content(1024):
                    buf += chunk
                    start = buf.find(b"\xff\xd8")
                    end = buf.find(b"\xff\xd9")

                    if start != -1 and end != -1:
                        jpg = buf[start:end + 2]
                        buf = buf[end + 2:]
                        img = cv2.imdecode(
                            np.frombuffer(jpg, np.uint8),
                            cv2.IMREAD_COLOR,
                        )
                        if img is not None:
                            with self._lock:
                                self.frame = img
                                self.fid += 1

                    if not self.running:
                        break

            except Exception:
                self.connected = False
                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)

    def read(self):
        with self._lock:
            return self.frame, self.fid

    def stop(self):
        self.running = False
```
Concept Exercise B detect_stream.py Project 1 MobileVideoStream Open HTTP connection requests.get(url, stream=True) Same line, same parameters. Read bytes response.iter_content(1024) Same line. Find JPEG markers buf.find(b"\xff\xd8") Same bytes, same logic. Decode frame cv2.imdecode(...) Identical. Thread? No, it runs in the main loop. Yes, it runs in the background. Why thread? Not needed for a simple stream demo. The main loop also runs YOLO, PID, UDP, telemetry, and HUD drawing. Reconnect on failure? No. Yes, with exponential backoff.

### Transformation Notes
Exercise B line Project 1 line Why it changed read_frame() _run() inside a class The reader now stores state: latest frame, frame ID, connection status, and running flag. Main loop waits at read_frame() threading.Thread(...).start() The camera reader runs in parallel so YOLO, PID, UDP, and HUD work do not pause while the next JPEG arrives. No reconnect logic try , except , and backoff A physical robot must survive WiFi drops without restarting the whole Python program. No duplicate-frame marker self.fid += 1 The main loop can skip YOLO inference when it sees the same frame ID twice. No shared-data protection with self._lock The background thread writes frames while the main loop reads them, so both sides need a lock.

### The Only New Concepts
**Threading:** Exercise B read frames one at a time in the main loop because there was nothing else to do between frames. Project 1's main loop runs detection, PID computations, UDP sends, and telemetry reads. If it also waited for the next JPEG from the phone, the robot control loop would be as slow as the network. The background thread decouples camera speed from detection speed, so the latest available frame is always ready.
**The lock:** `self._lock` prevents the background thread from writing `self.frame` at the exact same moment the main loop is reading it. Without the lock, the main loop could read a half-written frame with corrupted pixels.
**The frame ID:** The main loop may run faster than the phone camera. If the camera delivers 15 fps and the main loop checks at 30 Hz, every frame may be seen twice. `fid` lets the main loop detect duplicates and skip expensive repeated YOLO inference.
Skipping duplicate framesMain loop
```python
if fid == last_fid:
    time.sleep(0.005)
    continue
```
!!! info "Exercise B is the reference"
    If this class feels unfamiliar, go back to PB.6 and compare the stream parsing lines. Most of the code is the same; Project 1 just packages it for a larger control loop.

---

## Sending Commands (UDP)


---

The `Commander` class is `cmd_sock` from Exercise D wrapped in a class. Open `robot_control.py` from Exercise D beside `shared.py`.

### Exercise D Code: Two Simple Functions
Exercise D robot_control.py senderReference
```python
cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_command(left, right):
    msg = f"MOTOR,{int(left)},{int(right)}"
    cmd_sock.sendto(msg.encode(), (ARDUINO_IP, CMD_PORT))

def send_stop():
    cmd_sock.sendto(b"STOP", (ARDUINO_IP, CMD_PORT))
```

### Project 1 Code: The Same Sender as a Class
`Commander` is in `shared.py`.
Project 1 Commandershared.py
```python
class Commander:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def motors(self, left, right, light="off"):
        msg = f"MOTOR,{int(left)},{int(right)},{light.upper()}"
        self._sock.sendto(msg.encode(), (self.ip, self.port))

    def stop(self):
        self._sock.sendto(b"STOP,OFF", (self.ip, self.port))
```
Exercise D robot_control.py Project 1 Commander cmd_sock = socket.socket(...) self._sock = socket.socket(...) cmd_sock.sendto(msg.encode(), ...) self._sock.sendto(msg.encode(), ...) msg = f"MOTOR,{left},{right}" msg = f"MOTOR,{left},{right},{light}"
Exercise D Project 1 Why it changed ARDUINO_IP global self.ip from __init__() The sender can be reused with any robot IP. CMD_PORT global self.port from __init__() The UDP port travels with the object instead of living as a loose global. send_command(left, right) motors(left, right, light) The command now carries motor speeds plus visible robot state. b"STOP" b"STOP,OFF" The same stop packet also tells the LED to turn off.
The only difference is the added LED state field. When Project 1 sends `MOTOR,150,100,TRACKING`, the Arduino sketch must parse the third field and set the RGB LED.
**The third field** — `MOTOR,leftSpeed,rightSpeed,LED_STATE`. The tracker passes the current state name so the robot can show `TRACKING`, `SEARCHING`, `ACQUIRING`, or `OFF` on the physical LED.
State travels with the motor commandMain loop
```python
commander.motors(left_speed, right_speed, debug["state"])
```

### Arduino Sketch Update: RGB LED State
Update the Exercise D `robot_udp` sketch so it accepts the three-field command and controls the WS2812B LED on pin `A3`.
Exercise D parseCommand()Before
```cpp
void parseCommand(char* packet) {
  if (strncmp(packet, "MOTOR,", 6) == 0) {
    int leftSpeed = 0;
    int rightSpeed = 0;
    sscanf(packet, "MOTOR,%d,%d", &leftSpeed, &rightSpeed);
    setMotors(leftSpeed, rightSpeed);
  }
}
```
LED and parseCommand update
```cpp
#include <FastLED.h>

#define LED_PIN  A3
#define NUM_LEDS 1

CRGB leds[NUM_LEDS];

void setupLED() {
  FastLED.addLeds<WS2812B, LED_PIN, GRB>(leds, NUM_LEDS);
  FastLED.setBrightness(50);
  leds[0] = CRGB::Black;
  FastLED.show();
}

void setLEDState(const char* state) {
  if (strcmp(state, "TRACKING") == 0) {
    leds[0] = CRGB::Green;
  } else if (strcmp(state, "SEARCHING") == 0) {
    leds[0] = CRGB::Orange;
  } else if (strcmp(state, "ACQUIRING") == 0) {
    leds[0] = CRGB::Yellow;
  } else {
    leds[0] = CRGB::Black;
  }
  FastLED.show();
}

void parseCommand(char* packet) {
  if (strncmp(packet, "STOP", 4) == 0) {
    stopMotors();
    leds[0] = CRGB::Red;
    FastLED.show();
    return;
  }

  if (strncmp(packet, "MOTOR,", 6) == 0) {
    int leftSpeed = 0;
    int rightSpeed = 0;
    char ledState[16] = "OFF";
    sscanf(packet, "MOTOR,%d,%d,%15s", &leftSpeed, &rightSpeed, ledState);
    setMotors(leftSpeed, rightSpeed);
    setLEDState(ledState);
    lastCmdTime = millis();
  }
}
```
Call `setupLED();` once inside Arduino `setup()` after the motor and encoder pins are configured and before printing `Ready.`.
!!! info "Install FastLED and check wiring"
    In Arduino IDE Library Manager, search **FastLED** and install the library by Daniel Garcia. The LED wiring from Exercise C is: data to `A3`, VCC to `5V`, and GND to GND.

### Why The LED Matters
LED Robot state Meaning Green TRACKING Ball found and confirmed. Orange SEARCHING Ball lost recently; robot is trying to reacquire. Yellow ACQUIRING Ball just appeared; robot is being cautious. Off STOPPED No active tracking command. Red STOP Stop packet received.

---

## Receiving Telemetry


---

The `Telemetry` class is `telem_sock` from Exercise D wrapped in a class with a background thread. The reason for the thread is identical to `MobileVideoStream`: the main loop cannot block waiting for a UDP packet.

### Exercise D Code: Blocking Telemetry Read
Exercise D robot_control.py telemetryReference
```python
telem_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
telem_sock.bind(("", TELEM_PORT))
telem_sock.settimeout(0.1)

def read_telemetry():
    try:
        data, _ = telem_sock.recvfrom(128)
        parts = data.decode().strip().split(",")
        if parts[0] == "ENC" and len(parts) == 3:
            return int(parts[1]), int(parts[2])
    except socket.timeout:
        pass
    return None, None
```

### Project 1 Code: Threaded Telemetry
`Telemetry` is in `shared.py`.
Project 1 Telemetryshared.py
```python
class Telemetry:
    def __init__(self, port):
        self.left_ticks = 0
        self.right_ticks = 0
        self.cmd_left = 0
        self.cmd_right = 0
        self.running = True
        self._lock = threading.Lock()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("", port))
        self._sock.settimeout(1.0)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.running:
            try:
                data, _ = self._sock.recvfrom(128)
                parts = data.decode().strip().split(",")
                if parts[0] == "ENC" and len(parts) == 5:
                    with self._lock:
                        self.left_ticks = int(parts[1])
                        self.right_ticks = int(parts[2])
                        self.cmd_left = int(parts[3])
                        self.cmd_right = int(parts[4])
            except Exception:
                pass

    def read(self):
        with self._lock:
            return {
                "left_ticks": self.left_ticks,
                "right_ticks": self.right_ticks,
                "cmd_left": self.cmd_left,
                "cmd_right": self.cmd_right,
            }

    def stop(self):
        self.running = False
        self._sock.close()
```
Exercise D robot_control.py Project 1 Telemetry telem_sock.bind(("", TELEM_PORT)) self._sock.bind(("", port)) telem_sock.settimeout(0.1) self._sock.settimeout(1.0) parts = data.decode().split(",") Same parsing pattern. parts[0] == "ENC" Same packet-type check. int(parts[1]), int(parts[2]) Same ticks, extended to five fields total.
Exercise D Project 1 Why it changed telem_sock global self._sock inside a class The socket and latest values are bundled together. read_telemetry() called from the loop _run() runs in a background thread The main loop never waits for recvfrom() . len(parts) == 3 len(parts) == 5 Project 1 adds last commanded left and right motor values. No lock needed threading.Lock() The telemetry thread writes while the HUD reads.

### Project 1 Telemetry Packet
Exercise D sent `ENC,leftTicks,rightTicks`. Project 1 sends five fields so the HUD can show both measured ticks and the last commanded wheel speeds:
**Telemetry packet** — `ENC,leftTicks,rightTicks,cmdLeft,cmdRight`

### Arduino Sketch Update
Exercise D's Arduino telemetry sent only encoder ticks:
Exercise D sendTelemetry()Before
```cpp
void sendTelemetry() {
  char buf[64];
  snprintf(buf, sizeof(buf), "ENC,%ld,%ld", ticksLeft, ticksRight);
  udp.beginPacket(LAPTOP_IP, TELEM_PORT);
  udp.write((uint8_t*)buf, strlen(buf));
  udp.endPacket();
}
```
Add these globals near the other motor and encoder globals:
Track last commanded speeds
```cpp
int lastCmdLeft = 0;
int lastCmdRight = 0;
```
Inside the `MOTOR` branch of `parseCommand()`, after `setMotors(leftSpeed, rightSpeed);`, store the values:
Store command values
```cpp
lastCmdLeft = leftSpeed;
lastCmdRight = rightSpeed;
```
Then replace `sendTelemetry()` with the five-field version:
Updated sendTelemetry()
```cpp
void sendTelemetry() {
  char buf[64];

  noInterrupts();
  long l = ticksLeft;
  long r = ticksRight;
  interrupts();

  snprintf(buf, sizeof(buf), "ENC,%ld,%ld,%d,%d",
           l, r, lastCmdLeft, lastCmdRight);

  udp.beginPacket(LAPTOP_IP, TELEM_PORT);
  udp.write((uint8_t*)buf, strlen(buf));
  udp.endPacket();
}
```
!!! tip "Commander vs Telemetry"
    `Commander` sends to the robot with `sendto()`. `Telemetry` listens with `bind()` and receives with `recvfrom()`. That is exactly the send/listen split from Exercise D.

---

## State Machine: How the Robot Thinks


---

In Chapter 5 you built a Q-learning agent that navigated FrozenLake. The agent was always in one state and took actions based on that state. Project 1's robot works the same way, except the policy is hand-designed instead of learned.
FrozenLake Project 1 Robot Tile number 0-15 STOPPED , SEARCHING , ACQUIRING , or TRACKING Action: left, right, up, down Action: left and right motor speeds Transition: moved to a new tile Transition: ball found, ball lost, or timeout Reward: +1 for goal No reward; behavior is rule-based and predictable
The learned FrozenLake policy handles uncertainty through training. The Project 1 state machine is hand-designed because early physical robot testing needs behavior that is predictable and debuggable.

### The Four Robot States
This code is in `shared.py`.
RobotState constantsshared.py
```python
class RobotState:
    STOPPED = "STOPPED"
    SEARCHING = "SEARCHING"
    ACQUIRING = "ACQUIRING"
    TRACKING = "TRACKING"
```
**STOPPED**
No reliable target is visible. Motor output is zero and both PIDs are reset.
**SEARCHING**
The target disappeared recently. The robot spins slowly in the last known target direction.
**ACQUIRING**
The target just appeared. Speed ramps up gradually while the detection proves it is stable.
**TRACKING**
The target has been confirmed for enough frames. The robot uses full PID authority.

### The Full `_update_state()` Method
This code is in `obj_track_adv.py`.
State transition logicobj_track_adv.py
```python
def _update_state(self, bbox_found):
    if bbox_found:
        self.last_seen_t = time.time()

        if self.state in (RobotState.STOPPED, RobotState.SEARCHING):
            self.state = RobotState.ACQUIRING
            self.acquire_count = 0
            self.speed_limit = 0.4

        elif self.state == RobotState.ACQUIRING:
            self.acquire_count += 1
            self.speed_limit = 0.4 + 0.6 * self.acquire_count / ACQUIRE_FRAMES
            if self.acquire_count >= ACQUIRE_FRAMES:
                self.state = RobotState.TRACKING
                self.speed_limit = 1.0

        elif self.state == RobotState.TRACKING:
            self.speed_limit = 1.0

    else:
        elapsed = time.time() - self.last_seen_t
        if elapsed < LOST_TIMEOUT:
            self.state = RobotState.SEARCHING
            self.speed_limit = 0.3
        else:
            self.state = RobotState.STOPPED
            self.speed_limit = 0.0
            self.smoothed_angle = 0.0
            self.pid_turn.reset()
            self.pid_dist.reset()
```

### Line-by-line Decisions
self.last_seen_t = time.time() This stores the timestamp of the most recent successful detection. When the ball disappears, the robot subtracts this time from the current time to decide whether it should search or stop. STOPPED or SEARCHING becomes ACQUIRING Both states mean the robot was not actively tracking. A new bounding box does not immediately earn full speed; it starts in ACQUIRING with speed_limit = 0.4 . self.acquire_count += 1 While the robot remains in ACQUIRING , each visible frame increments this counter. The count resets when the robot first enters ACQUIRING from STOPPED or SEARCHING . 0.4 + 0.6 * acquire_count / ACQUIRE_FRAMES This is a linear speed ramp from 40 percent to 100 percent. With ACQUIRE_FRAMES = 5 , the first visible acquiring frame uses 52 percent, the second uses 64 percent, and the fifth reaches 100 percent. elapsed < LOST_TIMEOUT A brief missing detection sends the robot to SEARCHING , not STOPPED . This handles one-frame YOLO misses and small camera glitches without killing the run. self.pid_turn.reset() and self.pid_dist.reset() The PIDs reset only when the target has been lost long enough to stop. They are not reset on every brief search, because that would erase useful accumulated correction and cause a lurch when the ball reappears.

### Searching Spin
This code is in `obj_track_adv.py`.
Spin toward the last known target directioncontrol()
```python
elif self.state == RobotState.SEARCHING:
    tgt_left = 10 * self.last_turn_dir * self.speed_limit
    tgt_right = -10 * self.last_turn_dir * self.speed_limit
```
`last_turn_dir` remembers which side the target was on before it disappeared. If the ball was last seen on the right, the robot searches right first because that is the most likely direction to reacquire it.

### Transitions
STOPPED -> ACQUIRING : ball appears Trigger: YOLO returns a bounding box in _detect() , so bbox_found is true while the robot is stopped or searching. Code pattern: if self.state in (RobotState.STOPPED, RobotState.SEARCHING) . Action: set speed_limit to 0.4 and reset acquire_count . ACQUIRING -> TRACKING : 5 consecutive frames Trigger: self.acquire_count >= ACQUIRE_FRAMES . A single YOLO false positive can happen for one frame; five consecutive frames is much stronger evidence. Action: set speed_limit to 1.0 so the PIDs can use full authority. TRACKING -> SEARCHING : ball disappears briefly Trigger: _detect() returns None , but time.time() - self.last_seen_t < LOST_TIMEOUT . Action: spin in last_turn_dir at reduced speed to reacquire the ball. SEARCHING -> STOPPED : timeout exceeded Trigger: the ball has been missing longer than LOST_TIMEOUT . Action: stop motors, clear smoothed angle, and reset both PIDs.
!!! info "Why reset PID on STOP?"
    If the robot tracked a ball moving left for two seconds, the integral term may contain a leftward correction. If a new ball appears on the right, stale integral would initially push the wrong way. Resetting on STOP makes every tracking session start fresh.

---

## The Tracker: Putting It Together


---

This lesson follows one frame from camera input to motor command. The goal is to see how all components interact in the order they run.

### The Full `control()` Method
This code is in `obj_track_adv.py`.
Tracker.control()obj_track_adv.py
```python
def control(self, frame):
    h, w = frame.shape[:2]
    bbox = self._detect(frame)
    self._update_state(bbox is not None)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        area = (x2 - x1) * (y2 - y1)
        tgt_left, tgt_right, dist_error, turn_cmd, base_speed = \
            self._motor_outputs_for_ball(cx, area, w)

    elif self.state == RobotState.SEARCHING:
        tgt_left = 10 * self.last_turn_dir * self.speed_limit
        tgt_right = -10 * self.last_turn_dir * self.speed_limit
        dist_error = turn_cmd = base_speed = 0.0
        area = 0
        cx = w / 2

    else:
        tgt_left = tgt_right = 0
        dist_error = turn_cmd = base_speed = 0.0
        area = 0
        cx = w / 2

    self.cur_left = ramp(self.cur_left, tgt_left, MAX_ACCEL)
    self.cur_right = ramp(self.cur_right, tgt_right, MAX_ACCEL)

    debug = {
        "state": self.state,
        "bbox": bbox,
        "cx": cx,
        "area": area,
        "dist_error": dist_error,
        "turn_cmd": turn_cmd,
        "base_speed": base_speed,
        "smoothed_angle": self.smoothed_angle,
        "speed_limit": self.speed_limit,
    }
    return round(self.cur_left), round(self.cur_right), debug
```

### Where Each Line Came From
Line Origin Explanation bbox = self._detect(frame) Exercise A Runs YOLO and returns the best bounding box, using the same xyxy box format you inspected earlier. self._update_state(...) P1.8 Converts detection presence into STOPPED , SEARCHING , ACQUIRING , or TRACKING . cx = (x1 + x2) / 2 Exercise A Finds the horizontal center of the detected object. This becomes the steering error. area = (x2 - x1) * (y2 - y1) Exercise A Estimates distance from bounding-box size. Larger area means the target is closer. _motor_outputs_for_ball(...) P1.4 Runs the turn and distance PID logic and returns target wheel speeds. ramp(...) Exercise C concept Limits how fast motor commands are allowed to change, like a continuous version of slowing near a target. debug = {...} P1.10 Packages internal control values for the HUD without recomputing them.

### The Ramp Function
This code is in `shared.py`.
Motor acceleration limiterControl helper
```python
def ramp(current, target, max_step):
    diff = target - current
    if abs(diff) <= max_step:
        return target
    return current + max_step * np.sign(diff)
```
With `MAX_ACCEL = 2`, a motor command can change by only 2 PWM units per control cycle. At about 33 Hz, going from 0 to 255 would take roughly `255 / 2 / 33 = 3.9` seconds, which is intentionally gentle for first testing.

### The Full `_motor_outputs_for_ball()` Method
This code is in `obj_track_adv.py`.
Converting one bounding box into two wheel speedsobj_track_adv.py
```python
def _motor_outputs_for_ball(self, cx, area, frame_width):
    raw_angle_error = (cx - frame_width / 2) / frame_width

    if abs(raw_angle_error) < ANGLE_DEAD_ZONE:
        raw_angle_error = 0.0
    if raw_angle_error != 0.0:
        self.last_turn_dir = np.sign(raw_angle_error)

    self.smoothed_angle = raw_angle_error
    turn_cmd = self.pid_turn.update(self.smoothed_angle)

    dist_error = (area - TARGET_AREA) / TARGET_AREA
    if abs(dist_error) < AREA_DEAD_ZONE:
        dist_error = 0.0
    self.pid_dist.update(dist_error)

    base_speed = 10
    tgt_left = base_speed + turn_cmd
    tgt_right = base_speed - turn_cmd

    max_raw = max(abs(tgt_left), abs(tgt_right))
    if max_raw > MAX_SPEED:
        scale = MAX_SPEED / max_raw
        tgt_left *= scale
        tgt_right *= scale

    return tgt_left, tgt_right, dist_error, turn_cmd, base_speed
```
raw_angle_error = (cx - frame_width / 2) / frame_width This converts pixel position into normalized error. On a 640 px frame, cx = 320 gives 0 , cx = 480 gives +0.25 , and cx = 160 gives -0.25 . ANGLE_DEAD_ZONE The dead zone prevents tiny left-right corrections caused by detection jitter. The default can be kept at zero while tuning, then raised slightly if the robot twitches when the target is centered. self.last_turn_dir = np.sign(raw_angle_error) This stores the last side where the ball was seen. The SEARCHING state uses it to spin in the likely direction. turn_cmd = self.pid_turn.update(...) The steering PID returns a correction between -MAX_TURN and +MAX_TURN . Positive values make the left motor faster and the right motor slower. dist_error = (area - TARGET_AREA) / TARGET_AREA This normalizes distance error. 0 means the target is at the calibrated distance, positive means too close, and negative means too far. tgt_left = base_speed + turn_cmd This is differential steering. Adding the turn command to one side and subtracting it from the other curves the robot while the base speed keeps it moving forward. scale = MAX_SPEED / max_raw If either side exceeds the speed limit, both sides are scaled down together. Scaling preserves the turn ratio better than clipping one side independently.

### Frame-by-frame Walkthrough
Step 1: Get a frame. frame, fid = video.read() returns the latest frame from MobileVideoStream 's background thread. If fid equals last_fid , no new frame has arrived, so the loop skips this iteration and avoids wasting YOLO inference on duplicate frames. Step 2: Run detection and control. left_speed, right_speed, debug = tracker.control(frame) runs _detect(frame) , advances the state machine, computes PID outputs if tracking, applies ramping from Exercise C's acceleration concept, and returns two wheel commands plus debug data. Step 3: Send motor command. At a fixed interval, the main loop sends commander.motors(left_speed, right_speed) . The command path is the Exercise D UDP sender, now wrapped in Commander . Step 4: Draw the HUD. draw_overlay(...) combines the tracker debug dictionary, motor outputs, and telemetry.read() into the on-screen display. Step 5: Show the frame. cv2.imshow(...) displays the annotated camera feed, just like Exercise A and Exercise B did.

### What `tracker.control(frame)` Contains
Inside control() Origin Purpose _detect(frame) Exercise A Runs YOLO and returns a bounding box or None . _update_state(bbox_found) P1.8 Moves between STOPPED, SEARCHING, ACQUIRING, and TRACKING. _motor_outputs_for_ball() P1.4 Runs angle and distance PID logic. ramp() Exercise C concept Limits acceleration so motor commands do not jump instantly. debug dictionary P1.10 Feeds the HUD with state, area, error, speed limit, and box data.
!!! warning "Motor sign can be chassis-specific"
    Line: `commander.motors(left_speed, right_speed)`. This version sends the left motor speed as a positive value, matching the current Project 1 robot wiring. On some chassis, motor orientation may still be reversed.
    Test with Exercise D's keyboard script by sending `MOTOR,100,100`. If the robot spins instead of moving forward, change the send line to `commander.motors(-left_speed, right_speed)` or swap the side that is negated according to your wiring.

### Why the debug dictionary matters
The tracker does not just return motor speeds. It also returns state, area, smoothed angle, distance error, speed limit, and bounding box data. This keeps the control logic and HUD connected without making the HUD rerun detection or recompute control decisions.

---

## Visualisation and the HUD


---

The HUD has no new robotics concepts. It is a live dashboard that draws each piece of internal data onto the camera frame so you can debug without staring at terminal logs.

### The Full `draw_overlay()` Function
This code is in `obj_track_adv.py`.
HUD drawing functionobj_track_adv.py
```python
def draw_overlay(frame, debug, left_speed, right_speed, telem, scale_x, scale_y):
    h, w = frame.shape[:2]
    state_color = Tracker.STATE_COLORS.get(debug["state"], (255, 255, 255))

    cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)

    smooth_x = int(w / 2 + debug["smoothed_angle"] * w)
    cv2.line(frame, (smooth_x, 0), (smooth_x, h), (0, 255, 255), 2)

    if debug["bbox"] is not None:
        x1, y1, x2, y2 = debug["bbox"]
        x1, y1, x2, y2 = (
            int(x1 * scale_x),
            int(y1 * scale_y),
            int(x2 * scale_x),
            int(y2 * scale_y),
        )

        if abs(debug["dist_error"]) < AREA_DEAD_ZONE:
            box_color = (0, 255, 0)
        elif debug["dist_error"] > 0:
            box_color = (0, 0, 255)
        else:
            box_color = (255, 100, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    lines = [
        (f"STATE:  {debug['state']}", state_color),
        (f"SMOOTH: {debug['smoothed_angle']:+.3f}", (0, 255, 255)),
        (f"AREA:   {int(debug['area'])} / {TARGET_AREA}", (255, 255, 255)),
        (f"SPD LIM:{debug['speed_limit']:.0%}", (255, 255, 255)),
        (f"L:{int(left_speed):+4d} R:{int(right_speed):+4d}", (0, 255, 0)),
        (
            f"ENC L:{telem['left_ticks']:+4d} R:{telem['right_ticks']:+4d}",
            (255, 165, 0),
        ),
    ]
    for i, (text, color) in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (10, 25 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            1,
        )
```

### Visual Elements and Data Sources
HUD element Data source Meaning Bounding box color: green abs(dist_error) < AREA_DEAD_ZONE The ball is at the calibrated distance. Bounding box color: red dist_error > 0 The ball is too close; the box is larger than TARGET_AREA . Bounding box color: blue dist_error < 0 The ball is too far; the box is smaller than TARGET_AREA . State text color STATE_COLORS Matches the robot LED colors: green for tracking, orange for searching, yellow for acquiring. Cyan smooth line smoothed_angle Shows filtered horizontal error rather than raw detection jitter. Motor command text left_speed , right_speed Shows what Python is asking the robot to do. Encoder text telemetry.read() Shows what the robot reports back over UDP.

### How to Read the HUD While Testing
STATE This is the robot's current behavior mode. If it stays STOPPED while the target is visible, detection failed or TARGET_OBJECT does not match the model class name. SMOOTH This is the normalized horizontal error after smoothing. It should move toward 0.000 as the target becomes centered. If it never approaches zero, increase or retune the turn PID. AREA This compares the current bounding-box area against TARGET_AREA . Current greater than target means too close, current less than target means too far, and close to equal means correct distance. SPD LIM This is the state-machine speed multiplier: 0% for stopped, about 30% while searching, 40% to 100% while acquiring, and 100% while tracking. L and R These are the motor commands Python is sending. Opposite signs usually mean turning or searching. Similar signs usually mean driving while steering. ENC These are encoder ticks reported by the Arduino. If motor commands are nonzero but encoder ticks stay at zero, check encoder wiring and interrupt pins. scale_x and scale_y The display frame is resized to a fixed size. These scale factors convert bounding-box coordinates from the original camera frame into the resized display frame so the rectangle lands in the correct place.

### Why the smooth line matters
YOLO detections can flicker slightly from frame to frame even when the ball is still. The cyan line uses `smoothed_angle`, not raw angle error, so it tracks the trend without twitching on every small detection change. This is one reason the robot looks calmer than the raw camera measurements.
!!! info "HUD and hardware feedback match"
    The state text colors match the RGB LED states from P1.6. When you cannot see the laptop screen, the robot LED gives the same high-level state feedback.

---

## The Main Loop


---

The `main()` function is the integration point. Every line traces back to a project you already completed.

### The Complete `main()` Function
This code is in `obj_track_adv.py`.
Project 1 main loopobj_track_adv.py
```python
def main():
    global SMOOTH_RATE

    mobile_ip = "10.18.204.215"
    stream_url = f"http://{mobile_ip}:8080/video"

    video = MobileVideoStream(stream_url)
    commander = Commander(ESP_IP, UDP_CMD_PORT)
    telemetry = Telemetry(UDP_TELEM_PORT)

    print("Waiting for camera", end="", flush=True)
    for _ in range(30):
        if video.connected:
            break
        time.sleep(0.5)
        print(".", end="", flush=True)
    print()

    if not video.connected:
        print("Could not connect to camera")
        return

    tracker = Tracker("best.pt", TARGET_OBJECT)
    commander.stop()
    time.sleep(0.5)

    last_fid = -1
    last_cmd_time = 0.0
    CMD_INTERVAL = 0.03
    fps_count = 0
    fps_timer = time.time()

    while True:
        frame, fid = video.read()
        if frame is None or fid == last_fid:
            time.sleep(0.005)
            continue

        last_fid = fid
        fps_count += 1

        if fps_count >= 30:
            fps = fps_count / (time.time() - fps_timer)
            print(f"FPS: {fps:.1f}")
            fps_count = 0
            fps_timer = time.time()

        left_speed, right_speed, debug = tracker.control(frame)

        orig_h, orig_w = frame.shape[:2]
        frame = cv2.resize(frame, (640, 480))

        now = time.time()
        if now - last_cmd_time >= CMD_INTERVAL:
            commander.motors(left_speed, right_speed, debug["state"])
            last_cmd_time = now

        draw_overlay(
            frame,
            debug,
            left_speed,
            right_speed,
            telemetry.read(),
            640 / orig_w,
            480 / orig_h,
        )
        cv2.imshow("Object Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            tracker.reset()

    commander.stop()
    time.sleep(0.2)
    video.stop()
    telemetry.stop()
    cv2.destroyAllWindows()
```
Line in main() Origin video = MobileVideoStream(url) Exercise B stream concept. commander = Commander(ESP_IP, ...) Exercise D command socket concept. telemetry = Telemetry(TELEM_PORT) Exercise D telemetry socket concept. tracker = Tracker("best.pt", ...) Exercise A YOLO model loading. commander.stop() Exercise D send_stop() function. frame, fid = video.read() Exercise B frame-read concept, now threaded. tracker.control(frame) Exercise A detection plus Project 1 PID and state machine. commander.motors(l, r) Exercise D send_command(l, r) . draw_overlay(...) Project 1 HUD drawing. cv2.imshow(...) Exercise A and B display pattern.

### Non-obvious Lines
Waiting for the camera The for _ in range(30) loop checks for a connection every 0.5 seconds for up to 15 seconds. Phone camera apps often need a few seconds before the stream is ready. commander.stop() before the loop This sends a fresh stop packet before tracking begins. If a previous run crashed while motors were moving, startup still begins from a known safe command state. CMD_INTERVAL = 0.03 This limits outgoing motor commands to about 33 Hz. Sending faster than that does not improve robot control and can flood the Arduino UDP receive buffer. if frame is None or fid == last_fid This skips empty reads and duplicate frames. The loop does not waste YOLO inference on the same image twice. commander.motors(left_speed, right_speed, debug["state"]) The current Project 1 code sends the left motor speed as a positive value. Verify motor direction with the Exercise D keyboard test. The third argument sends the current state to the Arduino LED code from P1.6. telemetry.read() inside draw_overlay() Project 1 displays encoder ticks but does not yet use them for closed-loop wheel control. The telemetry data is for HUD diagnosis in this project. Shutdown order The program sends commander.stop() before stopping video and telemetry threads. Motor power is cut before communication objects are torn down.

### Decoupled Timing
The camera may run at 10 to 20 fps. The command sender runs up to about 33 Hz using `CMD_INTERVAL = 0.03`. These are deliberately decoupled. The motor command rate stays stable even if the phone camera stream is slower or briefly repeats a frame.
!!! info "Why duplicate frames are skipped"
    If `fid == last_fid`, the detection loop is running faster than the camera. Skipping that iteration prevents duplicate YOLO inference on the same image and keeps CPU time available for fresh frames.

### Shutdown Order
The program sends `commander.stop()` before stopping video and telemetry threads. Motor power is cut before the communication objects are torn down, which is the correct order for physical hardware.

---

## Running and Testing


---

### Pre-run Hardware Checklist
Arduino is running the updated robot_udp sketch from P1.6 and P1.7. Arduino Serial Monitor shows its IP address and Ready. Battery is connected and the shield jumper is set to BATT PWR . Phone stream is active and verified in a browser. Robot has at least 2 meters of clear floor space in all directions. Target object is nearby and visible to the phone camera.

### First Run Procedure
Stage 1: Detection only. Temporarily set MAX_SPEED = 0 . Run python obj_track_adv.py . Hold the ball in front of the phone camera. Verify the state changes from STOPPED to ACQUIRING to TRACKING and the box turns green near the target distance. Stage 2: Turn command check. Keep MAX_SPEED = 0 and watch motor values in the HUD. Move the ball left and right. Left and right command values should become nonzero and oppose each other when the ball is off-center. Stage 3: First live run. Restore MAX_SPEED = 20 for a slow first test. Place the ball about 60 cm in front of the phone. The robot should slowly orient toward and approach it. Stage 4: Tune and increase speed. Once basic tracking works, raise MAX_SPEED to 25, then 30. Use the P1.4 tuning guide if behavior is oscillatory.

### Common Failures and Fixes
Robot spins in circles constantly Likely cause: motor sign wrong or bad target area Run the Exercise D keyboard test and send MOTOR,100,100 . If the robot spins, add or remove the negative sign in the commander.motors() call. Also recalibrate TARGET_AREA using P1.3. STATE stays STOPPED, no detection Likely cause: wrong model or target string Print model.names in Tracker initialization and confirm TARGET_OBJECT exactly matches the model class name. STATE is TRACKING but motors do not move Likely cause: wrong ESP_IP or WiFi path Ping the Arduino IP from the laptop. Then rerun the Exercise D keyboard test to confirm the UDP command path works independently. Robot approaches then drives past the ball Likely cause: distance damping too low Increase DIST_KD and confirm MAX_ACCEL is still 2 . Robot oscillates left-right Likely cause: turn gain too high Reduce BODY_TURN_KP by about 30 percent, then add BODY_TURN_KD to dampen overshoot. Robot loses ball immediately during turns Likely cause: timeout or acquisition too strict Increase LOST_TIMEOUT to 1.5 or reduce ACQUIRE_FRAMES to 3 .

### What Success Looks Like
Place the ball on the floor while the robot faces a different direction. Run the script. Within about two seconds, the robot turns toward the ball, moves toward it, and settles at the calibrated distance. Move the ball a meter to the side and the robot pivots and approaches again. Pick up the ball and the robot searches briefly, then stops after `LOST_TIMEOUT`.
!!! tip "Closed-loop vision robot"
    You have combined a neural network you trained, streaming video from your phone, PID controllers, UDP commands at robotics speed, encoder telemetry, and a state machine into one working physical system.

[Download](original/shared.py)

[Download](original/obj_track_adv.py)



#  Project 2: Lane Follower

---

## What Is Classical Computer Vision?


---

In Project 1 you used YOLO, a neural network, to find an object in an image. YOLO learned from thousands of labeled examples. Project 2 uses classical computer vision instead: direct mathematical operations on pixels.
Classical computer vision does not learn. It applies rules such as grayscale conversion, blurring, thresholding, morphology, and row scanning. For a controlled problem like following tape on the floor, those rules are faster, smaller, and easier to debug than a trained model.
The robot follows black tape on the floor using only pixel math : no trained model, no GPU required.
Approach YOLO in Project 1 OpenCV classical vision in Project 2 Input Any image with the trained object visible. A controlled image where the line contrasts with the floor. What it detects Objects it was trained to recognize. Specific pixel patterns: dark line, light floor, or the inverse. Speed Usually 15-30 fps on CPU. Often 60-120 fps on CPU. Training Required. Not required. Data collection Required: many labeled images. Not required. Common failure Object looks different from training data. Lighting or floor color changes too much. Files needed A model file such as best.pt . No model file, only code.
!!! tip "Industrial robotics uses both"
    Factory robots often use classical vision for measurement, barcode reading, scratch detection, and line following because the environment is controlled. Neural networks are used when the visual world is messy. Your floor track is controlled, so classical vision is the right engineering choice.

<video controls width="100%"><source src="../original/lane_follower.mp4" type="video/mp4"></video>


### By the End of This Project
You will place black tape on a light floor to make a high-contrast track. Your phone camera will watch the floor from a forward-down angle. Python will process each camera frame in milliseconds using OpenCV. The robot will follow the tape through turns without any trained model. You will understand every line that makes that behavior happen.

### The Big Idea
Phone camera floor image → OpenCV pipeline threshold + scan rows → LaneFollower two-point steering + PID → Commander UDP motor commands
Everything after the vision result is familiar. The robot still receives motor commands over UDP and sends telemetry back. Only the project-specific vision and control code changes.

### The entire difference between Project 1 and Project 2 is what comes after the import line.
**`obj_track_adv.py`**
Project 1 structureSource file
```python
from shared import ...
# identical line

class Tracker:
    def _detect(self, frame):  # <- YOLO
        ...

def draw_overlay(...):
    ...
```
**`lane_follower_adv.py`**
Project 2 structureSource file
```python
from shared import ...
# identical line

class PD:
    ...

def detect_lane(frame):  # <- OpenCV
    ...

class LaneFollower:
    ...

def draw_debug(...):
    ...
```
`shared.py` is unchanged. The vision module swapped. Everything else : the camera, the UDP, the state machine, the PID, the main loop structure : is identical.

### Full Code
This is the complete file for reference. Do not try to read it all now : it will not make sense yet. As you work through each lesson, come back here to find the exact section being explained. By the end of the project every line will be familiar.
Complete lane_follower_adv.pyCopy or download
```python
from shared import PID, MobileVideoStream, Commander, Telemetry, RobotState, ramp
import cv2
import numpy as np
import time

# ==================== CONFIGURATION ====================
ESP_IP         = "YOUR_ESP_IP"        # e.g. "192.168.1.100"
MOBILE_IP      = "YOUR_PHONE_IP"      # e.g. "192.168.1.101"
UDP_CMD_PORT   = 5001
UDP_TELEM_PORT = 5002

BINARY_THRESHOLD   = 70
ROI_HEIGHT_PERCENT = 0.88
MORPH_KERNEL_SIZE  = 7
MIN_LINE_AREA      = 500
BASE_SPEED         = 100
TARGET_AREA        = 20000
MAX_SPEED          = 255
MAX_TURN           = 30
MAX_ACCEL          = 2
MOTOR_OUTPUT_SCALE = 10

SMOOTH_RATE = 0.04

BODY_TURN_KP = 100
BODY_TURN_KD = 4
BODY_TURN_KI = 0

DIST_KP = 0
DIST_KI = 0
DIST_KD = 0
MAX_DIST_INTEGRAL = 0.8

ANGLE_DEAD_ZONE = 0.00
AREA_DEAD_ZONE  = 0.2

ACQUIRE_FRAMES = 5
LOST_TIMEOUT   = 1.0

ROW_FRACS = [0.90, 0.75, 0.60, 0.45, 0.30]

STATE_COLORS = {
    "STOPPED":   (100, 100, 100),
    "SEARCHING": (255, 165,   0),
    "ACQUIRING": (255, 255,   0),
    "TRACKING":  (  0, 255,   0),
}


# ==================== CONTROLLERS ====================
class PD:
    def __init__(self, kp, kd, output_limits=None):
        self.kp = kp
        self.kd = kd
        self.output_limits = output_limits
        self._prev_error = 0.0
        self._prev_time  = None

    def update(self, error):
        now = time.time()
        dt  = 0.02 if self._prev_time is None else max(now - self._prev_time, 1e-4)
        self._prev_time  = now
        derivative       = (error - self._prev_error) / dt
        self._prev_error = error
        out = self.kp * error + self.kd * derivative
        if self.output_limits:
            out = np.clip(out, *self.output_limits)
        return out

    def reset(self):
        self._prev_error = 0.0
        self._prev_time  = None


# ==================== HELPERS ====================

def scale_motors(left, right):
    """Scale motor commands so neither exceeds MAX_SPEED."""
    peak = max(abs(left), abs(right))
    if peak > MAX_SPEED:
        scale = MAX_SPEED / peak
        return left * scale, right * scale
    return left, right


# ==================== LANE DETECTION ====================
def detect_lane(frame):
    h, w  = frame.shape[:2]
    roi_y = int(h * (1 - ROI_HEIGHT_PERCENT))
    roi   = frame[roi_y:, :]

    blur = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    _, binary = cv2.threshold(blur, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    points = []
    for frac in ROW_FRACS:
        ry  = int(binary.shape[0] * frac)
        xs  = np.where(binary[ry] > 0)[0]
        if len(xs) < 15:
            continue
        runs = np.split(xs, np.where(np.diff(xs) > 1)[0] + 1)
        best = max(runs, key=len)
        if len(best) < 12:
            continue
        points.append((int((best[0] + best[-1]) / 2), ry + roi_y))

    area = int(np.count_nonzero(binary))

    if len(points) < 2:
        return None, None, None, area, roi_y, binary, points

    return points[0], points[-1], points, area, roi_y, binary, points

# ==================== LANE FOLLOWER ====================
class LaneFollower:
    def __init__(self):
        self.smoothed_angle = 0.0
        self.pid_turn = PID(BODY_TURN_KP, BODY_TURN_KI, BODY_TURN_KD,
                            output_limits=(-MAX_TURN, MAX_TURN))
        self.pid_dist = PID(DIST_KP, DIST_KI, DIST_KD,
                            max_integral=MAX_DIST_INTEGRAL,
                            output_limits=(-MAX_SPEED, MAX_SPEED))
        self.state         = RobotState.STOPPED
        self.acquire_count = 0
        self.last_seen_t   = 0.0
        self.last_turn_dir = 1.0
        self.cur_left      = 0.0
        self.cur_right     = 0.0
        self.speed_limit   = 1.0

    def _update_state(self, line_found):
        if line_found:
            self.last_seen_t = time.time()
            if self.state in (RobotState.STOPPED, RobotState.SEARCHING):
                self.state, self.acquire_count, self.speed_limit = RobotState.ACQUIRING, 0, 0.4
            elif self.state == RobotState.ACQUIRING:
                self.acquire_count += 1
                self.speed_limit = 0.4 + 0.6 * self.acquire_count / ACQUIRE_FRAMES
                if self.acquire_count >= ACQUIRE_FRAMES:
                    self.state, self.speed_limit = RobotState.TRACKING, 1.0
            # TRACKING → no change needed
        else:
            elapsed = time.time() - self.last_seen_t
            if elapsed < LOST_TIMEOUT:
                self.state, self.speed_limit = RobotState.SEARCHING, 0.3
            else:
                self.state, self.speed_limit = RobotState.STOPPED, 0.0
                self.smoothed_angle = 0.0
                self.pid_turn.reset()
                self.pid_dist.reset()

    def control(self, frame):
        h, w = frame.shape[:2]
        near_pt, far_pt, centers, area, roi_y, binary, points = detect_lane(frame)
        line_found = near_pt is not None and far_pt is not None

        self._update_state(line_found)

        if line_found:
            near_cx, near_cy = near_pt
            far_cx,  _       = far_pt

            position_error  = (near_cx - w / 2) / w
            direction_error = (far_cx  - near_cx) / w
            raw_angle_error = 0.25 * position_error + 0.75 * direction_error

            if abs(raw_angle_error) < ANGLE_DEAD_ZONE:
                raw_angle_error = 0.0
            if raw_angle_error != 0.0:
                self.last_turn_dir = np.sign(raw_angle_error)

            self.smoothed_angle = (
                SMOOTH_RATE * self.smoothed_angle
                + (1 - SMOOTH_RATE) * raw_angle_error
            )

            turn_cmd   = self.pid_turn.update(self.smoothed_angle)
            dist_error = (TARGET_AREA - area) / TARGET_AREA
            if abs(dist_error) < AREA_DEAD_ZONE:
                dist_error = 0.0
            dist_cmd = self.pid_dist.update(dist_error)

            tgt_left, tgt_right = scale_motors(
                (BASE_SPEED + turn_cmd) * self.speed_limit,
                (BASE_SPEED - turn_cmd) * self.speed_limit,
            )
            cx, cy = near_cx, near_cy

        elif self.state == RobotState.SEARCHING:
            search_turn = 8 * self.last_turn_dir * self.speed_limit
            forward     = 5 * self.speed_limit
            tgt_left, tgt_right = scale_motors(forward + search_turn, forward - search_turn)
            dist_error, turn_cmd, area, cx = 0.0, 0.0, 0, w // 2
            cy = h // 2

        else:  # STOPPED
            tgt_left = tgt_right = 0
            dist_error, turn_cmd, area, cx = 0.0, 0.0, 0, w // 2
            cy = h // 2

        self.cur_left  = ramp(self.cur_left,  tgt_left,  MAX_ACCEL)
        self.cur_right = ramp(self.cur_right, tgt_right, MAX_ACCEL)

        debug = {
            'state':          self.state,
            'line_found':     line_found,
            'cx':             cx,
            'cy':             cy,
            'area':           area,
            'roi_y':          roi_y,
            'binary':         binary,
            'smoothed_angle': self.smoothed_angle,
            'turn_cmd':       turn_cmd,
            'dist_error':     dist_error,
            'base_speed':     BASE_SPEED if line_found else 0,
            'speed_limit':    self.speed_limit,
            'points':         points if line_found else [],
        }
        return round(self.cur_left), round(self.cur_right), debug


# ==================== VISUALIZATION ====================
def draw_debug(frame, debug, left_speed, right_speed, telem):
    h, w = frame.shape[:2]

    # ROI line
    cv2.line(frame, (0, debug['roi_y']), (w, debug['roi_y']), (0, 165, 255), 1)
    cv2.putText(frame, "ROI", (5, debug['roi_y'] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

    # Centre and smoothed-angle guide lines
    cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
    smooth_x = int(w / 2 + debug['smoothed_angle'] * w)
    cv2.line(frame, (smooth_x, 0), (smooth_x, h), (0, 255, 255), 2)
    cv2.putText(frame, "SMOOTH", (smooth_x + 5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Detected lane points and centroid
    if debug['line_found']:
        color = ((0, 255, 0) if abs(debug['dist_error']) < AREA_DEAD_ZONE
                 else (0, 0, 255) if debug['dist_error'] < 0
                 else (255, 100, 0))
        cv2.circle(frame, (debug['cx'], debug['cy']), 8, color, -1)
        cv2.line(frame, (w // 2, debug['cy']), (debug['cx'], debug['cy']), (0, 255, 255), 2)

        for px, py in debug['points']:
            cv2.circle(frame, (px, py), 5, (255, 0, 255), -1)
        if len(debug['points']) >= 2:
            cv2.line(frame, debug['points'][0], debug['points'][-1], (255, 0, 0), 2)

    # HUD text
    sc = STATE_COLORS.get(debug['state'], (255, 255, 255))
    lines = [
        (f"STATE:  {debug['state']}",                       sc,            25),
        (f"SMOOTH: {debug['smoothed_angle']:+.3f}",          (0, 255, 255), 50),
        (f"AREA:   {int(debug['area'])} / {TARGET_AREA}",   (255,255,255), 75),
        (f"SPD LIM:{debug['speed_limit']:.0%}",             (255,255,255), 100),
        (f"L:{int(left_speed):+4d} R:{int(right_speed):+4d}", (0,255,0),   125),
        (f"DIST:   {debug['dist_error']:+.3f}",             (255, 165, 0), 150),
        (f"ENC L:{telem['left_ticks']:+4d} R:{telem['right_ticks']:+4d}", (255,165,0), 175),
        (f"THRESH: {BINARY_THRESHOLD}",                     (200,200,200), 200),
    ]
    for text, color, y in lines:
        thickness = 2 if y in (25, 125) else 1
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55 if thickness == 1 else 0.65, color, thickness)

    # Binary window
    if debug['binary'] is not None:
        bin_disp = cv2.resize(cv2.cvtColor(debug['binary'], cv2.COLOR_GRAY2BGR), (640, 480))
        cv2.putText(bin_disp, f"THRESH={BINARY_THRESHOLD}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.imshow("Binary (lane detection)", bin_disp)


# ==================== MAIN ====================
def main():
    global BINARY_THRESHOLD

    stream_url = f"http://{MOBILE_IP}:8080/video"
    print(f"Stream: {stream_url}")

    video     = MobileVideoStream(stream_url)
    commander = Commander(ESP_IP, UDP_CMD_PORT)
    telemetry = Telemetry(UDP_TELEM_PORT)

    print("Waiting for camera...")
    for _ in range(30):
        if video.connected:
            break
        time.sleep(0.5)
        print(".", end="", flush=True)
    print()

    if not video.connected:
        print("Could not connect to camera")
        exit(1)

    follower = LaneFollower()
    commander.stop()
    time.sleep(0.5)

    print("\nLane Following Active")
    print(
        f"BINARY_THRESHOLD: {BINARY_THRESHOLD}  "
        f"ROI: {ROI_HEIGHT_PERCENT:.0%}  "
        f"KERNEL: {MORPH_KERNEL_SIZE}x{MORPH_KERNEL_SIZE}  "
        f"TARGET_AREA: {TARGET_AREA}"
    )
    print("Q=Quit  R=Reset  T=Threshold up  Y=Threshold down")
    print("=" * 60 + "\n")

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
            print(f"FPS: {fps_count / (time.time() - fps_timer):.1f}")
            fps_count = 0
            fps_timer = time.time()

        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        left_speed, right_speed, debug = follower.control(frame)
        print(left_speed, right_speed)

        now = time.time()
        if now - last_cmd_time >= CMD_INTERVAL:
            commander.motors(0, 0) if (left_speed == right_speed == 0) \
                else commander.motors(
                    -left_speed / MOTOR_OUTPUT_SCALE,
                    right_speed / MOTOR_OUTPUT_SCALE,
                )
            last_cmd_time = now

        draw_debug(frame, debug, left_speed, right_speed, telemetry.read())
        cv2.imshow("Lane Follower", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            follower.pid_turn.reset()
            follower.pid_dist.reset()
            follower.smoothed_angle = 0.0
            print("Reset")
        elif key == ord('t'):
            BINARY_THRESHOLD = min(BINARY_THRESHOLD + 5, 255)
            print(f"BINARY_THRESHOLD -> {BINARY_THRESHOLD}")
        elif key == ord('y'):
            BINARY_THRESHOLD = max(BINARY_THRESHOLD - 5, 0)
            print(f"BINARY_THRESHOLD -> {BINARY_THRESHOLD}")

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

The code for this project is in the course GitHub repository. You need the shared infrastructure file and the Project 2 lane follower file before continuing.
Step 1 — Open the repository. Go to github.com/purwar-lab/ml-for-robotics- . Step 2 — Download the files below. Save them to your my-detector project folder from Exercise A. Step 3 — Verify the files are there. Open VS Code and open the my-detector folder. You should see shared.py and lane_follower_adv.py in the file explorer on the left. Step 4 — Open the project file and read the configuration block. Open lane_follower_adv.py . The very top section after the import lines is the CONFIGURATION block. Every value in this section must be filled in before running. The configuration lesson walks through each one.

### Download Files
**`shared.py`**
Reusable infrastructure: PID controller, camera stream, UDP commands, telemetry, and state machine constants. Used by both Project 1 and Project 2.
Already have this from Project 1? You do not need to download it again. Skip to `lane_follower_adv.py`.
**`lane_follower_adv.py`**
Project 2 specific code: detect_lane(), LaneFollower class, binary thresholding pipeline, two-point steering, and the main loop.
!!! info "Do not run the files yet"
    They will not work with the default placeholder values. Complete every lesson in this project first, then run the project file in the final lesson.

[shared.py](original/shared.py)

[lane_follower_adv.py](original/lane_follower_adv.py)


---

## Physical Setup: Your Track and Camera


---

Project 2 depends on controlled visual conditions. Before running a single experiment, set up the tape, floor, track shape, and phone camera so the algorithm has something reliable to see.

### The Tape
**Use dark tape**
Black electrical tape or black gaffer tape works well. A width near `1.5 cm` is ideal; wider is fine, narrower is harder to detect.
**Use a light floor**
Light grey, beige, white, or cream gives the detector strong contrast. The default code looks for dark pixels on a light background.
**Reverse if needed**
On a dark floor, use white tape and change `cv2.THRESH_BINARY_INV` to `cv2.THRESH_BINARY`.
!!! info "Why black tape specifically?"
    The detector converts the frame to grayscale and looks for pixels below a brightness threshold. Black tape on a light floor creates a strong brightness difference, which makes detection reliable even when lighting changes slightly. A colored line can have less grayscale contrast than expected because color is discarded.

### The Track Shape
Lay the tape in a closed loop. The minimum useful size is roughly `80 cm x 120 cm`. Start with a gentle oval, then try sharper corners after the robot follows the oval consistently.
Robot direction Tape width: 1.5 cm Keep 10 cm clear on both sides 120 cm 80 cm
!!! warning "Leave space around the tape"
    If the tape is close to a wall, furniture edge, or robot body shadow, the algorithm may detect that edge instead of the tape.

### Phone Mounting and Camera Angle
Mount the phone pointing forward and downward. It should see the floor about `15 to 40 cm` ahead of the robot, with the tape in the lower two-thirds of the frame.
50-60 degree forward-down view Tape visible ahead of robot
Forward-facing matters because the robot needs preview. A straight-down camera only sees the floor under the chassis; by the time a curve appears, the robot is already on it.
lane_follower_adv.py: frame rotation in main loop
Phone stream rotation
```python
frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
```
Phone camera streams may arrive sideways when the phone is physically rotated. If the HUD appears sideways, this line is probably fixing it. If the image is upside down instead, change the rotation to `cv2.ROTATE_180`.
!!! warning "Test camera position before code"
    Open the IP Webcam stream in your browser, place the robot on the track, and verify that the tape is visible in the lower portion of the image. No code change can fix a camera that cannot see the tape.

### Setup Checklist
Black tape is on a light floor, or white tape is on a dark floor with threshold mode noted. Tape width is close to 1.5 cm and the track is at least 80 cm x 120 cm . The track has at least 10 cm of clear space on both sides of the tape. Phone is mounted forward-downward and sees tape in the lower two-thirds of the browser stream. Project 1 phone stream, UDP commands, telemetry, state machine, and PID loop already work.

---

## The OpenCV Experiment Notebook


---

Before touching the robot code, build the lane detector step by step on one still image in Colab. You should see the math work on a photo before asking a moving robot to trust it.
[Open in Colab →](https://colab.research.google.com/github/purwar-lab/ml-for-robotics-/blob/main/notebooks/proj2-opencv-experiments.ipynb)
!!! warning "How to take the right test photo"
    The row scanning algorithm expects the tape to look like it does from the robot's camera: a stripe running away from you into the distance, narrow at the top and slightly wider at the bottom.
    **CORRECT photo:** stand behind the robot position and hold your phone at roughly a 50-60 degree angle pointing forward-down at the tape. The tape should appear as a vertical stripe running from the bottom of the frame toward the top, like a road disappearing into the distance.
    **WRONG photo:** holding the phone straight down above the tape. This produces a horizontal blob in the center of the frame. Row scanning will miss it because it scans horizontal rows looking for a vertical stripe, and because the blob sits in the middle rows instead of the lower rows where the scan lines are.
    If you only have a horizontal photo right now, rotate it in the notebook to simulate the correct perspective. Add this line immediately after `cv2.imread`:
    This is the same rotation the robot code applies in `main()`. Run it and the horizontal tape becomes a vertical stripe.

### Cell 1b: Quick Photo Sanity Check
Run this immediately after loading the image. It prints a simple row histogram so you can see whether the tape is actually in the lower 60% of the frame before row scanning begins.
Where is the tape in your image?
```python
# Quick check — where is the tape in your image?
gray_check = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
_, binary_check = cv2.threshold(gray_check, BINARY_THRESHOLD,
                                 255, cv2.THRESH_BINARY_INV)

print("Rows containing white pixels (potential tape rows):")
for ry in range(0, frame.shape[0], int(frame.shape[0]/20)):
    count = np.count_nonzero(binary_check[ry])
    bar = "█" * min(count // 10, 40)
    print(f"  {ry/frame.shape[0]:.0%} from top: {bar} ({count}px)")

print()
print("The tape should appear as a vertical band of rows")
print("concentrated in the LOWER 60% of the image (60-100% from top)")
print("If it appears in the middle rows only, rotate your photo first.")
```

### How This Lesson Works
Take one photo of your tape on the floor from roughly the robot camera angle. Upload it to Colab using the Files panel and name it test_frame.jpg . Run each notebook cell in order and write down the threshold and kernel values that work. Paste those values into the configuration block in lane_follower_adv.py .
!!! info "Experiment first, robot second"
    The notebook demonstrates binary thresholding, morphological cleaning, row scanning, and two-point steering before the physical robot moves. If the still image pipeline is wrong, the live robot will not fix it.

### Notebook Cells
Cell What it shows What to write down 1. Load image Reads test_frame.jpg , displays it, and explains BGR versus RGB. Confirm the photo loaded and tape is visible. 1b. Photo sanity check Prints which image rows contain potential tape pixels before the main pipeline runs. Confirm the tape appears in the lower 60% of the frame, or rotate the photo first. 2. Grayscale Shows that the detector uses brightness, not color. Confirm the tape still contrasts in grayscale. 3. Blur Compares raw grayscale to a 7x7 Gaussian blur. Notice whether blur removes small specks. 4. Threshold Interactive BINARY_THRESHOLD tuning. Your working threshold value. 5. Morphology Shows OPEN for noise removal and CLOSE for gap filling. Your working kernel size. 6. Row scanning Draws scan rows and magenta center points. Typical number of detected points. 7. Steering error Computes the same combined error the PID receives in the robot script. Whether left/right steering signs make sense.

### Cell 6: Row Scanning Warning
!!! warning "Your test photo matters"
    The scan rows in this notebook are set up for a photo where the tape appears in the lower half of the frame, as it would when the phone is mounted on the robot pointing forward-downward. If your test photo was taken straight down or the tape is in the center of the image, the scan rows will miss it. Take your test photo from the same angle the phone will be mounted on the robot: forward-facing, angled down so the tape appears in the bottom 60% of the frame. If the dots still do not appear, add print statements to check which rows the tape actually occupies, then adjust `ROW_FRACS` to match.
There is no ROI cropping in the Colab notebook. Cell 6 processes the full image, so `ROW_FRACS = [0.90, 0.75, 0.60, 0.45, 0.30]` scans rows in the lower part of that full frame. If your tape is in the upper or middle half, those rows can miss it entirely.
Find which rows contain white pixels
```python
# Find which rows contain white pixels
for ry in range(0, h, 20):   # check every 20 rows
    count = np.count_nonzero(clean[ry])
    if count > 0:
        print(f"Row {ry} ({ry/h:.0%} from top): {count} white pixels")
```

### The Most Important Cell
Binary threshold experiment
```python
BINARY_THRESHOLD = 70  # Change this value and run again.

_, binary = cv2.threshold(blur, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
white_pixels = np.count_nonzero(binary)
total_pixels = binary.size

print(f"White pixels: {white_pixels} ({100 * white_pixels / total_pixels:.1f}% of image)")
```
Change `BINARY_THRESHOLD` and rerun until only the tape stripe is white. This is the correct time to find the value: on a still photo, not while the robot is moving.

### Values to Bring Back
I wrote down BINARY_THRESHOLD = _____ from Cell 4. I wrote down the kernel size that cleaned the image without damaging the tape. I confirmed Cell 6 finds magenta points on the tape. I ran Cell 7 and saw the steering error sign for my photo.

---

## Configuration: Fill In Your Values


---

Before the lane follower can run, the configuration block at the top of `lane_follower_adv.py` must match your robot, your phone stream, and your Colab calibration result.
CONFIGURATION block
```python
# ==================== CONFIGURATION ====================
ESP_IP         = "192.168.137.228"
UDP_CMD_PORT   = 5001
UDP_TELEM_PORT = 5002

BINARY_THRESHOLD   = 70
ROI_HEIGHT_PERCENT = 0.88
MORPH_KERNEL_SIZE  = 7
MIN_LINE_AREA      = 500
BASE_SPEED         = 100
TARGET_AREA        = 20000
MAX_SPEED          = 255
MAX_TURN           = 30
MAX_ACCEL          = 2
MOTOR_OUTPUT_SCALE = 10

SMOOTH_RATE = 0.04

BODY_TURN_KP = 100
BODY_TURN_KD = 4
BODY_TURN_KI = 0

DIST_KP = 0
DIST_KI = 0
DIST_KD = 0
MAX_DIST_INTEGRAL = 0.8

ANGLE_DEAD_ZONE = 0.00
AREA_DEAD_ZONE  = 0.2

ACQUIRE_FRAMES = 5
LOST_TIMEOUT   = 1.0

ROW_FRACS = [0.90, 0.75, 0.60, 0.45, 0.30]

STATE_COLORS = {
    "STOPPED":   (100, 100, 100),
    "SEARCHING": (255, 165,   0),
    "ACQUIRING": (255, 255,   0),
    "TRACKING":  (  0, 255,   0),
}
```

### Where Each Value Comes From
Constant Where to get the value ESP_IP Serial Monitor output from Exercise D. UDP_CMD_PORT Must match CMD_PORT in the robot_udp Arduino sketch. UDP_TELEM_PORT Must match TELEM_PORT in the Arduino sketch. BINARY_THRESHOLD From the Colab experiment notebook Cell 4 in P2.2. ROI_HEIGHT_PERCENT Leave at 0.88 until the camera is mounted and tested. MORPH_KERNEL_SIZE Start with the notebook value. Keep 7 unless the binary window has noise or gaps. MIN_LINE_AREA Leave at 500 for the first run. BASE_SPEED Start at 60 for the first run, then increase after testing. TARGET_AREA Leave at 20000 ; tune later in P2.8 Running and Tuning. MAX_SPEED , MAX_TURN , MAX_ACCEL Motion safety limits. Keep the defaults for the first run. MOTOR_OUTPUT_SCALE Leave at 10 . Lower values make the robot respond more strongly; higher values reduce response. SMOOTH_RATE Leave at 0.04 unless the row-scan output jitters. BODY_TURN_KP Leave at 100 ; tune later in P2.8. BODY_TURN_KD Leave at 4 ; tune later in P2.8. BODY_TURN_KI Leave at 0 for this robot. DIST_KP , DIST_KI , DIST_KD Distance-control path. Leave at 0 until the lane follower works reliably. MAX_DIST_INTEGRAL Safety clamp for distance PID integral. Leave the default. ANGLE_DEAD_ZONE , AREA_DEAD_ZONE Ignore tiny control errors. Keep defaults for the first run. ACQUIRE_FRAMES , LOST_TIMEOUT State-machine timing. Keep defaults unless detection flickers. ROW_FRACS Scan-row positions. Adjust only if the visible tape does not cross the default rows. STATE_COLORS HUD colors only. No tuning needed.

### Fill In Before Running
Value Your value Source ESP_IP "____________" Exercise D Serial Monitor BINARY_THRESHOLD ____________ Colab Cell 4 mobile_ip "____________" Exercise B phone camera stream
!!! warning "BINARY_THRESHOLD matters most"
    `BINARY_THRESHOLD` is the most important value in this file. If you have not run the Colab experiment notebook yet, stop here and complete lesson P2.2 first. Running with the wrong threshold is the most common reason the robot does not respond to the tape at all.

---

## The detect_lane Function


---

All four image processing steps : thresholding, ROI cropping, morphological cleaning, and row scanning : live inside a single function called `detect_lane()`. It takes one camera frame as input and returns the detected line position. Read it as one piece, not four separate ideas.
Reminder: what is in shared.py

`PID` : the controller used for steering
`MobileVideoStream` : phone camera thread
`Commander` : sends UDP motor commands
`Telemetry` : receives encoder data
`RobotState` : STOPPED / SEARCHING / ACQUIRING / TRACKING
`ramp()` : acceleration limiter

These are unchanged from Project 1. If anything in `lane_follower_adv.py` references one of these names, it is using the version from `shared.py`.
Complete detect_lane() function
```python
def detect_lane(frame):
    h, w  = frame.shape[:2]
    roi_y = int(h * (1 - ROI_HEIGHT_PERCENT))
    roi   = frame[roi_y:, :]

    blur = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    _, binary = cv2.threshold(blur, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    points = []
    for frac in ROW_FRACS:
        ry  = int(binary.shape[0] * frac)
        xs  = np.where(binary[ry] > 0)[0]
        if len(xs) < 15:
            continue
        runs = np.split(xs, np.where(np.diff(xs) > 1)[0] + 1)
        best = max(runs, key=len)
        if len(best) < 12:
            continue
        points.append((int((best[0] + best[-1]) / 2), ry + roi_y))

    area = int(np.count_nonzero(binary))

    if len(points) < 2:
        return None, None, None, area, roi_y, binary, points

    return points[0], points[-1], points, area, roi_y, binary, points
```

### 1. ROI
The first lines define the part of the frame that the detector is allowed to inspect. The phone sees floor, walls, furniture, shadows, and possibly the robot chassis; the lane detector should process only the floor area where the line can appear.
ROI crop
```python
h, w  = frame.shape[:2]
roi_y = int(h * (1 - ROI_HEIGHT_PERCENT))
roi   = frame[roi_y:, :]
```
`ROI_HEIGHT_PERCENT = 0.88` means the ROI starts 12 percent down from the top of the frame and continues to the bottom. The upper 12 percent is ignored. Detected y-coordinates are shifted back into full-frame coordinates later with `ry + roi_y` so `draw_debug()` can place the magenta dots on the original camera image.
IGNORED roi_y ROI - line detection happens here

### 2. Threshold Pipeline
You already saw each of these steps in the Colab experiment notebook (P2.2). Here they are combined into the pipeline that runs on every frame.
Blur, threshold, and morphology
```python
blur = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (7, 7), 0)
_, binary = cv2.threshold(blur, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
```
cv2.cvtColor(..., cv2.COLOR_BGR2GRAY) The phone image has blue, green, and red channels. The detector uses brightness contrast, so color is collapsed to one grayscale channel from 0 to 255. cv2.GaussianBlur(..., (7, 7), 0) Blur removes tiny camera noise before thresholding. A 7 by 7 Gaussian blur prevents isolated bright or dark specks from becoming false line pixels. cv2.THRESH_BINARY_INV Every pixel at or below BINARY_THRESHOLD becomes white. The inverted threshold is correct for a dark line on a light floor. If your line is white on a dark floor, use cv2.THRESH_BINARY instead. MORPH_OPEN then MORPH_CLOSE Opening removes small white blobs that are not the line. Closing repairs small holes and breaks in the tape stripe. The square kernel controls how aggressively both operations clean the binary image.
Before Line plus noise blobs. After OPEN Noise removed. After CLOSE Gaps filled.

### 3. Row Scanning
After thresholding and cleaning, the binary image shows the line as a white region. Row scanning finds the horizontal center of that white region at several heights in the image.
Row scanning loop
```python
points = []
for frac in ROW_FRACS:
    ry  = int(binary.shape[0] * frac)
    xs  = np.where(binary[ry] > 0)[0]
    if len(xs) < 15:
        continue
    runs = np.split(xs, np.where(np.diff(xs) > 1)[0] + 1)
    best = max(runs, key=len)
    if len(best) < 12:
        continue
    points.append((int((best[0] + best[-1]) / 2), ry + roi_y))
```
Line Meaning ROW_FRACS The five scan rows inside the ROI. 0.90 is near the bottom, closest to the robot; 0.30 is farther ahead. np.where(binary[ry] > 0)[0] Returns the x-coordinates of all white pixels in row ry . In the binary image, white means line. if len(xs) < 15 Rejects rows with too few white pixels. A tiny white run is usually noise, not tape. np.diff(xs) Measures gaps between consecutive white x-coordinates. np.split(...) Splits the white pixels into separate contiguous runs whenever a gap appears. best = max(runs, key=len) Chooses the longest continuous white run because the tape should be larger than reflections or leftover noise. if len(best) < 12 Rejects a run that is still too narrow to trust. (best[0] + best[-1]) / 2 Computes the center x-position of the longest white run. ry + roi_y Converts the y-position from ROI coordinates back to full-frame coordinates for drawing.

### 4. Two-Point Steering Error
Once `detect_lane()` has returned at least two points, `LaneFollower.control()` computes the same steering error you saw in Colab Cell 7. A line has direction, so one point is not enough: the robot uses a near point and a far point.
Two-point error calculation
```python
position_error  = (near_cx - w / 2) / w
direction_error = (far_cx  - near_cx) / w
raw_angle_error = 0.25 * position_error + 0.75 * direction_error
```
Straight Near and far aligned. Direction error is near zero. Gentle curve Far point shifts right. Robot starts turning early. Sharp curve Far point moves strongly. Turn command grows.
`position_error` is the current offset from image center. `direction_error` is the line direction from near to far. The default blend gives 25 percent weight to current position and 75 percent to lookahead direction, so the robot anticipates curves instead of reacting late.

### 5. Exponential Smoothing
Row scanning can jitter because a slightly different set of pixels may pass the threshold each frame. Immediately after the raw angle calculation, `LaneFollower.control()` smooths the error before it reaches the PID. This is the same idea as the browser interpreter demo from the old smoothing lesson.
Smoothed steering error
```python
self.smoothed_angle = (
    SMOOTH_RATE * self.smoothed_angle
    + (1 - SMOOTH_RATE) * raw_angle_error
)
```
With `SMOOTH_RATE = 0.04`, the old smoothed value has 4 percent influence and the new raw measurement has 96 percent influence. Lower values react faster but can twitch on noisy detections; higher values react more slowly but can help on unstable lighting.
Exponential smoothing demo
```python
smoothed = 0.0
raw_values = [0.1, 0.15, 0.12, 0.3, 0.28, 0.25, 0.05, 0.02]
SMOOTH_RATE = 0.04

for raw in raw_values:
    smoothed = SMOOTH_RATE * smoothed + (1 - SMOOTH_RATE) * raw
    print(f"raw={raw:.2f}  smoothed={smoothed:.3f}")
```

### 6. What detect_lane Returns
The return values are the contract between the vision function, `LaneFollower.control()`, and `draw_debug()`.
Return values
```python
if len(points) < 2:
    return None, None, None, area, roi_y, binary, points

return points[0], points[-1], points, area, roi_y, binary, points
```
Returned value Where it is used points[0] / near_pt The near line center used by LaneFollower.control() for current position error and by draw_debug() for the first magenta point. points[-1] / far_pt The far line center used for direction error and the blue lookahead line in the main debug window. points All detected row centers. draw_debug() draws them as magenta dots; LaneFollower.control() stores them in the debug dictionary. area Total white pixels in the cleaned binary image. The controller uses it as a distance/speed diagnostic through dist_error . roi_y The y-coordinate of the ROI boundary. draw_debug() draws this as the orange horizontal line. binary The cleaned black-and-white image shown in the binary debug window. None values If fewer than two scan points are found, LaneFollower.control() treats the line as missing and enters the search/stop behavior.
!!! info "One function, one job"
    `detect_lane()` does one thing: convert a raw camera frame into lane points and debug values that become the steering error. Everything before it (phone stream, state machine, PID, UDP) and everything after it (LaneFollower, Commander, Telemetry) does not know or care how those points were computed. Swap `detect_lane()` for a YOLO call and you have Project 1. Swap it for a different algorithm and you have a different robot entirely. This separation is the most important software design principle in the whole course.

---

## The LaneFollower Class


---

`LaneFollower.control()` mirrors Project 1's `Tracker.control()`. The method gets a frame, detects the target, updates the state machine, computes motor targets, ramps the output, and returns debug data.
lane_follower_adv.py:300-368
LaneFollower.control()
```python
def control(self, frame):
    h, w = frame.shape[:2]
    near_pt, far_pt, centers, area, roi_y, binary, points = detect_lane(frame)
    line_found = near_pt is not None and far_pt is not None

    self._update_state(line_found)

    if line_found:
        near_cx, near_cy = near_pt
        far_cx, _ = far_pt

        position_error = (near_cx - w / 2) / w
        direction_error = (far_cx - near_cx) / w
        raw_angle_error = 0.25 * position_error + 0.75 * direction_error

        if abs(raw_angle_error) < ANGLE_DEAD_ZONE:
            raw_angle_error = 0.0
        if raw_angle_error != 0.0:
            self.last_turn_dir = np.sign(raw_angle_error)

        self.smoothed_angle = (
            SMOOTH_RATE * self.smoothed_angle
            + (1 - SMOOTH_RATE) * raw_angle_error
        )

        turn_cmd = self.pid_turn.update(self.smoothed_angle)
        dist_error = (TARGET_AREA - area) / TARGET_AREA
        if abs(dist_error) < AREA_DEAD_ZONE:
            dist_error = 0.0
        dist_cmd = self.pid_dist.update(dist_error)

        tgt_left, tgt_right = scale_motors(
            (BASE_SPEED + turn_cmd) * self.speed_limit,
            (BASE_SPEED - turn_cmd) * self.speed_limit,
        )
        cx, cy = near_cx, near_cy

    elif self.state == RobotState.SEARCHING:
        search_turn = 8 * self.last_turn_dir * self.speed_limit
        forward = 5 * self.speed_limit
        tgt_left, tgt_right = scale_motors(forward + search_turn, forward - search_turn)
        dist_error, turn_cmd, area, cx = 0.0, 0.0, 0, w // 2
        cy = h // 2

    else:
        tgt_left = tgt_right = 0
        dist_error, turn_cmd, area, cx = 0.0, 0.0, 0, w // 2
        cy = h // 2

    self.cur_left = ramp(self.cur_left, tgt_left, MAX_ACCEL)
    self.cur_right = ramp(self.cur_right, tgt_right, MAX_ACCEL)

    debug = {
        "state": self.state,
        "line_found": line_found,
        "cx": cx,
        "cy": cy,
        "area": area,
        "roi_y": roi_y,
        "binary": binary,
        "smoothed_angle": self.smoothed_angle,
        "turn_cmd": turn_cmd,
        "dist_error": dist_error,
        "base_speed": BASE_SPEED if line_found else 0,
        "speed_limit": self.speed_limit,
        "points": points if line_found else [],
    }
    return round(self.cur_left), round(self.cur_right), debug
```

### Compared to Project 1
detect_lane(frame) replaces _detect(frame) Project 1 returned one bounding box. Project 2 returns near_pt , far_pt , area , roi_y , binary , and scan points . line_found = near_pt is not None and far_pt is not None One lane point is not enough because steering needs direction. The controller requires both near and far points. self._update_state(line_found) The state machine is the same as Project 1. Only the input changed from "bounding box found" to "line found." dist_error = (TARGET_AREA - area) / TARGET_AREA The sign is intentionally different from Project 1. In Project 1, large object area meant the robot was too close. In Project 2, large binary lane area means more line coverage. DIST_KP is 0 by default, so this speed-control path is disabled until you choose to tune it. scale_motors() Project 1 performed motor scaling inline. Project 2 extracts the same idea into a helper so neither side exceeds MAX_SPEED .
lane_follower_adv.py:212-218
scale_motors()
```python
def scale_motors(left, right):
    peak = max(abs(left), abs(right))
    if peak > MAX_SPEED:
        scale = MAX_SPEED / peak
        return left * scale, right * scale
    return left, right
```

---

## The Main Loop


---

`main()` connects the phone stream, lane follower, UDP motor sender, telemetry receiver, keyboard controls, and OpenCV windows into one live robot loop.
Shared import line
```python
from shared import PID, MobileVideoStream, Commander, Telemetry, RobotState, ramp
```
Complete main() function
```python
def main():
    global BINARY_THRESHOLD

    mobile_ip = "10.18.204.215"
    stream_url = f"http://{mobile_ip}:8080/video"
    print(f"Stream: {stream_url}")

    video     = MobileVideoStream(stream_url)
    commander = Commander(ESP_IP, UDP_CMD_PORT)
    telemetry = Telemetry(UDP_TELEM_PORT)

    print("Waiting for camera...")
    for _ in range(30):
        if video.connected:
            break
        time.sleep(0.5)
        print(".", end="", flush=True)
    print()

    if not video.connected:
        print("Could not connect to camera")
        exit(1)

    follower = LaneFollower()
    commander.stop()
    time.sleep(0.5)

    print("\nLane Following Active")
    print(
        f"BINARY_THRESHOLD: {BINARY_THRESHOLD}  "
        f"ROI: {ROI_HEIGHT_PERCENT:.0%}  "
        f"KERNEL: {MORPH_KERNEL_SIZE}x{MORPH_KERNEL_SIZE}  "
        f"TARGET_AREA: {TARGET_AREA}"
    )
    print("Q=Quit  R=Reset  T=Threshold up  Y=Threshold down")
    print("=" * 60 + "\n")

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
            print(f"FPS: {fps_count / (time.time() - fps_timer):.1f}")
            fps_count = 0
            fps_timer = time.time()

        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        left_speed, right_speed, debug = follower.control(frame)
        print(left_speed, right_speed)

        now = time.time()
        if now - last_cmd_time >= CMD_INTERVAL:
            commander.motors(0, 0) if (left_speed == right_speed == 0) \
                else commander.motors(
                    -left_speed / MOTOR_OUTPUT_SCALE,
                    right_speed / MOTOR_OUTPUT_SCALE,
                )
            last_cmd_time = now

        draw_debug(frame, debug, left_speed, right_speed, telemetry.read())
        cv2.imshow("Lane Follower", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            follower.pid_turn.reset()
            follower.pid_dist.reset()
            follower.smoothed_angle = 0.0
            print("Reset")
        elif key == ord('t'):
            BINARY_THRESHOLD = min(BINARY_THRESHOLD + 5, 255)
            print(f"BINARY_THRESHOLD -> {BINARY_THRESHOLD}")
        elif key == ord('y'):
            BINARY_THRESHOLD = max(BINARY_THRESHOLD - 5, 0)
            print(f"BINARY_THRESHOLD -> {BINARY_THRESHOLD}")

    print("\nShutting down...")
    commander.stop()
    time.sleep(0.2)
    video.stop()
    telemetry.stop()
    cv2.destroyAllWindows()
    print("Done")
```

### Frame Rotation
Project 2-only frame rotation
```python
frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
```
This line does not exist in Project 1. The phone camera outputs frames with the wrong orientation when mounted sideways on the robot. This rotation corrects it before any processing. Adjust the constant if your image appears rotated in the wrong direction.

### Scaled Motor Command
Motor send line
```python
commander.motors(
    -left_speed / MOTOR_OUTPUT_SCALE,
    right_speed / MOTOR_OUTPUT_SCALE,
)
```
There are two differences from Project 1. First, `MOTOR_OUTPUT_SCALE` is `10` by default: `BODY_TURN_KP = 100` can produce outputs up to about +/-3000, so dividing by 10 maps that to about +/-300 before the driver clips to +/-255. If response feels weak, try `8`; if it oversteers, try `12`. Second, this line may still need a motor-sign adjustment depending on your chassis. Project 1 now sends `left_speed` as positive by default; if the robot spins instead of going forward, add or remove the negative sign for your wiring.

### Runtime Threshold Adjustment
T/Y threshold keys
```python
elif key == ord('t'):
    BINARY_THRESHOLD = min(BINARY_THRESHOLD + 5, 255)
    print(f"BINARY_THRESHOLD -> {BINARY_THRESHOLD}")
elif key == ord('y'):
    BINARY_THRESHOLD = max(BINARY_THRESHOLD - 5, 0)
    print(f"BINARY_THRESHOLD -> {BINARY_THRESHOLD}")
```
`T` increases `BINARY_THRESHOLD` by 5. `Y` decreases it by 5. Use this for fine tuning only after you found your baseline value in Colab. If you need to adjust by more than +/-15 from your Colab value, your lighting conditions have changed significantly; rerun the Colab notebook with a new photo.

---

## Visualisation and the Binary Window


---

Project 2 has two OpenCV windows. The first shows the camera frame with a HUD. The second shows the binary image the algorithm actually uses for lane detection.
**Lane Follower**
Main camera frame with ROI line, detected points, steering guide, state text, motor commands, and encoder telemetry.
**Binary window**
Processed black-and-white image after thresholding and morphology. This is the most important tuning view.
**Live threshold**
Press `T` and `Y` while watching the binary window to tune detection in real lighting.

### Main Window Elements
Element Meaning Orange horizontal line Start of the ROI. Everything above it is ignored. Magenta dots Row-scan lane centers found in P2.4. Blue line Line from near point to far point, showing detected lane direction. Green center line Frame center, the desired steering reference. Cyan smooth line Smoothed steering error after exponential filtering. Colored near-point circle Distance/speed diagnostic from line area.
!!! info "Binary window screenshot substitute"
    The live OpenCV window cannot be embedded here. During testing, treat the binary window as your truth source: a clean setup shows a black floor and one continuous white stripe for the lane.

### How to Diagnose the Binary Window
Too much white everywhere Threshold is too high Floor pixels are being detected as line. Press Y to decrease BINARY_THRESHOLD . No white at all Threshold is too low The line is being excluded. Press T to increase BINARY_THRESHOLD . White stripe exists but no magenta dots ROI or scan rows are wrong Adjust ROI_HEIGHT_PERCENT or ROW_FRACS so the scan rows cross the visible line. White stripe has gaps Morphology or threshold needs adjustment Increase the closing kernel or lower the threshold slightly so broken line segments reconnect.
lane_follower_adv.py:371-421
draw_debug() overview
```python
def draw_debug(frame, debug, left_speed, right_speed, telem):
    h, w = frame.shape[:2]
    cv2.line(frame, (0, debug["roi_y"]), (w, debug["roi_y"]), (0, 165, 255), 1)
    cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
    smooth_x = int(w / 2 + debug["smoothed_angle"] * w)
    cv2.line(frame, (smooth_x, 0), (smooth_x, h), (0, 255, 255), 2)

    if debug["line_found"]:
        for px, py in debug["points"]:
            cv2.circle(frame, (px, py), 5, (255, 0, 255), -1)
        if len(debug["points"]) >= 2:
            cv2.line(frame, debug["points"][0], debug["points"][-1], (255, 0, 0), 2)

    if debug["binary"] is not None:
        bin_disp = cv2.resize(cv2.cvtColor(debug["binary"], cv2.COLOR_GRAY2BGR), (640, 480))
        cv2.imshow("Binary (lane detection)", bin_disp)
```

---

## Running and Tuning


---

Tune Project 2 in stages. The first calibration happens in Colab on a still image. Runtime tuning only makes small adjustments after the still-image pipeline already works.

### Stage 0: Before Running Code
BINARY_THRESHOLD is filled in from the Colab notebook, not left at the default 70 . Phone is mounted on the robot and the stream is verified in a browser. Robot is on the track and tape is visible in the browser image. Arduino is running the robot_udp sketch from Exercise D. Battery is connected and the shield jumper is set to BATT PWR .
!!! warning "Do not skip Colab calibration"
    If `BINARY_THRESHOLD` is still `70`, stop here and run P2.2. A wrong threshold is the most common reason students think the robot code is broken.

### Stage 1: Run with Motors Disabled
Temporarily disable motor output so you can inspect the binary window while moving the robot by hand.
Disable motors for calibration
```python
# Temporarily replace the live motor command with this:
commander.motors(0, 0)  # motors disabled for calibration
```
Run `python lane_follower_adv.py`. Watch the binary window only. The tape should appear as one clear white stripe, and the main window should draw magenta dots on the tape.
What you see What to do White stripe present and dots appear. Threshold is correct. Floor is included as white. Press Y to decrease threshold. No white stripe. Press T to increase threshold. You need more than about +/-15 from the Colab value. Retake a photo under current lighting and rerun P2.2.

### Stage 2: Verify Steering Direction
Restore the motor send line, set `BASE_SPEED = 30`, and place the robot on a straight section. It should drive forward while keeping the tape centered.
Restore live motor output
```python
commander.motors(-left_speed / MOTOR_OUTPUT_SCALE, right_speed / MOTOR_OUTPUT_SCALE)
```
If the robot veers left when it should go right, invert the steering mix inside `LaneFollower.control()`.
Invert steering if needed
```python
# Default:
tgt_left = (BASE_SPEED + turn_cmd) * self.speed_limit
tgt_right = (BASE_SPEED - turn_cmd) * self.speed_limit

# Inverted:
tgt_left = (BASE_SPEED - turn_cmd) * self.speed_limit
tgt_right = (BASE_SPEED + turn_cmd) * self.speed_limit
```

### Command Scaling
The motor driver accepts roughly `-255` to `255`. The turn PID uses `BODY_TURN_KP = 100`, so internal controller values can be much larger than a first-run motor command should be.
Scaled motor send
```python
MOTOR_OUTPUT_SCALE = 10
commander.motors(-left_speed / MOTOR_OUTPUT_SCALE, right_speed / MOTOR_OUTPUT_SCALE)
```
`MOTOR_OUTPUT_SCALE` is a calibration constant, not a magic number. If the robot reacts weakly to curves, try `8` or `7`. If it oversteers, try `12` or `15`.
**Write your working scale** — `commander.motors(-left_speed / _____, right_speed / _____)`

### Stage 3: First Full Run
Set `BASE_SPEED = 60`, place the robot on the track, and run the script. The robot should follow a gentle oval.

### Stage 4: Tune for Speed and Curves
Robot cuts corners Lookahead too weak or speed too high Increase direction-error weight from 0.75 toward 0.85 , or reduce BASE_SPEED . Robot oscillates left-right on a straight line Turn gain or smoothing issue Reduce BODY_TURN_KP by about 30 percent. If the binary detection is noisy, increase smoothing by raising SMOOTH_RATE in this code's formula. Robot detects the line in the wrong position ROI does not match phone angle Adjust ROI_HEIGHT_PERCENT until the line appears in the useful middle region of the binary window. Robot works on straight sections but loses curves Scan rows miss the tight bend Add more values to ROW_FRACS or shift the fractions so more rows cross the curved line.

---

## Connecting the Dots


---

Project 2 proves a robotics design principle: the architecture outlasts the algorithm. The robot does not know whether Python is following a ball or a line. It receives motor commands and sends encoder ticks.
**The whole change** — `Remove: YOLO import and class Tracker`
`Add: detect_lane() and class LaneFollower`
The laptop still does the thinking. The Arduino still does the driving. UDP still connects them. In professional systems, the laptop may be ROS, UDP may become ROS topics, and the Arduino may become a motor-control node, but the pattern is the same.

### What You Have Built Across the Course
Part What you learned Where it appears in Project 2 Chapter 1 Python fundamentals. Every file and function. Chapters 2-6 ML and vision concepts. Knowing when not to use ML. Exercise A YOLO training and inference. Intentionally replaced by detect_lane() . Exercise B Phone stream and MJPEG parsing. MobileVideoStream , unchanged. Exercise C Encoder motors and precise movement. Encoder telemetry in the HUD. Exercise D UDP WiFi control. Commander and Telemetry , unchanged. Project 1 PID, state machine, full loop. LaneFollower keeps the same control architecture. Project 2 Classical CV and two-point steering. The new lane-detection module.

### Final Checklist
Binary window shows only the line white and the floor dark. Magenta dots appear on the line when the camera is pointed at it. Robot drives straight on a straight section. Robot follows a 90-degree corner without losing the line. You can explain position_error versus direction_error . You can explain how SMOOTH_RATE affects cornering. You understand why DIST_KP is 0 by default and when you would enable it. You can diagnose detection problems from the binary window.
!!! tip "Classical vision robot"
    You now have two complete robot behaviors running on the same architecture: a YOLO object tracker and an OpenCV lane follower.


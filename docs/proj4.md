#  Project 4: Pan-Tilt Ball Tracker

---

## What You Are Building


---

In Project 1 the camera was mounted on a robot that moved its whole body to follow a target. This exercise keeps the robot fixed and moves only the camera with two small servos.
A pan-tilt mount is the same idea used in tracking security cameras, telescope mounts, and camera gimbals. The system has one camera, two servo motors, one YOLO model, and two PID controllers running at the same time.
Phone Camera MJPEG over WiFi → Laptop Python YOLO finds cx, cy PAN PID horizontal error to delta angle TILT PID vertical error to delta angle UDP command SERVO,pan,tilt → Arduino UNO R4 WiFi drives pan and tilt servos Arduino replies with SERVO_ACK,current_pan,current_tilt .

### What Is New
Concept Where you saw it What is new here YOLO detection Exercise A, Project 1 Same model call. Phone stream Exercise B Same MobileVideoStream . UDP communication Exercise D New packet: SERVO,pan,tilt . PID controller Project 1 Velocity mode: add the output to an angle. DC motors Exercises C and D Replaced by servo motors.

---

## What Is a Servo Motor?


---

A DC motor spins continuously when powered. A servo motor moves to a commanded angle and holds that position.
A servo has a built-in position sensor and control circuit. You tell it `servo.write(90)`, and the Arduino Servo library generates the timing signal that makes the motor move to 90 degrees.
1 ms pulse Servo goes near 0 degrees. 1.5 ms pulse Servo goes near 90 degrees. 2 ms pulse Servo goes near 180 degrees.
!!! info "Servo vs DC motor"
    Use a servo when you need position control. Use a DC motor when you need continuous rotation. A camera mount needs to point at a specific angle, so servos are the correct choice.

### Servo Angle Range
Most hobby servos rotate from 0 to 180 degrees. This exercise clamps pan to `10..170` and tilt to `30..150` to keep the mount and camera cable away from mechanical limits.
!!! warning "Never command past the physical stop"
    The Arduino `clampAngle()` function and Python `np.clip()` call both exist to prevent overheating a stalled servo.

---

## The Pan-Tilt Mechanism


---

A pan-tilt mechanism is two servo motors mounted at 90 degrees to each other. One controls left-right pan. The other controls up-down tilt.
Top viewleft**Pan servo**rightThe pan axis rotates the camera horizontally.
Side viewup**Tilt servo**downThe tilt axis raises and lowers the camera.

### Home Position
`PAN_HOME = 90` points straight ahead. `TILT_HOME = 40` points slightly downward, useful when the ball is on a table or floor.
!!! tip "First upload test"
    When the Arduino sketch starts, both servos immediately move to home. That confirms power, signal wiring, and pin choices before Python runs.

### Inversion Variables
If the camera moves the wrong way, change the matching Python flag: `PAN_INVERT` or `TILT_INVERT`. No rewiring is required.

---

## Get the Code


---

Place these files in the same folder as `shared.py` and `best.pt`. Open the Arduino sketch in Arduino IDE and upload it to the UNO R4 WiFi.
shared.pyAlready used in Project 1. Keep it beside the tracker script.`shared.py`
pan_and_tilt.pyThe Python YOLO tracker and servo commander.`pan_and_tilt.py`
pan_tilt_arduino.inoThe Arduino servo-over-WiFi sketch.`pan_tilt_arduino.ino`
test_servos.pyA quick command-only test before running YOLO.`test_servos.py`

### Values You Must Fill In
File Variable Meaning pan_and_tilt.py ESP_IP Arduino IP from Serial Monitor. pan_and_tilt.py MOBILE_IP Phone camera app IP address. pan_and_tilt.py TARGET_OBJECT Your YOLO class name, usually ball . pan_and_tilt.py PAN_INVERT , TILT_INVERT Set to True only if an axis moves backward. pan_tilt_arduino.ino SSID , PASSWORD Your WiFi network name and password.
!!! warning "Fill in WiFi first"
    If `SSID` or `PASSWORD` is wrong, the Arduino will never print an IP address and Python cannot reach it.
Complete Python file - loads from `pan_and_tilt.py`Complete pan_and_tilt.pyVS Code"""
pan_and_tilt.py

YOLO ball tracker for a fixed pan-tilt camera rig.

Architecture:
  - A phone streams camera frames over WiFi.
  - YOLO detects the target object in each frame.
  - One PID controls pan from horizontal image error.
  - One PID controls tilt from vertical image error.
  - Each PID outputs degrees per frame, which are added to the current angle.
  - Python sends UDP packets as SERVO,<pan>,<tilt> to the Arduino.
  - The Arduino replies with SERVO_ACK,<pan>,<tilt> for the HUD.

Place this file in the same folder as shared.py and best.pt.
"""

import socket
import threading
import time

import cv2
import numpy as np

from shared import PID, MobileVideoStream, RobotState


# -----------------------------------------------------------------------------
# CONFIGURATION - fill these in for your setup
# -----------------------------------------------------------------------------

# Network
ESP_IP = "192.168.x.x"       # Arduino IP from Serial Monitor
MOBILE_IP = "192.168.x.x"    # Phone IP from IP Webcam or Simple IP Camera
UDP_CMD_PORT = 5001
UDP_TELEM_PORT = 5002

# Vision
TARGET_OBJECT = "ball"

# Servo physical limits in degrees
PAN_MIN, PAN_MAX = 10, 170
TILT_MIN, TILT_MAX = 30, 150
PAN_HOME = 90
TILT_HOME = 40

# Flip these if the camera moves the wrong direction on that axis.
PAN_INVERT = False
TILT_INVERT = False

# Ignore tiny image errors near center to prevent buzzing.
ANGLE_DEAD_ZONE_H = 0.03
ANGLE_DEAD_ZONE_V = 0.01

# PID gains for velocity-mode output in degrees per frame.
PAN_KP = 8.0
PAN_KI = 0.0
PAN_KD = 2.0
MAX_PAN_SPEED = 3.0
MAX_PAN_INTEGRAL = 0.3

TILT_KP = 6.0
TILT_KI = 0.0
TILT_KD = 1.5
MAX_TILT_SPEED = 3.0
MAX_TILT_INTEGRAL = 0.3

# State machine
ACQUIRE_FRAMES = 5
LOST_TIMEOUT = 1.2
SEARCH_PAN_SPEED = 1.5
SEARCH_TILT_SPEED = 0.0
CMD_INTERVAL = 0.03


# -----------------------------------------------------------------------------
# COMMANDER - sends SERVO commands to the Arduino
# -----------------------------------------------------------------------------

class ServoCommander:
    """Sends UDP servo commands to the Arduino."""

    def __init__(self, ip, port):
        self._ip = ip
        self._port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def _send(self, message):
        self._sock.sendto(message.encode("utf-8"), (self._ip, self._port))

    def servos(self, pan, tilt):
        """Send absolute servo angles in integer degrees."""
        self._send(f"SERVO,{int(pan)},{int(tilt)}")

    def home(self):
        """Ask the Arduino to return both servos to their home angles."""
        self._send("HOME")

    def stop(self):
        """Ask the Arduino to home the rig before the Python script exits."""
        self._send("STOP")
        self._sock.close()


# -----------------------------------------------------------------------------
# TELEMETRY - receives optional SERVO_ACK packets from the Arduino
# -----------------------------------------------------------------------------

class ServoTelemetry:
    """
    Receives SERVO_ACK,pan,tilt packets from the Arduino.

    Servos do not provide encoder feedback in this project. The ACK contains the
    Arduino's current commanded angle after its smoothing step.
    """

    def __init__(self, port):
        self.pan = PAN_HOME
        self.tilt = TILT_HOME
        self.running = True
        self._lock = threading.Lock()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("", port))
        self._sock.settimeout(1.0)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.running:
            try:
                data, _ = self._sock.recvfrom(64)
                parts = data.decode("utf-8").strip().split(",")
                if parts[0] == "SERVO_ACK" and len(parts) == 3:
                    with self._lock:
                        self.pan = int(parts[1])
                        self.tilt = int(parts[2])
            except (OSError, socket.timeout, UnicodeDecodeError, ValueError):
                pass

    def read(self):
        with self._lock:
            return {"pan": self.pan, "tilt": self.tilt}

    def stop(self):
        self.running = False
        self._sock.close()


# -----------------------------------------------------------------------------
# PAN-TILT TRACKER
# -----------------------------------------------------------------------------

class PanTiltTracker:
    """YOLO-based target tracker that drives two independent servo axes."""

    STATE_COLORS = {
        RobotState.STOPPED: (100, 100, 100),
        RobotState.SEARCHING: (255, 165, 0),
        RobotState.ACQUIRING: (255, 255, 0),
        RobotState.TRACKING: (0, 255, 0),
    }

    def __init__(self, model_path, target):
        print("Loading YOLO model...")
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.target = target

        try:
            self.model.to("cuda")
            print("  Using CUDA")
        except Exception:
            print("  Using CPU")
        self.model.fuse()

        self.pid_pan = PID(
            PAN_KP,
            PAN_KI,
            PAN_KD,
            max_integral=MAX_PAN_INTEGRAL,
            output_limits=(-MAX_PAN_SPEED, MAX_PAN_SPEED),
        )
        self.pid_tilt = PID(
            TILT_KP,
            TILT_KI,
            TILT_KD,
            max_integral=MAX_TILT_INTEGRAL,
            output_limits=(-MAX_TILT_SPEED, MAX_TILT_SPEED),
        )

        self.pan_angle = float(PAN_HOME)
        self.tilt_angle = float(TILT_HOME)

        self.state = RobotState.STOPPED
        self.acquire_count = 0
        self.last_seen_t = 0.0
        self.last_turn_dir = 1.0
        self.speed_limit = 1.0

    def _detect(self, frame):
        """Return the largest target bounding box as (x1, y1, x2, y2), or None."""
        results = self.model(frame, imgsz=320, verbose=False, conf=0.60)
        best = None
        best_area = 0

        for result in results:
            for box in result.boxes:
                class_name = self.model.names[int(box.cls[0])]
                if class_name != self.target:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best = (x1, y1, x2, y2)

        return best

    def _update_state(self, found):
        if found:
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
            else:
                self.speed_limit = 1.0
            return

        elapsed = time.time() - self.last_seen_t
        if elapsed < LOST_TIMEOUT:
            self.state = RobotState.SEARCHING
            self.speed_limit = 0.4
        else:
            self.state = RobotState.STOPPED
            self.speed_limit = 0.0
            self.pid_pan.reset()
            self.pid_tilt.reset()

    def control(self, frame):
        """
        Run one detection and control cycle.

        Returns:
            pan_angle, tilt_angle, debug_dict
        """
        h, w = frame.shape[:2]
        bbox = self._detect(frame)
        self._update_state(bbox is not None)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            err_pan = (cx - w / 2) / w
            err_tilt = (cy - h / 2) / h

            if abs(err_pan) < ANGLE_DEAD_ZONE_H:
                err_pan = 0.0
            if abs(err_tilt) < ANGLE_DEAD_ZONE_V:
                err_tilt = 0.0

            if err_pan != 0.0:
                self.last_turn_dir = float(np.sign(err_pan))

            pan_sign = -1.0 if PAN_INVERT else 1.0
            tilt_sign = -1.0 if TILT_INVERT else 1.0

            # PID outputs angular velocity in degrees per frame.
            delta_pan = self.pid_pan.update(err_pan) * self.speed_limit * pan_sign
            delta_tilt = self.pid_tilt.update(err_tilt) * self.speed_limit * tilt_sign
            debug_cx, debug_cy = int(cx), int(cy)

        elif self.state == RobotState.SEARCHING:
            if self.pan_angle <= PAN_MIN + 1:
                self.last_turn_dir = 1.0
            elif self.pan_angle >= PAN_MAX - 1:
                self.last_turn_dir = -1.0

            delta_pan = SEARCH_PAN_SPEED * self.last_turn_dir * self.speed_limit
            delta_tilt = SEARCH_TILT_SPEED
            err_pan = 0.0
            err_tilt = 0.0
            debug_cx, debug_cy = w // 2, h // 2

        else:
            delta_pan = 0.0
            delta_tilt = 0.0
            err_pan = 0.0
            err_tilt = 0.0
            debug_cx, debug_cy = w // 2, h // 2

        self.pan_angle = float(np.clip(self.pan_angle + delta_pan, PAN_MIN, PAN_MAX))
        self.tilt_angle = float(np.clip(self.tilt_angle + delta_tilt, TILT_MIN, TILT_MAX))

        debug = {
            "state": self.state,
            "bbox": bbox,
            "err_pan": err_pan,
            "err_tilt": err_tilt,
            "delta_pan": delta_pan,
            "delta_tilt": delta_tilt,
            "pan_angle": self.pan_angle,
            "tilt_angle": self.tilt_angle,
            "speed_limit": self.speed_limit,
            "cx": debug_cx,
            "cy": debug_cy,
        }
        return round(self.pan_angle), round(self.tilt_angle), debug

    def home(self):
        self.pan_angle = float(PAN_HOME)
        self.tilt_angle = float(TILT_HOME)
        self.pid_pan.reset()
        self.pid_tilt.reset()


# -----------------------------------------------------------------------------
# VISUALIZATION
# -----------------------------------------------------------------------------

def draw_overlay(frame, debug, pan, tilt, telem, scale_x=1.0, scale_y=1.0):
    h, w = frame.shape[:2]
    state_color = PanTiltTracker.STATE_COLORS.get(debug["state"], (255, 255, 255))

    cx_frame, cy_frame = w // 2, h // 2
    cv2.line(frame, (cx_frame, 0), (cx_frame, h), (0, 255, 0), 1)
    cv2.line(frame, (0, cy_frame), (w, cy_frame), (0, 255, 0), 1)

    if debug["bbox"] is not None:
        x1, y1, x2, y2 = debug["bbox"]
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
        cx_scaled = int(debug["cx"] * scale_x)
        cy_scaled = int(debug["cy"] * scale_y)

        centered = (
            abs(debug["err_pan"]) < ANGLE_DEAD_ZONE_H
            and abs(debug["err_tilt"]) < ANGLE_DEAD_ZONE_V
        )
        box_color = (0, 255, 0) if centered else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.circle(frame, (cx_scaled, cy_scaled), 6, (0, 0, 255), -1)
        cv2.line(frame, (cx_frame, cy_frame), (cx_scaled, cy_scaled), (0, 200, 255), 1)

    lines = [
        (f"STATE:  {debug['state']}", state_color),
        (f"PAN:    {pan:3d} deg  err:{debug['err_pan']:+.3f}", (0, 255, 255)),
        (f"TILT:   {tilt:3d} deg  err:{debug['err_tilt']:+.3f}", (255, 200, 0)),
        (
            f"dPAN:  {debug['delta_pan']:+.2f} deg/f  "
            f"dTILT:{debug['delta_tilt']:+.2f} deg/f",
            (200, 200, 200),
        ),
        (f"SPD LIM: {debug['speed_limit']:.0%}", (255, 255, 255)),
        (f"TELEM pan:{telem['pan']} deg  tilt:{telem['tilt']} deg", (255, 165, 0)),
    ]

    for i, (text, color) in enumerate(lines):
        bold = i in (0, 1, 2)
        cv2.putText(
            frame,
            text,
            (10, 25 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60 if bold else 0.50,
            color,
            2 if bold else 1,
        )


# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------

def main():
    stream_url = f"http://{MOBILE_IP}:8080/video"
    print(f"Video stream : {stream_url}")
    print(f"ESP target   : {ESP_IP}:{UDP_CMD_PORT}")

    video = MobileVideoStream(stream_url)
    commander = ServoCommander(ESP_IP, UDP_CMD_PORT)
    telemetry = ServoTelemetry(UDP_TELEM_PORT)

    print("Waiting for camera", end="", flush=True)
    for _ in range(30):
        if video.connected:
            break
        time.sleep(0.5)
        print(".", end="", flush=True)
    print()

    if not video.connected:
        print("Could not connect to camera. Check MOBILE_IP and the camera app.")
        video.stop()
        telemetry.stop()
        return

    tracker = PanTiltTracker("best.pt", TARGET_OBJECT)
    commander.home()
    time.sleep(0.5)

    print(f"\nTracking: {TARGET_OBJECT}")
    print("Q = quit   H = home servos   R = reset PIDs")
    print("=" * 60)

    last_fid = -1
    last_cmd_time = 0.0
    fps_count = 0
    fps_timer = time.time()

    try:
        while True:
            frame, fid = video.read()
            if frame is None or fid == last_fid:
                time.sleep(0.005)
                continue

            last_fid = fid
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            fps_count += 1
            if fps_count >= 30:
                elapsed = time.time() - fps_timer
                fps = fps_count / elapsed
                print(
                    f"FPS:{fps:.1f}  pan:{tracker.pan_angle:.1f}  "
                    f"tilt:{tracker.tilt_angle:.1f}  state:{tracker.state}"
                )
                fps_count = 0
                fps_timer = time.time()

            pan_cmd, tilt_cmd, debug = tracker.control(frame)

            now = time.time()
            if now - last_cmd_time >= CMD_INTERVAL:
                commander.servos(pan_cmd, tilt_cmd)
                last_cmd_time = now

            orig_h, orig_w = frame.shape[:2]
            disp = cv2.resize(frame, (640, 480))
            draw_overlay(
                disp,
                debug,
                pan_cmd,
                tilt_cmd,
                telemetry.read(),
                640 / orig_w,
                480 / orig_h,
            )

            cv2.imshow(f"Pan-Tilt Tracker - {TARGET_OBJECT}", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("h"):
                tracker.home()
                commander.home()
                print(f"Servos homed to {PAN_HOME}/{TILT_HOME}")
            elif key == ord("r"):
                tracker.pid_pan.reset()
                tracker.pid_tilt.reset()
                print("PIDs reset")

    finally:
        print("\nShutting down...")
        try:
            commander.stop()
        finally:
            video.stop()
            telemetry.stop()
            cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
Complete Arduino sketch - loads from `pan_tilt_arduino.ino`Complete pan_tilt_arduino.inoArduino#include <Servo.h>
#include <WiFiS3.h>
#include <WiFiUDP.h>

// Fill in your WiFi network before uploading.
const char* SSID = "YOUR_WIFI_NAME";
const char* PASSWORD = "YOUR_WIFI_PASSWORD";

const int UDP_PORT = 5001;
const int TELEM_PORT = 5002;

#define PAN_PIN  A1
#define TILT_PIN 9

const int PAN_MIN = 10;
const int PAN_MAX = 170;
const int TILT_MIN = 30;
const int TILT_MAX = 150;
const int PAN_HOME = 90;
const int TILT_HOME = 40;

const int SMOOTH_STEP = 2;
const unsigned long SMOOTH_INTERVAL_MS = 20;

Servo panServo;
Servo tiltServo;

int currentPan = PAN_HOME;
int currentTilt = TILT_HOME;
int targetPan = PAN_HOME;
int targetTilt = TILT_HOME;

WiFiUDP udp;
char buf[64];

IPAddress laptopIP;
bool laptopKnown = false;
unsigned long lastSmoothMs = 0;

int clampAngle(int angle, int minAngle, int maxAngle) {
  if (angle < minAngle) return minAngle;
  if (angle > maxAngle) return maxAngle;
  return angle;
}

int stepToward(int current, int target, int step) {
  if (current == target) return current;

  int delta = (target > current) ? step : -step;
  int next = current + delta;

  if ((delta > 0 && next > target) || (delta < 0 && next < target)) {
    next = target;
  }

  return next;
}

void sendAck() {
  if (!laptopKnown) return;

  char ack[32];
  snprintf(ack, sizeof(ack), "SERVO_ACK,%d,%d", currentPan, currentTilt);
  udp.beginPacket(laptopIP, TELEM_PORT);
  udp.write((const uint8_t*)ack, strlen(ack));
  udp.endPacket();
}

void parseCommand(char* message) {
  if (strncmp(message, "SERVO,", 6) == 0) {
    int pan = 0;
    int tilt = 0;
    if (sscanf(message + 6, "%d,%d", &pan, &tilt) == 2) {
      targetPan = clampAngle(pan, PAN_MIN, PAN_MAX);
      targetTilt = clampAngle(tilt, TILT_MIN, TILT_MAX);
    }
  }
  else if (strcmp(message, "HOME") == 0 || strcmp(message, "STOP") == 0) {
    targetPan = PAN_HOME;
    targetTilt = TILT_HOME;
  }
}

void connectWiFi() {
  Serial.print("Connecting to WiFi: ");
  Serial.println(SSID);

  while (WiFi.status() != WL_CONNECTED) {
    WiFi.begin(SSID, PASSWORD);

    for (int i = 0; i < 20 && WiFi.status() != WL_CONNECTED; i++) {
      delay(500);
      Serial.print(".");
    }
  }

  Serial.println();
  Serial.print("Connected! IP: ");
  Serial.println(WiFi.localIP());
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  panServo.attach(PAN_PIN);
  tiltServo.attach(TILT_PIN);
  panServo.write(PAN_HOME);
  tiltServo.write(TILT_HOME);

  connectWiFi();

  udp.begin(UDP_PORT);
  Serial.print("Listening for servo commands on UDP port ");
  Serial.println(UDP_PORT);
  Serial.println("Commands: SERVO,pan,tilt   HOME   STOP");
}

void loop() {
  int packetSize = udp.parsePacket();
  if (packetSize > 0) {
    int len = udp.read(buf, sizeof(buf) - 1);
    if (len > 0) {
      buf[len] = '\0';
      laptopIP = udp.remoteIP();
      laptopKnown = true;
      parseCommand(buf);
    }
  }

  unsigned long now = millis();
  if (now - lastSmoothMs >= SMOOTH_INTERVAL_MS) {
    lastSmoothMs = now;

    int nextPan = stepToward(currentPan, targetPan, SMOOTH_STEP);
    int nextTilt = stepToward(currentTilt, targetTilt, SMOOTH_STEP);
    bool moved = (nextPan != currentPan) || (nextTilt != currentTilt);

    currentPan = nextPan;
    currentTilt = nextTilt;

    if (moved) {
      panServo.write(currentPan);
      tiltServo.write(currentTilt);
      sendAck();
    }
  }
}

---

## The Arduino Sketch: Servos Over WiFi


---

The Arduino sketch is the servo version of Exercise D's UDP firmware. The WiFi and UDP pattern is the same; the output hardware changes from motors to servos.

### Includes and Constants
Includes, WiFi, and servo pinsArduino
```

```
`Servo.h` handles PWM timing. `WiFiS3.h` and `WiFiUDP.h` are the same UNO R4 WiFi libraries used in Exercise D.

### Servo State
Current angles and target anglesArduino
```

```
The current angle moves toward the target angle by a small step each loop. That keeps the camera from snapping violently to a new direction.

### stepToward()
Smoothing helperArduino
```

```

### Parsing Commands
SERVO, HOME, and STOP parserArduino
```

```

### Acknowledgment
Telemetry replyArduino
```

```
The Arduino learns the laptop IP from the first packet it receives, then sends the current commanded angles back on port `5002`.

---

## Understanding the SERVO Command Protocol


---

Project 4 uses four short text packets. All servo values are integer degrees.
Direction Format Example Python to Arduino SERVO,pan,tilt SERVO,95,42 Python to Arduino HOME HOME Python to Arduino STOP STOP Arduino to Python SERVO_ACK,pan,tilt SERVO_ACK,95,42

### Compared To Exercise D
Exercise D: motorsPython sends `MOTOR,left,right,LEDSTATE`.Arduino sends encoder telemetry with wheel ticks.
Project 4: servosPython sends `SERVO,panAngle,tiltAngle`.Arduino sends the current commanded servo angles.
The protocol is simpler because hobby servos do not report measured position. The ACK confirms command handling and smoothing state, not a measured encoder angle.

---

## Configuration: Your Variables


---

Most setup problems come from configuration values. Start by editing only the configuration block at the top of `pan_and_tilt.py`.
Configuration blockVS Code
```

```
Network`ESP_IP` is the Arduino IP printed in Serial Monitor. `MOBILE_IP` is the phone running the camera app. Ports must match the Arduino sketch.
Vision`TARGET_OBJECT` must exactly match the class label in your trained YOLO model, including capitalization.
Servo limits`PAN_MIN`, `PAN_MAX`, `TILT_MIN`, and `TILT_MAX` protect the mount from hitting its physical stops.
Invert flagsIf the camera moves opposite the target, flip only the matching axis flag and test again.
Dead zonesHorizontal and vertical dead zones ignore tiny errors near the center so the servos do not constantly buzz.
PID gains`KP` sets response strength, `KD` damps motion, and `MAX_*_SPEED` limits degrees per frame.
!!! info "Velocity-mode output"
    In this exercise the PID output is how many degrees to move this frame, not a motor PWM command.

---

## Velocity-Mode PID


---

Project 1 used PID output as a direct motor speed. Project 4 uses velocity mode: the PID output is a change in angle.
Mode Output Accumulated by Used in Position-like motor command Wheel speed Nothing Project 1 Velocity-mode servo command Degrees per frame Adding to current angle Project 4
Velocity mode works well for servos because every frame is a small nudge. The accumulated angle is clamped to the safe physical range after each update.
Velocity-mode updateVS Code
```

```

### Try It In The Browser
Run this simulation and change `kp` to see the angle converge faster or slower.
Velocity-mode PID simulationRun in browser
```

```

---

## The PanTiltTracker Class


---

`PanTiltTracker` has the same structure as the Project 1 tracker: initialize, detect, update state, control.
Method Project 1 equivalent What changed __init__ Tracker.__init__ Two angle PIDs and home angles. _detect Tracker._detect Same YOLO pattern. _update_state Tracker._update_state Same four states. control Tracker.control Pan and tilt errors become servo angle changes.
Error calculation and dead zonesVS Code
```

```
`err_pan = 0` means the ball is horizontally centered. `err_tilt = 0` means the ball is vertically centered. Dead zones prevent constant one-pixel corrections.
Open the complete Python file againComplete pan_and_tilt.pyVS Code"""
pan_and_tilt.py

YOLO ball tracker for a fixed pan-tilt camera rig.

Architecture:
  - A phone streams camera frames over WiFi.
  - YOLO detects the target object in each frame.
  - One PID controls pan from horizontal image error.
  - One PID controls tilt from vertical image error.
  - Each PID outputs degrees per frame, which are added to the current angle.
  - Python sends UDP packets as SERVO,<pan>,<tilt> to the Arduino.
  - The Arduino replies with SERVO_ACK,<pan>,<tilt> for the HUD.

Place this file in the same folder as shared.py and best.pt.
"""

import socket
import threading
import time

import cv2
import numpy as np

from shared import PID, MobileVideoStream, RobotState


# -----------------------------------------------------------------------------
# CONFIGURATION - fill these in for your setup
# -----------------------------------------------------------------------------

# Network
ESP_IP = "192.168.x.x"       # Arduino IP from Serial Monitor
MOBILE_IP = "192.168.x.x"    # Phone IP from IP Webcam or Simple IP Camera
UDP_CMD_PORT = 5001
UDP_TELEM_PORT = 5002

# Vision
TARGET_OBJECT = "ball"

# Servo physical limits in degrees
PAN_MIN, PAN_MAX = 10, 170
TILT_MIN, TILT_MAX = 30, 150
PAN_HOME = 90
TILT_HOME = 40

# Flip these if the camera moves the wrong direction on that axis.
PAN_INVERT = False
TILT_INVERT = False

# Ignore tiny image errors near center to prevent buzzing.
ANGLE_DEAD_ZONE_H = 0.03
ANGLE_DEAD_ZONE_V = 0.01

# PID gains for velocity-mode output in degrees per frame.
PAN_KP = 8.0
PAN_KI = 0.0
PAN_KD = 2.0
MAX_PAN_SPEED = 3.0
MAX_PAN_INTEGRAL = 0.3

TILT_KP = 6.0
TILT_KI = 0.0
TILT_KD = 1.5
MAX_TILT_SPEED = 3.0
MAX_TILT_INTEGRAL = 0.3

# State machine
ACQUIRE_FRAMES = 5
LOST_TIMEOUT = 1.2
SEARCH_PAN_SPEED = 1.5
SEARCH_TILT_SPEED = 0.0
CMD_INTERVAL = 0.03


# -----------------------------------------------------------------------------
# COMMANDER - sends SERVO commands to the Arduino
# -----------------------------------------------------------------------------

class ServoCommander:
    """Sends UDP servo commands to the Arduino."""

    def __init__(self, ip, port):
        self._ip = ip
        self._port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def _send(self, message):
        self._sock.sendto(message.encode("utf-8"), (self._ip, self._port))

    def servos(self, pan, tilt):
        """Send absolute servo angles in integer degrees."""
        self._send(f"SERVO,{int(pan)},{int(tilt)}")

    def home(self):
        """Ask the Arduino to return both servos to their home angles."""
        self._send("HOME")

    def stop(self):
        """Ask the Arduino to home the rig before the Python script exits."""
        self._send("STOP")
        self._sock.close()


# -----------------------------------------------------------------------------
# TELEMETRY - receives optional SERVO_ACK packets from the Arduino
# -----------------------------------------------------------------------------

class ServoTelemetry:
    """
    Receives SERVO_ACK,pan,tilt packets from the Arduino.

    Servos do not provide encoder feedback in this project. The ACK contains the
    Arduino's current commanded angle after its smoothing step.
    """

    def __init__(self, port):
        self.pan = PAN_HOME
        self.tilt = TILT_HOME
        self.running = True
        self._lock = threading.Lock()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("", port))
        self._sock.settimeout(1.0)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.running:
            try:
                data, _ = self._sock.recvfrom(64)
                parts = data.decode("utf-8").strip().split(",")
                if parts[0] == "SERVO_ACK" and len(parts) == 3:
                    with self._lock:
                        self.pan = int(parts[1])
                        self.tilt = int(parts[2])
            except (OSError, socket.timeout, UnicodeDecodeError, ValueError):
                pass

    def read(self):
        with self._lock:
            return {"pan": self.pan, "tilt": self.tilt}

    def stop(self):
        self.running = False
        self._sock.close()


# -----------------------------------------------------------------------------
# PAN-TILT TRACKER
# -----------------------------------------------------------------------------

class PanTiltTracker:
    """YOLO-based target tracker that drives two independent servo axes."""

    STATE_COLORS = {
        RobotState.STOPPED: (100, 100, 100),
        RobotState.SEARCHING: (255, 165, 0),
        RobotState.ACQUIRING: (255, 255, 0),
        RobotState.TRACKING: (0, 255, 0),
    }

    def __init__(self, model_path, target):
        print("Loading YOLO model...")
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.target = target

        try:
            self.model.to("cuda")
            print("  Using CUDA")
        except Exception:
            print("  Using CPU")
        self.model.fuse()

        self.pid_pan = PID(
            PAN_KP,
            PAN_KI,
            PAN_KD,
            max_integral=MAX_PAN_INTEGRAL,
            output_limits=(-MAX_PAN_SPEED, MAX_PAN_SPEED),
        )
        self.pid_tilt = PID(
            TILT_KP,
            TILT_KI,
            TILT_KD,
            max_integral=MAX_TILT_INTEGRAL,
            output_limits=(-MAX_TILT_SPEED, MAX_TILT_SPEED),
        )

        self.pan_angle = float(PAN_HOME)
        self.tilt_angle = float(TILT_HOME)

        self.state = RobotState.STOPPED
        self.acquire_count = 0
        self.last_seen_t = 0.0
        self.last_turn_dir = 1.0
        self.speed_limit = 1.0

    def _detect(self, frame):
        """Return the largest target bounding box as (x1, y1, x2, y2), or None."""
        results = self.model(frame, imgsz=320, verbose=False, conf=0.60)
        best = None
        best_area = 0

        for result in results:
            for box in result.boxes:
                class_name = self.model.names[int(box.cls[0])]
                if class_name != self.target:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best = (x1, y1, x2, y2)

        return best

    def _update_state(self, found):
        if found:
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
            else:
                self.speed_limit = 1.0
            return

        elapsed = time.time() - self.last_seen_t
        if elapsed < LOST_TIMEOUT:
            self.state = RobotState.SEARCHING
            self.speed_limit = 0.4
        else:
            self.state = RobotState.STOPPED
            self.speed_limit = 0.0
            self.pid_pan.reset()
            self.pid_tilt.reset()

    def control(self, frame):
        """
        Run one detection and control cycle.

        Returns:
            pan_angle, tilt_angle, debug_dict
        """
        h, w = frame.shape[:2]
        bbox = self._detect(frame)
        self._update_state(bbox is not None)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            err_pan = (cx - w / 2) / w
            err_tilt = (cy - h / 2) / h

            if abs(err_pan) < ANGLE_DEAD_ZONE_H:
                err_pan = 0.0
            if abs(err_tilt) < ANGLE_DEAD_ZONE_V:
                err_tilt = 0.0

            if err_pan != 0.0:
                self.last_turn_dir = float(np.sign(err_pan))

            pan_sign = -1.0 if PAN_INVERT else 1.0
            tilt_sign = -1.0 if TILT_INVERT else 1.0

            # PID outputs angular velocity in degrees per frame.
            delta_pan = self.pid_pan.update(err_pan) * self.speed_limit * pan_sign
            delta_tilt = self.pid_tilt.update(err_tilt) * self.speed_limit * tilt_sign
            debug_cx, debug_cy = int(cx), int(cy)

        elif self.state == RobotState.SEARCHING:
            if self.pan_angle <= PAN_MIN + 1:
                self.last_turn_dir = 1.0
            elif self.pan_angle >= PAN_MAX - 1:
                self.last_turn_dir = -1.0

            delta_pan = SEARCH_PAN_SPEED * self.last_turn_dir * self.speed_limit
            delta_tilt = SEARCH_TILT_SPEED
            err_pan = 0.0
            err_tilt = 0.0
            debug_cx, debug_cy = w // 2, h // 2

        else:
            delta_pan = 0.0
            delta_tilt = 0.0
            err_pan = 0.0
            err_tilt = 0.0
            debug_cx, debug_cy = w // 2, h // 2

        self.pan_angle = float(np.clip(self.pan_angle + delta_pan, PAN_MIN, PAN_MAX))
        self.tilt_angle = float(np.clip(self.tilt_angle + delta_tilt, TILT_MIN, TILT_MAX))

        debug = {
            "state": self.state,
            "bbox": bbox,
            "err_pan": err_pan,
            "err_tilt": err_tilt,
            "delta_pan": delta_pan,
            "delta_tilt": delta_tilt,
            "pan_angle": self.pan_angle,
            "tilt_angle": self.tilt_angle,
            "speed_limit": self.speed_limit,
            "cx": debug_cx,
            "cy": debug_cy,
        }
        return round(self.pan_angle), round(self.tilt_angle), debug

    def home(self):
        self.pan_angle = float(PAN_HOME)
        self.tilt_angle = float(TILT_HOME)
        self.pid_pan.reset()
        self.pid_tilt.reset()


# -----------------------------------------------------------------------------
# VISUALIZATION
# -----------------------------------------------------------------------------

def draw_overlay(frame, debug, pan, tilt, telem, scale_x=1.0, scale_y=1.0):
    h, w = frame.shape[:2]
    state_color = PanTiltTracker.STATE_COLORS.get(debug["state"], (255, 255, 255))

    cx_frame, cy_frame = w // 2, h // 2
    cv2.line(frame, (cx_frame, 0), (cx_frame, h), (0, 255, 0), 1)
    cv2.line(frame, (0, cy_frame), (w, cy_frame), (0, 255, 0), 1)

    if debug["bbox"] is not None:
        x1, y1, x2, y2 = debug["bbox"]
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
        cx_scaled = int(debug["cx"] * scale_x)
        cy_scaled = int(debug["cy"] * scale_y)

        centered = (
            abs(debug["err_pan"]) < ANGLE_DEAD_ZONE_H
            and abs(debug["err_tilt"]) < ANGLE_DEAD_ZONE_V
        )
        box_color = (0, 255, 0) if centered else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.circle(frame, (cx_scaled, cy_scaled), 6, (0, 0, 255), -1)
        cv2.line(frame, (cx_frame, cy_frame), (cx_scaled, cy_scaled), (0, 200, 255), 1)

    lines = [
        (f"STATE:  {debug['state']}", state_color),
        (f"PAN:    {pan:3d} deg  err:{debug['err_pan']:+.3f}", (0, 255, 255)),
        (f"TILT:   {tilt:3d} deg  err:{debug['err_tilt']:+.3f}", (255, 200, 0)),
        (
            f"dPAN:  {debug['delta_pan']:+.2f} deg/f  "
            f"dTILT:{debug['delta_tilt']:+.2f} deg/f",
            (200, 200, 200),
        ),
        (f"SPD LIM: {debug['speed_limit']:.0%}", (255, 255, 255)),
        (f"TELEM pan:{telem['pan']} deg  tilt:{telem['tilt']} deg", (255, 165, 0)),
    ]

    for i, (text, color) in enumerate(lines):
        bold = i in (0, 1, 2)
        cv2.putText(
            frame,
            text,
            (10, 25 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60 if bold else 0.50,
            color,
            2 if bold else 1,
        )


# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------

def main():
    stream_url = f"http://{MOBILE_IP}:8080/video"
    print(f"Video stream : {stream_url}")
    print(f"ESP target   : {ESP_IP}:{UDP_CMD_PORT}")

    video = MobileVideoStream(stream_url)
    commander = ServoCommander(ESP_IP, UDP_CMD_PORT)
    telemetry = ServoTelemetry(UDP_TELEM_PORT)

    print("Waiting for camera", end="", flush=True)
    for _ in range(30):
        if video.connected:
            break
        time.sleep(0.5)
        print(".", end="", flush=True)
    print()

    if not video.connected:
        print("Could not connect to camera. Check MOBILE_IP and the camera app.")
        video.stop()
        telemetry.stop()
        return

    tracker = PanTiltTracker("best.pt", TARGET_OBJECT)
    commander.home()
    time.sleep(0.5)

    print(f"\nTracking: {TARGET_OBJECT}")
    print("Q = quit   H = home servos   R = reset PIDs")
    print("=" * 60)

    last_fid = -1
    last_cmd_time = 0.0
    fps_count = 0
    fps_timer = time.time()

    try:
        while True:
            frame, fid = video.read()
            if frame is None or fid == last_fid:
                time.sleep(0.005)
                continue

            last_fid = fid
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            fps_count += 1
            if fps_count >= 30:
                elapsed = time.time() - fps_timer
                fps = fps_count / elapsed
                print(
                    f"FPS:{fps:.1f}  pan:{tracker.pan_angle:.1f}  "
                    f"tilt:{tracker.tilt_angle:.1f}  state:{tracker.state}"
                )
                fps_count = 0
                fps_timer = time.time()

            pan_cmd, tilt_cmd, debug = tracker.control(frame)

            now = time.time()
            if now - last_cmd_time >= CMD_INTERVAL:
                commander.servos(pan_cmd, tilt_cmd)
                last_cmd_time = now

            orig_h, orig_w = frame.shape[:2]
            disp = cv2.resize(frame, (640, 480))
            draw_overlay(
                disp,
                debug,
                pan_cmd,
                tilt_cmd,
                telemetry.read(),
                640 / orig_w,
                480 / orig_h,
            )

            cv2.imshow(f"Pan-Tilt Tracker - {TARGET_OBJECT}", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("h"):
                tracker.home()
                commander.home()
                print(f"Servos homed to {PAN_HOME}/{TILT_HOME}")
            elif key == ord("r"):
                tracker.pid_pan.reset()
                tracker.pid_tilt.reset()
                print("PIDs reset")

    finally:
        print("\nShutting down...")
        try:
            commander.stop()
        finally:
            video.stop()
            telemetry.stop()
            cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()

---

## Two PIDs at Once


---

The pan and tilt controllers run independently. The pan PID only sees horizontal error. The tilt PID only sees vertical error.
This works because the problem is separable: moving left-right does not directly solve up-down error, and moving up-down does not directly solve left-right error.
Ball position err_pan err_tilt Pan moves Tilt moves Right of center positive zero right stays Below center zero positive stays down Upper-left negative negative left up Dead center zero zero stays stays
!!! tip "Tune axes separately"
    If pan oscillates, reduce `PAN_KP`. If tilt is sluggish, increase `TILT_KP`. Do not change both axes when only one is misbehaving.

---

## The Main Loop


---

The main loop reads a phone frame, runs the tracker, sends the servo command at a fixed interval, draws the HUD, and handles keyboard commands.
Main loop coreVS Code
```

```

### What Differs From Project 1
Rotation`cv2.rotate` handles a sideways-mounted phone. Remove it if your mount is upright.
Servo command`commander.servos()` replaces motor commands.
Shutdown`commander.stop()` sends `STOP`, which homes both servos.

### Keyboard Commands
Q Quit and home servos. H Home servos immediately. R Reset both PID integrators.

---

## Running and Testing


---

Test the hardware in layers. Do not start with YOLO tracking if the servos have never responded to a simple UDP packet.

### Step 0: Pre-flight
shared.py , best.pt , and pan_and_tilt.py are in the same folder. SSID and PASSWORD are set in the Arduino sketch. Serial Monitor shows the Arduino IP address. The phone stream opens in a browser. ESP_IP , MOBILE_IP , and TARGET_OBJECT are filled in.

### Step 1: Test Servos
Standalone servo testVS Code
```

```

### Step 2: Run The Tracker
Run Project 4Terminal
```

```

### Common Issues
Problem Likely cause Fix Servos do not move Wrong ESP_IP or port Check Serial Monitor and port 5001 . Servo moves wrong direction Invert flag wrong Toggle PAN_INVERT or TILT_INVERT . Centered target jitters Dead zone too small Increase the matching dead zone. Camera shakes Speed or KP too high Reduce MAX_*_SPEED or *_KP . Tracker loses fast targets Speed too low Increase MAX_PAN_SPEED gradually.

---

## Connecting the Dots


---

Project 4 adds servo control and dual-axis PID to the ladder you have already built.
Unit What you built New concept A Custom YOLO detector Training and labels. B Phone stream MJPEG and threaded frame reading. C Encoder motor control Ticks, calibration, closed-loop movement. D WiFi UDP robot control Command packets and telemetry. 1 Full autonomous ball tracker State machine plus all previous infrastructure. 2 Lane follower (classical CV) Thresholding, ROI, and two-point steering without YOLO. 3 Gesture controlled robot MediaPipe hand landmarks mapped to motor commands. 4 Pan-tilt camera tracker Servo control, velocity-mode PID, two independent PIDs.
Velocity-mode PIDThe output is a small change, and the angle accumulates over frames.
Separable axesPan and tilt are independent enough to control with two separate PIDs.
Reusable protocol designChanging from `MOTOR` to `SERVO` reuses the same UDP infrastructure.
!!! tip "Robotics connection"
    The same architecture appears in security cameras, telescope mounts, drone gimbals, and target-tracking camera systems: detect the target, compute image error, update angles, repeat.

[Download](original/shared.py)

[Download](original/pan_and_tilt.py)

[Download](original/pan_tilt_arduino.ino)

[Download](original/test_servos.py)

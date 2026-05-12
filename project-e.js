(() => {
  if (window.__PROJECT_E_INSTALLED__) return;
  window.__PROJECT_E_INSTALLED__ = true;

  const ACCENT = "#06b6d4";
  const CHAPTER_TITLE = "* Project E: Pan-Tilt Ball Tracker";

  const projectELessons = [
    { id: "projE-overview", label: "PE.0 What You Are Building", title: "PE.0 What You Are Building", exercise: true },
    { id: "projE-servo", label: "PE.1 Servo Motors", title: "PE.1 What Is a Servo Motor?", exercise: false },
    { id: "projE-mechanism", label: "PE.2 Pan-Tilt Mechanism", title: "PE.2 The Pan-Tilt Mechanism", exercise: false },
    { id: "projE-get-code", label: "PE.3 Get the Code", title: "PE.3 Get the Code", exercise: false },
    { id: "projE-arduino", label: "PE.4 Arduino Sketch", title: "PE.4 The Arduino Sketch: Servos Over WiFi", exercise: false },
    { id: "projE-protocol", label: "PE.5 SERVO Protocol", title: "PE.5 Understanding the SERVO Command Protocol", exercise: false },
    { id: "projE-config", label: "PE.6 Configuration", title: "PE.6 Configuration: Your Variables", exercise: false },
    { id: "projE-velocity-pid", label: "PE.7 Velocity PID", title: "PE.7 Velocity-Mode PID", exercise: false },
    { id: "projE-tracker-class", label: "PE.8 PanTiltTracker", title: "PE.8 The PanTiltTracker Class", exercise: false },
    { id: "projE-dual-pid", label: "PE.9 Two PIDs", title: "PE.9 Two PIDs at Once", exercise: false },
    { id: "projE-main-loop", label: "PE.10 Main Loop", title: "PE.10 The Main Loop", exercise: false },
    { id: "projE-running", label: "PE.11 Running", title: "PE.11 Running and Testing", exercise: false },
    { id: "projE-dots", label: "PE.12 Connecting Dots", title: "PE.12 Connecting the Dots", exercise: false },
  ].map((lesson) => ({
    ...lesson,
    chapter: "projE",
    chapterTitle: CHAPTER_TITLE,
  }));

  function lessonLink(lesson) {
    const star = lesson.exercise ? "* " : "";
    return `<a class="lesson-link${lesson.exercise ? " is-exercise" : ""}" href="?lesson=${lesson.id}" data-lesson-link="${lesson.id}"><span>${star}${lesson.label}</span><span class="completion-dot" aria-hidden="true"></span></a>`;
  }

  function codeBlock(filename, lang, code, options = {}) {
    const runnable = options.runnable ? "true" : "false";
    const chapter = options.chapter || "projE";
    const download = options.download ? ` data-download-filename="${filename}"` : "";
    const session = options.session ? ` data-session="${options.session}"` : "";
    const langLabel = options.langLabel || (lang === "cpp" || lang === "arduino" ? "Arduino" : "");
    const langLabelAttr = langLabel ? ` data-lang-label="${langLabel}"` : "";
    return `<figure class="code-card" data-runnable="${runnable}" data-chapter="${chapter}"${session} data-filename="${filename}"${download} data-lang="${lang}"${langLabelAttr}><figcaption><span>${options.caption || filename}</span>${options.badge ? `<span class="local-run-badge">${options.badge}</span>` : ""}</figcaption><script type="text/plain" class="code-source">${code.replace(/<\/script/gi, "<\\/script")}</script></figure>`;
  }

  function sourceFileBlock(filename, lang, caption, badge) {
    const langLabel = lang === "cpp" || lang === "arduino" ? "Arduino" : "";
    const langLabelAttr = langLabel ? ` data-lang-label="${langLabel}"` : "";
    return `<figure class="code-card pe-source-code" data-source-file="${filename}" data-runnable="false" data-chapter="projE" data-filename="${filename}" data-download-filename="${filename}" data-lang="${lang}"${langLabelAttr}><figcaption><span>${caption}</span><span class="local-run-badge">${badge}</span></figcaption><script type="text/plain" class="code-source"></script></figure>`;
  }

  function nav(prev, next) {
    const prevHtml = prev
      ? `<a class="lesson-nav-btn" href="?lesson=${prev.id}" data-lesson-link="${prev.id}"><span class="lesson-nav-label">Previous</span><span class="lesson-nav-title">&larr; ${prev.label}</span></a>`
      : "<span></span>";
    const nextHtml = next
      ? `<a class="lesson-nav-btn next" href="?lesson=${next.id}" data-lesson-link="${next.id}"><span class="lesson-nav-label">Next</span><span class="lesson-nav-title">${next.label} &rarr;</span></a>`
      : "<span></span>";
    return `<nav class="lesson-nav" aria-label="Previous and next lessons">${prevHtml}${nextHtml}</nav>`;
  }

  function lessonArticle(index, body, meta) {
    const lesson = projectELessons[index];
    const prev = index === 0 ? { id: "projD-project1", label: "PD.7 Project 1 Bridge" } : projectELessons[index - 1];
    const next = index === projectELessons.length - 1 ? { id: "proj1-overview", label: "Project 1" } : projectELessons[index + 1];
    return `
<article class="lesson-content${lesson.exercise ? " exercise-lesson" : ""}" data-lesson="${lesson.id}" data-chapter="projE" data-chapter-title="${CHAPTER_TITLE}" style="--chapter-accent:${ACCENT};--chapter-color:${ACCENT}">
<div class="lesson-kicker">${CHAPTER_TITLE} <span>&rsaquo;</span> ${lesson.title}</div>
<h1>${lesson.title}</h1>
<div class="lesson-meta">${meta.map((item) => `<span>${item}</span>`).join("")}</div>
<hr class="lesson-divider" />
<div class="lesson-body">${body}</div>
<div class="lesson-actions"><button class="mark-complete-btn" data-lesson="${lesson.id}">Mark as Complete</button></div>
${nav(prev, next)}
</article>`;
  }

  const configSnippet = `# Network
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
TILT_INVERT = False`;

  const velocitySnippet = `# PID outputs angular velocity in degrees per frame.
delta_pan = self.pid_pan.update(err_pan) * self.speed_limit * pan_sign
delta_tilt = self.pid_tilt.update(err_tilt) * self.speed_limit * tilt_sign

# Integrate: add the delta to the current angle.
self.pan_angle = float(np.clip(self.pan_angle + delta_pan, PAN_MIN, PAN_MAX))
self.tilt_angle = float(np.clip(self.tilt_angle + delta_tilt, TILT_MIN, TILT_MAX))`;

  const velocityDemo = `import numpy as np

pan_angle = 90.0
kp = 8.0
speed_limit = 0.4
ball_positions = [0.2] * 20

print(f"{'Frame':>5}  {'err_pan':>8}  {'delta':>7}  {'angle':>7}")
print("-" * 35)

for frame, ball_x in enumerate(ball_positions):
    err_pan = ball_x
    delta = kp * err_pan * speed_limit
    pan_angle = np.clip(pan_angle + delta, 10, 170)
    print(f"{frame:>5}  {err_pan:>8.3f}  {delta:>7.3f}  {pan_angle:>7.1f}")`;

  const testServos = `import socket, time

ESP_IP = "YOUR_ARDUINO_IP"
CMD_PORT = 5001

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send(msg):
    sock.sendto(msg.encode(), (ESP_IP, CMD_PORT))
    print(f"Sent: {msg}")
    time.sleep(1.0)

send("HOME")
send("SERVO,45,40")
send("SERVO,135,40")
send("SERVO,90,80")
send("SERVO,90,40")
send("HOME")

sock.close()`;

  const articles = [
    lessonArticle(0, `
<p class="lead">In Projects 1 and 2 the camera was mounted on a robot that moved its whole body to follow a target. This project keeps the robot fixed and moves only the camera with two small servos.</p>
<p>A pan-tilt mount is the same idea used in tracking security cameras, telescope mounts, and camera gimbals. The system has one camera, two servo motors, one YOLO model, and two PID controllers running at the same time.</p>
<div class="diagram pe-system-diagram" aria-label="Pan tilt tracking system diagram">
  <div class="pe-system-row"><div class="flow-node">Phone Camera<br /><span class="mini muted">MJPEG over WiFi</span></div><div class="arrow">&rarr;</div><div class="flow-node">Laptop Python<br /><span class="mini muted">YOLO finds cx, cy</span></div></div>
  <div class="pe-system-row"><div class="flow-node">PAN PID<br /><span class="mini muted">horizontal error to delta angle</span></div><div class="flow-node">TILT PID<br /><span class="mini muted">vertical error to delta angle</span></div></div>
  <div class="pe-system-row"><div class="flow-node">UDP command<br /><code>SERVO,pan,tilt</code></div><div class="arrow">&rarr;</div><div class="flow-node">Arduino UNO R4 WiFi<br /><span class="mini muted">drives pan and tilt servos</span></div></div>
  <div class="pe-return-path">Arduino replies with <code>SERVO_ACK,current_pan,current_tilt</code>.</div>
</div>
<h3>What Is New</h3>
<div class="table-wrap"><table class="data-table"><thead><tr><th>Concept</th><th>Where you saw it</th><th>What is new here</th></tr></thead><tbody>
<tr><td>YOLO detection</td><td>Project A, Project 1</td><td>Same model call.</td></tr>
<tr><td>Phone stream</td><td>Project B</td><td>Same <code>MobileVideoStream</code>.</td></tr>
<tr><td>UDP communication</td><td>Project D</td><td>New packet: <code>SERVO,pan,tilt</code>.</td></tr>
<tr><td>PID controller</td><td>Project 1</td><td>Velocity mode: add the output to an angle.</td></tr>
<tr><td>DC motors</td><td>Projects C and D</td><td>Replaced by servo motors.</td></tr>
</tbody></table></div>`, ["Project", "YOLO + servos", "10 min"]),

    lessonArticle(1, `
<p class="lead">A DC motor spins continuously when powered. A servo motor moves to a commanded angle and holds that position.</p>
<p>A servo has a built-in position sensor and control circuit. You tell it <code>servo.write(90)</code>, and the Arduino Servo library generates the timing signal that makes the motor move to 90 degrees.</p>
<div class="pe-timing-grid">
  <div class="card"><h4>1 ms pulse</h4><p>Servo goes near 0 degrees.</p></div>
  <div class="card"><h4>1.5 ms pulse</h4><p>Servo goes near 90 degrees.</p></div>
  <div class="card"><h4>2 ms pulse</h4><p>Servo goes near 180 degrees.</p></div>
</div>
<div class="callout info" role="note"><div class="callout-icon">i</div><div><div class="callout-title">Servo vs DC motor</div><p>Use a servo when you need position control. Use a DC motor when you need continuous rotation. A camera mount needs to point at a specific angle, so servos are the correct choice.</p></div></div>
<h3>Servo Angle Range</h3>
<p>Most hobby servos rotate from 0 to 180 degrees. This project clamps pan to <code>10..170</code> and tilt to <code>30..150</code> to keep the mount and camera cable away from mechanical limits.</p>
<div class="callout warning" role="alert"><div class="callout-icon">!</div><div><div class="callout-title">Never command past the physical stop</div><p>The Arduino <code>clampAngle()</code> function and Python <code>np.clip()</code> call both exist to prevent overheating a stalled servo.</p></div></div>`, ["Hardware", "Servo basics", "12 min"]),

    lessonArticle(2, `
<p class="lead">A pan-tilt mechanism is two servo motors mounted at 90 degrees to each other. One controls left-right pan. The other controls up-down tilt.</p>
<div class="grid-2">
<article class="card"><h3>Top view</h3><div class="pe-axis-demo pe-pan-axis"><span>left</span><strong>Pan servo</strong><span>right</span></div><p>The pan axis rotates the camera horizontally.</p></article>
<article class="card"><h3>Side view</h3><div class="pe-axis-demo pe-tilt-axis"><span>up</span><strong>Tilt servo</strong><span>down</span></div><p>The tilt axis raises and lowers the camera.</p></article>
</div>
<h3>Home Position</h3>
<p><code>PAN_HOME = 90</code> points straight ahead. <code>TILT_HOME = 40</code> points slightly downward, useful when the ball is on a table or floor.</p>
<div class="callout tip" role="note"><div class="callout-icon">OK</div><div><div class="callout-title">First upload test</div><p>When the Arduino sketch starts, both servos immediately move to home. That confirms power, signal wiring, and pin choices before Python runs.</p></div></div>
<h3>Inversion Variables</h3>
<p>If the camera moves the wrong way, change the matching Python flag: <code>PAN_INVERT</code> or <code>TILT_INVERT</code>. No rewiring is required.</p>`, ["Mechanism", "Mounting", "10 min"]),

    lessonArticle(3, `
<p class="lead">Place these files in the same folder as <code>shared.py</code> and <code>best.pt</code>. Open the Arduino sketch in Arduino IDE and upload it to the UNO R4 WiFi.</p>
<div class="grid-2">
<article class="card download-card"><h3>shared.py</h3><p>Already used in Projects 1 and 2. Keep it beside the tracker script.</p><a class="download-card-btn" href="shared.py" download>Download</a></article>
<article class="card download-card"><h3>pan_and_tilt.py</h3><p>The Python YOLO tracker and servo commander.</p><a class="download-card-btn" href="pan_and_tilt.py" download>Download</a></article>
<article class="card download-card"><h3>pan_tilt_arduino.ino</h3><p>The Arduino servo-over-WiFi sketch.</p><a class="download-card-btn" href="pan_tilt_arduino.ino" download>Download</a></article>
<article class="card download-card"><h3>test_servos.py</h3><p>A quick command-only test before running YOLO.</p><a class="download-card-btn" href="test_servos.py" download>Download</a></article>
</div>
<h3>Values You Must Fill In</h3>
<div class="table-wrap"><table class="data-table"><thead><tr><th>File</th><th>Variable</th><th>Meaning</th></tr></thead><tbody>
<tr><td><code>pan_and_tilt.py</code></td><td><code>ESP_IP</code></td><td>Arduino IP from Serial Monitor.</td></tr>
<tr><td><code>pan_and_tilt.py</code></td><td><code>MOBILE_IP</code></td><td>Phone camera app IP address.</td></tr>
<tr><td><code>pan_and_tilt.py</code></td><td><code>TARGET_OBJECT</code></td><td>Your YOLO class name, usually <code>ball</code>.</td></tr>
<tr><td><code>pan_and_tilt.py</code></td><td><code>PAN_INVERT</code>, <code>TILT_INVERT</code></td><td>Set to <code>True</code> only if an axis moves backward.</td></tr>
<tr><td><code>pan_tilt_arduino.ino</code></td><td><code>SSID</code>, <code>PASSWORD</code></td><td>Your WiFi network name and password.</td></tr>
</tbody></table></div>
<div class="callout warning" role="alert"><div class="callout-icon">!</div><div><div class="callout-title">Fill in WiFi first</div><p>If <code>SSID</code> or <code>PASSWORD</code> is wrong, the Arduino will never print an IP address and Python cannot reach it.</p></div></div>
<details class="solution"><summary>Complete Python file - loads from <code>pan_and_tilt.py</code></summary>${sourceFileBlock("pan_and_tilt.py", "python", "Complete pan_and_tilt.py", "VS Code")}</details>
<details class="solution"><summary>Complete Arduino sketch - loads from <code>pan_tilt_arduino.ino</code></summary>${sourceFileBlock("pan_tilt_arduino.ino", "cpp", "Complete pan_tilt_arduino.ino", "Arduino")}</details>`, ["Code", "Downloads", "15 min"]),

    lessonArticle(4, `
<p class="lead">The Arduino sketch is the servo version of Project D's UDP firmware. The WiFi and UDP pattern is the same; the output hardware changes from motors to servos.</p>
<h3>Includes and Constants</h3>
${codeBlock("pan_tilt_arduino.ino", "cpp", `#include <Servo.h>
#include <WiFiS3.h>
#include <WiFiUDP.h>

const char* SSID = "YOUR_WIFI_NAME";
const char* PASSWORD = "YOUR_WIFI_PASSWORD";
const int UDP_PORT = 5001;
const int TELEM_PORT = 5002;

#define PAN_PIN  A1
#define TILT_PIN 9`, { caption: "Includes, WiFi, and servo pins", badge: "Arduino" })}
<p><code>Servo.h</code> handles PWM timing. <code>WiFiS3.h</code> and <code>WiFiUDP.h</code> are the same UNO R4 WiFi libraries used in Project D.</p>
<h3>Servo State</h3>
${codeBlock("pan_tilt_arduino.ino", "cpp", `Servo panServo;
Servo tiltServo;

int currentPan = PAN_HOME;
int currentTilt = TILT_HOME;
int targetPan = PAN_HOME;
int targetTilt = TILT_HOME;`, { caption: "Current angles and target angles", badge: "Arduino" })}
<p>The current angle moves toward the target angle by a small step each loop. That keeps the camera from snapping violently to a new direction.</p>
<h3>stepToward()</h3>
${codeBlock("pan_tilt_arduino.ino", "cpp", `int stepToward(int current, int target, int step) {
  if (current == target) return current;

  int delta = (target > current) ? step : -step;
  int next = current + delta;

  if ((delta > 0 && next > target) || (delta < 0 && next < target)) {
    next = target;
  }

  return next;
}`, { caption: "Smoothing helper", badge: "Arduino" })}
<h3>Parsing Commands</h3>
${codeBlock("pan_tilt_arduino.ino", "cpp", `if (strncmp(message, "SERVO,", 6) == 0) {
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
}`, { caption: "SERVO, HOME, and STOP parser", badge: "Arduino" })}
<h3>Acknowledgment</h3>
${codeBlock("pan_tilt_arduino.ino", "cpp", `snprintf(ack, sizeof(ack), "SERVO_ACK,%d,%d", currentPan, currentTilt);
udp.beginPacket(laptopIP, TELEM_PORT);
udp.write((const uint8_t*)ack, strlen(ack));
udp.endPacket();`, { caption: "Telemetry reply", badge: "Arduino" })}
<p>The Arduino learns the laptop IP from the first packet it receives, then sends the current commanded angles back on port <code>5002</code>.</p>`, ["Arduino", "Firmware", "25 min"]),

    lessonArticle(5, `
<p class="lead">Project E uses four short text packets. All servo values are integer degrees.</p>
<div class="table-wrap"><table class="data-table"><thead><tr><th>Direction</th><th>Format</th><th>Example</th></tr></thead><tbody>
<tr><td>Python to Arduino</td><td><code>SERVO,pan,tilt</code></td><td><code>SERVO,95,42</code></td></tr>
<tr><td>Python to Arduino</td><td><code>HOME</code></td><td><code>HOME</code></td></tr>
<tr><td>Python to Arduino</td><td><code>STOP</code></td><td><code>STOP</code></td></tr>
<tr><td>Arduino to Python</td><td><code>SERVO_ACK,pan,tilt</code></td><td><code>SERVO_ACK,95,42</code></td></tr>
</tbody></table></div>
<h3>Compared To Project D</h3>
<div class="grid-2">
<article class="card"><h3>Project D: motors</h3><p>Python sends <code>MOTOR,left,right,LEDSTATE</code>.</p><p>Arduino sends encoder telemetry with wheel ticks.</p></article>
<article class="card"><h3>Project E: servos</h3><p>Python sends <code>SERVO,panAngle,tiltAngle</code>.</p><p>Arduino sends the current commanded servo angles.</p></article>
</div>
<p>The protocol is simpler because hobby servos do not report measured position. The ACK confirms command handling and smoothing state, not a measured encoder angle.</p>`, ["Protocol", "UDP packets", "10 min"]),

    lessonArticle(6, `
<p class="lead">Most setup problems come from configuration values. Start by editing only the configuration block at the top of <code>pan_and_tilt.py</code>.</p>
${codeBlock("pan_and_tilt.py", "python", configSnippet, { caption: "Configuration block", badge: "VS Code" })}
<div class="grid-2">
<article class="card"><h3>Network</h3><p><code>ESP_IP</code> is the Arduino IP printed in Serial Monitor. <code>MOBILE_IP</code> is the phone running the camera app. Ports must match the Arduino sketch.</p></article>
<article class="card"><h3>Vision</h3><p><code>TARGET_OBJECT</code> must exactly match the class label in your trained YOLO model, including capitalization.</p></article>
<article class="card"><h3>Servo limits</h3><p><code>PAN_MIN</code>, <code>PAN_MAX</code>, <code>TILT_MIN</code>, and <code>TILT_MAX</code> protect the mount from hitting its physical stops.</p></article>
<article class="card"><h3>Invert flags</h3><p>If the camera moves opposite the target, flip only the matching axis flag and test again.</p></article>
<article class="card"><h3>Dead zones</h3><p>Horizontal and vertical dead zones ignore tiny errors near the center so the servos do not constantly buzz.</p></article>
<article class="card"><h3>PID gains</h3><p><code>KP</code> sets response strength, <code>KD</code> damps motion, and <code>MAX_*_SPEED</code> limits degrees per frame.</p></article>
</div>
<div class="callout info" role="note"><div class="callout-icon">PID</div><div><div class="callout-title">Velocity-mode output</div><p>In this project the PID output is how many degrees to move this frame, not a motor PWM command.</p></div></div>`, ["Configuration", "Tuning", "18 min"]),

    lessonArticle(7, `
<p class="lead">Project 1 used PID output as a direct motor speed. Project E uses velocity mode: the PID output is a change in angle.</p>
<div class="table-wrap"><table class="data-table"><thead><tr><th>Mode</th><th>Output</th><th>Accumulated by</th><th>Used in</th></tr></thead><tbody>
<tr><td>Position-like motor command</td><td>Wheel speed</td><td>Nothing</td><td>Project 1</td></tr>
<tr><td>Velocity-mode servo command</td><td>Degrees per frame</td><td>Adding to current angle</td><td>Project E</td></tr>
</tbody></table></div>
<p>Velocity mode works well for servos because every frame is a small nudge. The accumulated angle is clamped to the safe physical range after each update.</p>
${codeBlock("pan_and_tilt.py", "python", velocitySnippet, { caption: "Velocity-mode update", badge: "VS Code" })}
<h3>Try It In The Browser</h3>
<p>Run this simulation and change <code>kp</code> to see the angle converge faster or slower.</p>
${codeBlock("velocity_mode_demo.py", "python", velocityDemo, { runnable: true, chapter: "projE", session: "projE-velocity", caption: "Velocity-mode PID simulation", badge: "Run in browser" })}`, ["PID", "Browser demo", "16 min"]),

    lessonArticle(8, `
<p class="lead"><code>PanTiltTracker</code> has the same structure as the Project 1 tracker: initialize, detect, update state, control.</p>
<div class="table-wrap"><table class="data-table"><thead><tr><th>Method</th><th>Project 1 equivalent</th><th>What changed</th></tr></thead><tbody>
<tr><td><code>__init__</code></td><td><code>Tracker.__init__</code></td><td>Two angle PIDs and home angles.</td></tr>
<tr><td><code>_detect</code></td><td><code>Tracker._detect</code></td><td>Same YOLO pattern.</td></tr>
<tr><td><code>_update_state</code></td><td><code>Tracker._update_state</code></td><td>Same four states.</td></tr>
<tr><td><code>control</code></td><td><code>Tracker.control</code></td><td>Pan and tilt errors become servo angle changes.</td></tr>
</tbody></table></div>
${codeBlock("pan_and_tilt.py", "python", `err_pan = (cx - w / 2) / w
err_tilt = (cy - h / 2) / h

if abs(err_pan) < ANGLE_DEAD_ZONE_H:
    err_pan = 0.0
if abs(err_tilt) < ANGLE_DEAD_ZONE_V:
    err_tilt = 0.0`, { caption: "Error calculation and dead zones", badge: "VS Code" })}
<p><code>err_pan = 0</code> means the ball is horizontally centered. <code>err_tilt = 0</code> means the ball is vertically centered. Dead zones prevent constant one-pixel corrections.</p>
<details class="solution"><summary>Open the complete Python file again</summary>${sourceFileBlock("pan_and_tilt.py", "python", "Complete pan_and_tilt.py", "VS Code")}</details>`, ["Python class", "YOLO control", "20 min"]),

    lessonArticle(9, `
<p class="lead">The pan and tilt controllers run independently. The pan PID only sees horizontal error. The tilt PID only sees vertical error.</p>
<p>This works because the problem is separable: moving left-right does not directly solve up-down error, and moving up-down does not directly solve left-right error.</p>
<div class="table-wrap"><table class="data-table"><thead><tr><th>Ball position</th><th>err_pan</th><th>err_tilt</th><th>Pan moves</th><th>Tilt moves</th></tr></thead><tbody>
<tr><td>Right of center</td><td>positive</td><td>zero</td><td>right</td><td>stays</td></tr>
<tr><td>Below center</td><td>zero</td><td>positive</td><td>stays</td><td>down</td></tr>
<tr><td>Upper-left</td><td>negative</td><td>negative</td><td>left</td><td>up</td></tr>
<tr><td>Dead center</td><td>zero</td><td>zero</td><td>stays</td><td>stays</td></tr>
</tbody></table></div>
<div class="callout tip" role="note"><div class="callout-icon">Tune</div><div><div class="callout-title">Tune axes separately</div><p>If pan oscillates, reduce <code>PAN_KP</code>. If tilt is sluggish, increase <code>TILT_KP</code>. Do not change both axes when only one is misbehaving.</p></div></div>`, ["Control", "Dual PID", "12 min"]),

    lessonArticle(10, `
<p class="lead">The main loop reads a phone frame, runs the tracker, sends the servo command at a fixed interval, draws the HUD, and handles keyboard commands.</p>
${codeBlock("pan_and_tilt.py", "python", `frame, fid = video.read()
if frame is None or fid == last_fid:
    time.sleep(0.005)
    continue

last_fid = fid
frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

pan_cmd, tilt_cmd, debug = tracker.control(frame)

now = time.time()
if now - last_cmd_time >= CMD_INTERVAL:
    commander.servos(pan_cmd, tilt_cmd)
    last_cmd_time = now`, { caption: "Main loop core", badge: "VS Code" })}
<h3>What Differs From Project 1</h3>
<div class="grid-3">
<article class="card"><h3>Rotation</h3><p><code>cv2.rotate</code> handles a sideways-mounted phone. Remove it if your mount is upright.</p></article>
<article class="card"><h3>Servo command</h3><p><code>commander.servos()</code> replaces motor commands.</p></article>
<article class="card"><h3>Shutdown</h3><p><code>commander.stop()</code> sends <code>STOP</code>, which homes both servos.</p></article>
</div>
<h3>Keyboard Commands</h3>
<div class="table-wrap"><table class="data-table"><tbody><tr><td><code>Q</code></td><td>Quit and home servos.</td></tr><tr><td><code>H</code></td><td>Home servos immediately.</td></tr><tr><td><code>R</code></td><td>Reset both PID integrators.</td></tr></tbody></table></div>`, ["Main loop", "OpenCV", "14 min"]),

    lessonArticle(11, `
<p class="lead">Test the hardware in layers. Do not start with YOLO tracking if the servos have never responded to a simple UDP packet.</p>
<h3>Step 0: Pre-flight</h3>
<div class="steps project-checklist">
<div class="step"><label><input type="checkbox" data-checklist-key="projE-shared" /> <code>shared.py</code>, <code>best.pt</code>, and <code>pan_and_tilt.py</code> are in the same folder.</label></div>
<div class="step"><label><input type="checkbox" data-checklist-key="projE-wifi-creds" /> <code>SSID</code> and <code>PASSWORD</code> are set in the Arduino sketch.</label></div>
<div class="step"><label><input type="checkbox" data-checklist-key="projE-arduino-ip" /> Serial Monitor shows the Arduino IP address.</label></div>
<div class="step"><label><input type="checkbox" data-checklist-key="projE-phone-ip" /> The phone stream opens in a browser.</label></div>
<div class="step"><label><input type="checkbox" data-checklist-key="projE-python-config" /> <code>ESP_IP</code>, <code>MOBILE_IP</code>, and <code>TARGET_OBJECT</code> are filled in.</label></div>
</div>
<h3>Step 1: Test Servos</h3>
${codeBlock("test_servos.py", "python", testServos, { caption: "Standalone servo test", badge: "VS Code", download: true })}
<h3>Step 2: Run The Tracker</h3>
${codeBlock("terminal", "text", `python pan_and_tilt.py`, { caption: "Run Project E", badge: "Terminal" })}
<h3>Common Issues</h3>
<div class="table-wrap"><table class="data-table"><thead><tr><th>Problem</th><th>Likely cause</th><th>Fix</th></tr></thead><tbody>
<tr><td>Servos do not move</td><td>Wrong <code>ESP_IP</code> or port</td><td>Check Serial Monitor and port <code>5001</code>.</td></tr>
<tr><td>Servo moves wrong direction</td><td>Invert flag wrong</td><td>Toggle <code>PAN_INVERT</code> or <code>TILT_INVERT</code>.</td></tr>
<tr><td>Centered target jitters</td><td>Dead zone too small</td><td>Increase the matching dead zone.</td></tr>
<tr><td>Camera shakes</td><td>Speed or KP too high</td><td>Reduce <code>MAX_*_SPEED</code> or <code>*_KP</code>.</td></tr>
<tr><td>Tracker loses fast targets</td><td>Speed too low</td><td>Increase <code>MAX_PAN_SPEED</code> gradually.</td></tr>
</tbody></table></div>`, ["Run locally", "Testing", "18 min"]),

    lessonArticle(12, `
<p class="lead">Project E adds servo control and dual-axis PID to the ladder you have already built.</p>
<div class="table-wrap"><table class="data-table"><thead><tr><th>Project</th><th>What you built</th><th>New concept</th></tr></thead><tbody>
<tr><td>A</td><td>Custom YOLO detector</td><td>Training and labels.</td></tr>
<tr><td>B</td><td>Phone stream</td><td>MJPEG and threaded frame reading.</td></tr>
<tr><td>C</td><td>Encoder motor control</td><td>Ticks, calibration, closed-loop movement.</td></tr>
<tr><td>D</td><td>WiFi UDP robot control</td><td>Command packets and telemetry.</td></tr>
<tr><td>E</td><td>Pan-tilt camera tracker</td><td>Servo control, velocity-mode PID, two independent PIDs.</td></tr>
<tr><td>1</td><td>Full autonomous ball tracker</td><td>State machine plus all previous infrastructure.</td></tr>
</tbody></table></div>
<div class="grid-3">
<article class="card"><h3>Velocity-mode PID</h3><p>The output is a small change, and the angle accumulates over frames.</p></article>
<article class="card"><h3>Separable axes</h3><p>Pan and tilt are independent enough to control with two separate PIDs.</p></article>
<article class="card"><h3>Reusable protocol design</h3><p>Changing from <code>MOTOR</code> to <code>SERVO</code> reuses the same UDP infrastructure.</p></article>
</div>
<div class="callout robotics" role="note"><div class="callout-icon">BOT</div><div><div class="callout-title">Robotics connection</div><p>The same architecture appears in security cameras, telescope mounts, drone gimbals, and target-tracking camera systems: detect the target, compute image error, update angles, repeat.</p></div></div>`, ["Summary", "Architecture", "10 min"]),
  ].join("");

  function insertDom() {
    const sidebarProjectD = document.querySelector('.lesson-group[data-chapter="projD"]');
    if (sidebarProjectD && !document.querySelector('.lesson-group[data-chapter="projE"]')) {
      sidebarProjectD.insertAdjacentHTML("afterend", `
<div class="lesson-group" data-chapter="projE" style="--chapter-accent:${ACCENT}">
<button class="lesson-group-header" type="button" aria-expanded="false"><span>${CHAPTER_TITLE}</span><span class="group-chevron">v</span></button>
<div class="lesson-group-body">${projectELessons.map(lessonLink).join("")}</div>
</div>`);
    }

    const dashboardProjectD = document.querySelector('[data-chapter-card="projD"]');
    if (dashboardProjectD && !document.querySelector('[data-chapter-card="projE"]')) {
      dashboardProjectD.insertAdjacentHTML("afterend", `
<article class="chapter-card" data-chapter-card="projE" style="--chapter-accent:${ACCENT}">
<div class="chapter-card-strip"></div>
<div class="chapter-card-top"><span>Project E</span><span>13 lessons</span></div>
<h2>Pan-Tilt Ball Tracker</h2>
<p>Use YOLO and two velocity-mode PID loops to center a ball by rotating a camera with servos.</p>
<div class="chapter-card-progress"><div class="chapter-card-bar"><span></span></div><small data-card-progress="projE">0 / 13 complete</small></div>
<a class="chapter-start-btn" href="?lesson=projE-overview" data-lesson-link="projE-overview">Start</a>
</article>`);
    }

    const projectDLast = document.querySelector('[data-lesson="projD-project1"]');
    if (projectDLast && !document.querySelector('[data-lesson="projE-overview"]')) {
      projectDLast.insertAdjacentHTML("afterend", articles);
    }

    const projectDNext = projectDLast?.querySelector(".lesson-nav-btn.next");
    if (projectDNext) {
      projectDNext.href = "?lesson=projE-overview";
      projectDNext.dataset.lessonLink = "projE-overview";
      const title = projectDNext.querySelector(".lesson-nav-title");
      if (title) title.innerHTML = "Project E &rarr;";
    }

    const project1 = document.querySelector('[data-lesson="proj1-overview"]');
    const project1Prev = project1?.querySelector(".lesson-nav-btn:not(.next)");
    if (project1Prev) {
      project1Prev.href = "?lesson=projE-dots";
      project1Prev.dataset.lessonLink = "projE-dots";
      const title = project1Prev.querySelector(".lesson-nav-title");
      if (title) title.innerHTML = "&larr; PE.12 Connecting Dots";
    }

    const archiveProjectD = document.querySelector('[data-lesson="project-archive"] a[data-lesson-link="projD-udp"]');
    if (archiveProjectD && !document.querySelector('[data-lesson="project-archive"] a[data-lesson-link="projE-overview"]')) {
      archiveProjectD.insertAdjacentHTML("afterend", `
<a class="card" href="index.html?lesson=projE-overview" data-lesson-link="projE-overview" style="display:block;text-decoration:none;border-color:${ACCENT};margin-bottom:1rem">
<div class="project-label">Project E &mdash; Pan-Tilt Ball Tracker</div>
<h3>Two Servos, One Camera, One Neural Network</h3>
<p class="muted">A fixed camera rig that tracks a ball with YOLO, UDP servo commands, velocity-mode PID, and two independent pan/tilt control loops.</p>
<div class="chip-row" style="--chapter-color:${ACCENT}"><span class="chip">YOLO</span><span class="chip">Servo Motors</span><span class="chip">Pan-Tilt</span><span class="chip">Velocity PID</span><span class="chip">UNO R4 WiFi</span></div>
<strong>Open Project E lessons &rarr;</strong>
</a>`);
    }
  }

  function insertMetadata() {
    if (!window.COURSE_LESSONS?.some((lesson) => lesson.id === "projE-overview")) {
      const beforeProject1 = window.COURSE_LESSONS.findIndex((lesson) => lesson.id === "proj1-overview");
      const insertAt = beforeProject1 >= 0 ? beforeProject1 : window.COURSE_LESSONS.length;
      window.COURSE_LESSONS.splice(insertAt, 0, ...projectELessons);
    }

    if (!window.COURSE_CHAPTERS?.some((chapter) => chapter.key === "projE")) {
      const beforeProject1 = window.COURSE_CHAPTERS.findIndex((chapter) => chapter.key === "proj1");
      const insertAt = beforeProject1 >= 0 ? beforeProject1 : window.COURSE_CHAPTERS.length;
      window.COURSE_CHAPTERS.splice(insertAt, 0, {
        key: "projE",
        title: CHAPTER_TITLE,
        lessonIds: projectELessons.map((lesson) => lesson.id),
        firstLesson: "projE-overview",
      });
    }
  }

  async function hydrateSourceCode() {
    const figures = Array.from(document.querySelectorAll("figure.pe-source-code[data-source-file]"));
    await Promise.all(figures.map(async (figure) => {
      const source = figure.querySelector(".code-source");
      if (!source || source.textContent.trim()) return;

      try {
        const response = await fetch(figure.dataset.sourceFile, { cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        source.textContent = await response.text();
      } catch (error) {
        source.textContent = `Could not load ${figure.dataset.sourceFile} automatically. Use the download card above or open the file from the repository.`;
      }
    }));

    if (typeof window.initRunnableCells === "function") {
      window.initRunnableCells(document);
    }
  }

  insertDom();
  insertMetadata();

  if (document.readyState === "complete") {
    hydrateSourceCode();
  } else {
    window.addEventListener("load", hydrateSourceCode);
  }
})();

# * Exercise D: Talk to Your Robot Over WiFi

---

## PD.0 What Is UDP and Why Does a Robot Use It?

*Networking · Robot control · 12 min*

---

Your laptop and your Arduino are two separate computers on the same WiFi network. For them to exchange information, they need a communication protocol: a set of rules for how data is packaged, addressed, sent, received, and interpreted.
In this exercise, the laptop sends wheel-speed commands to the Arduino. The Arduino drives the motors and sends encoder tick counts back. There is no machine learning and no camera processing here. This is pure embedded networking, and it is the exact foundation used later by Project 1's `Commander` and `Telemetry` classes.

### TCP vs UDP
TCP Transmission Control Protocol TCP guarantees that packets arrive and arrive in order. If a packet is lost, the sender waits and retransmits it. This is what you want for web pages, file downloads, email, and anything where missing data would corrupt the result. The tradeoff is latency. Waiting for retransmits takes time, and that delay can be worse than a missing packet in real-time control. UDP User Datagram Protocol UDP sends packets and does not wait for confirmation. If a packet is lost, it is simply gone. There is no retransmit and no ordering guarantee. The benefit is speed. UDP is used for video streaming, online games, DNS lookups, and real-time robot control where fresh information matters more than perfect delivery. An important thing to keep in mind is that Arduino controlling robot actions can be easily flodded with UDP packets and some could go missing, so be careful with the rate of transmission.
!!! tip "Why robots use UDP for motor commands"
    For motor commands, a dropped packet is far better than a delayed one. If your laptop sends `left: 100, right: 80` and the packet is lost, the robot keeps doing what it was last told for one extra control cycle, roughly 30 ms. That is acceptable. If TCP waits for a retransmit, the robot can freeze for hundreds of milliseconds while the network sorts itself out. At 1.5 m/s, a 100 ms freeze is 15 cm of motion with no correction.

### What Is A Port?
An IP address identifies a device on the network. A port identifies a specific program running on that device. Think of the IP address as the building address and the port as the apartment number.
Your Arduino listens for motor commands on one port and sends telemetry to another. Your Python script sends to the Arduino command port and listens on the telemetry port. Use a port number above 1024 and avoid common service ports.
Role Address / Port Meaning Arduino command listener 5001 The Arduino waits here for packets like MOTOR,150,150 . Arduino telemetry sender 5002 The Arduino sends encoder packets back toward the laptop. Python command sender ARDUINO_IP:5001 The laptop sends each motor command to the Arduino's IP address. Python telemetry listener 0.0.0.0:5002 The laptop accepts telemetry packets on any local network interface.
!!! info "The numbers are arbitrary, but the match is not"
    Ports `5001` and `5002` are just convenient choices. What matters is that the Arduino sketch and Python script agree on the same values.

### What You Will Build
Laptop Python keyboard + UDP sockets → WiFi Router local network → Arduino UNO R4 WiFi motors + encoders Encoder telemetry travels back from Arduino to Python on port 5002 .

### Prerequisites
Exercise C is complete and working: motors move, encoders count, and direction detection works. The robot is physically assembled with the custom shield, motors, encoders, wheels, and battery connected. Your laptop and Arduino are on the same WiFi network. Arduino IDE is installed with the UNO R4 board package.

---

## PD.1 How WiFi Works on the UNO R4

*Arduino · WiFi setup · 14 min*

---

The Arduino UNO R4 WiFi has a separate WiFi module, the ESP-S3 chip, that handles wireless communication. Your sketch talks to that module through the `WiFiS3` and `WiFiUDP` libraries.
The Arduino does not automatically know where your laptop is. It needs a WiFi network name, a WiFi password, and your laptop's IP address so it knows where to send telemetry.
!!! info "WiFiS3 is already installed"
    The `WiFiS3` and `WiFiUDP` libraries are bundled inside the Arduino UNO R4 board package you installed in Exercise C. They are not listed in the Library Manager because they are not separate libraries; they come with the board itself.
    You do not need to install anything. The lines `#include <WiFiS3.h>` and `#include <WiFiUDP.h>` will work immediately in any sketch. If Arduino IDE underlines them as errors, it means the wrong board is selected. Check that **Arduino UNO R4 WiFi** is selected in the board dropdown at the top of the IDE.

### Finding the Arduino IP Address
When the Arduino connects to your WiFi network, the router assigns it an IP address automatically using DHCP. The Exercise D sketch prints that address to the Serial Monitor when it starts.
!!! warning "The Arduino IP can change"
    The Arduino may get a new IP address each time it reconnects to WiFi unless your router has a reserved address for it. If the Python script suddenly cannot reach the robot, open Serial Monitor and check that `ARDUINO_IP` in Python still matches what the Arduino printed.

### Finding Your Laptop IP Address
The Arduino also needs your laptop's IP address for the return path. That value goes into `LAPTOP_IP` in the Arduino sketch.
**Windows**
Open Command Prompt and run `ipconfig`. Find **IPv4 Address** under your WiFi adapter. It usually looks like `192.168.x.x`.
**macOS**
Open Terminal and run `ipconfig getifaddr en0` for WiFi. The command prints your laptop's local IP address.
**Linux**
Open Terminal and run `ip addr show`. Find the `inet` address under your WiFi interface.

### Same Network Check
Both addresses should start with the same network prefix. If the Arduino prints `192.168.1.42`, your laptop should usually be something like `192.168.1.15`. If one device is on `192.168.1.x` and the other is on `10.0.0.x`, they may not be reachable from each other.
Arduino 192.168.1.42 Laptop 192.168.1.15 Result Same prefix, likely reachable
!!! tip "Use a normal local WiFi network"
    School, guest, or public networks sometimes block device-to-device traffic. If UDP packets do not arrive even though both devices are connected, test with a phone hotspot or a home router.

---

## PD.1b Your First UDP Message: Blink an LED

*Hands-on · UDP + tkinter · 20 min*

---

Before the full robot sketch, let us prove the whole pipeline with the smallest possible example: a Python window with two buttons that turn the Arduino's built-in LED on and off over WiFi. Once a single text packet can light an LED, everything later --- motors, servos, telemetry --- is just more of the same idea.

### The Whole Idea in One Picture
Your laptop and the Arduino are on the same WiFi network. Python sends a tiny text packet such as `D 13 1` to the Arduino's IP address and port. The Arduino reads it, acts on it, and sends a short reply back.
Python button click "LED ON" → UDP packet D 13 1 → Arduino :5005 digitalWrite(13, HIGH) → Reply OK digital
!!! info "Why UDP?"
    UDP is connectionless: Python just throws a packet at an address and port, with no handshake to set up first. It can occasionally drop a packet, but it is tiny, fast, and perfect for short control commands sent many times per second. That is exactly what a robot control loop needs.

### A Tiny Command Language
The Arduino sketch understands four one-letter commands. Each is plain text with space-separated numbers, so they are easy to send and easy to read in the Serial Monitor. For the LED you only need the first one.
Command Format Example What it does Digital D pin value D 13 1 Sets a pin HIGH (1) or LOW (0). Pin 13 is the built-in LED. PWM P pin value P 5 128 Analog output 0–255 (LED brightness, motor speed). Servo S pin angle S 10 90 Moves a servo on that pin to an angle 0–180. Motor M in1 in2 pwm speed M 4 5 6 180 Drives a motor through a driver; negative speed reverses it.
Blinking the LED is just two messages: `D 13 1` to turn it on and `D 13 0` to turn it off.

### The Arduino Receiver
This sketch connects to WiFi, listens on UDP port `5005`, and runs a command every time a packet arrives. Your WiFi name and password live in a separate file called `arduino_secrets.h` in the same sketch folder, which keeps your credentials out of the main code. Create it first:
arduino_secrets.hArduino
```cpp
// arduino_secrets.h header file
#define SECRET_SSID "yournetwork" //replace yournetwork with your wifi name inside quotes
#define SECRET_PASS "yourpassword" //replace yourpassword with your wifi password inside quotes
```
In the Arduino IDE, click the small arrow (the three dots on the right of the tab bar), choose **New Tab**, name it `arduino_secrets.h`, and paste the lines above with your own network name and password. The main sketch then pulls them in with `#include "arduino_secrets.h"`.
The WiFi connection and info code lives in its own header, `wifi_helper.h`, so the Arduino sketches in this exercise stay short. Add it as another tab (the same way you added `arduino_secrets.h`):
wifi_helper.hArduino
```cpp
#pragma once
#include <WiFiS3.h>
#include "arduino_secrets.h"

// True once the board has a real IP (not 0.0.0.0 yet).
bool ipIsValid(IPAddress ip) {
  return !(ip[0] == 0 && ip[1] == 0 && ip[2] == 0 && ip[3] == 0);
}

// Connect to WiFi using the credentials in arduino_secrets.h.
// Blocks until a valid IP is assigned; retries up to 5 times.
void connectToWiFi() {
  if (WiFi.status() == WL_NO_MODULE) {
    Serial.println("Communication with WiFi module failed!");
    while (true) delay(1000);
  }

  String fv = WiFi.firmwareVersion();
  if (fv < WIFI_FIRMWARE_LATEST_VERSION) {
    Serial.println("Warning: WiFi firmware may need updating.");
  }

  Serial.print("Connecting to WiFi: ");
  Serial.println(SECRET_SSID);

  WiFi.disconnect();
  delay(1000);

  int attempts = 0;
  while (true) {
    WiFi.begin(SECRET_SSID, SECRET_PASS);

    unsigned long start = millis();
    while (millis() - start < 20000) {
      if (WiFi.status() == WL_CONNECTED && ipIsValid(WiFi.localIP())) {
        Serial.println("WiFi connected with valid IP.");
        return;
      }
      delay(1000);
    }

    attempts++;
    Serial.println("WiFi attempt failed (IP stayed 0.0.0.0). Retrying...");
    WiFi.disconnect();
    delay(2000);

    if (attempts >= 5) {
      Serial.println("Could not get a valid IP after 5 attempts.");
      Serial.println("Check SSID, password, router DHCP, and 2.4 GHz WiFi.");
      while (true) delay(1000);
    }
  }
}

// Print the assigned IP (paste it into the Python script) and signal strength.
void printWiFiInfo() {
  Serial.println();
  Serial.println("===== WiFi Info =====");
  Serial.print("SSID: ");
  Serial.println(WiFi.SSID());
  Serial.print("Arduino IP address: ");
  Serial.println(WiFi.localIP());
  Serial.print("Signal strength RSSI: ");
  Serial.print(WiFi.RSSI());
  Serial.println(" dBm");
  Serial.println("=====================");
}
```
The main sketch then includes `wifi_helper.h` and just calls `connectToWiFi()` and `printWiFiInfo()` in `setup()`:
udp_command_receiver.inoArduino
```cpp
#include <WiFiUdp.h>
#include <Servo.h>
#include "wifi_helper.h"   // WiFi connect/info + arduino_secrets.h credentials

WiFiUDP udp;
const unsigned int UDP_PORT = 5005;

char packet[128];

// Servo object
Servo myServo;
int currentServoPin = -1;
bool servoIsAttached = false;


void sendReply(const char* msg) {
  udp.beginPacket(udp.remoteIP(), udp.remotePort());
  udp.print(msg);
  udp.endPacket();
}


int getInt(char* token, bool& ok) {
  if (token == NULL) {
    ok = false;
    return 0;
  }

  ok = true;
  return atoi(token);
}


void attachServoIfNeeded(int pin) {
  if (!servoIsAttached) {
    myServo.attach(pin);
    currentServoPin = pin;
    servoIsAttached = true;
    Serial.print("Servo attached to pin ");
    Serial.println(pin);
  }
  else if (pin != currentServoPin) {
    myServo.detach();
    myServo.attach(pin);
    currentServoPin = pin;
    Serial.print("Servo moved to pin ");
    Serial.println(pin);
  }
}


void handleCommand(char* msg) {
  char* cmd = strtok(msg, " \t\r\n");

  if (cmd == NULL) {
    sendReply("ERR empty command");
    return;
  }

  // Digital command:
  // D pin value
  // Example: D 13 1
  if (strcmp(cmd, "D") == 0) {
    bool ok1, ok2;

    int pin = getInt(strtok(NULL, " \t\r\n"), ok1);
    int value = getInt(strtok(NULL, " \t\r\n"), ok2);

    if (!ok1 || !ok2) {
      sendReply("ERR use: D pin value");
      return;
    }

    pinMode(pin, OUTPUT);
    digitalWrite(pin, value ? HIGH : LOW);

    sendReply("OK digital");
    return;
  }

  // PWM command:
  // P pin value
  // Example: P 5 128
  if (strcmp(cmd, "P") == 0) {
    bool ok1, ok2;

    int pin = getInt(strtok(NULL, " \t\r\n"), ok1);
    int value = getInt(strtok(NULL, " \t\r\n"), ok2);

    if (!ok1 || !ok2) {
      sendReply("ERR use: P pin value");
      return;
    }

    value = constrain(value, 0, 255);

    pinMode(pin, OUTPUT);
    analogWrite(pin, value);

    sendReply("OK pwm");
    return;
  }

  // Servo command:
  // S pin angle
  // Example: S 10 90
  if (strcmp(cmd, "S") == 0) {
    bool ok1, ok2;

    int pin = getInt(strtok(NULL, " \t\r\n"), ok1);
    int angle = getInt(strtok(NULL, " \t\r\n"), ok2);

    if (!ok1 || !ok2) {
      sendReply("ERR use: S pin angle");
      return;
    }

    if (pin < 0) {
      sendReply("ERR bad servo pin");
      return;
    }

    angle = constrain(angle, 0, 180);

    attachServoIfNeeded(pin);
    myServo.write(angle);

    sendReply("OK servo");
    return;
  }

  // Motor command:
  // M in1 in2 pwm speed
  // Example: M 4 5 6 180
  // Example: M 4 5 6 -180
  if (strcmp(cmd, "M") == 0) {
    bool ok1, ok2, ok3, ok4;

    int in1 = getInt(strtok(NULL, " \t\r\n"), ok1);
    int in2 = getInt(strtok(NULL, " \t\r\n"), ok2);
    int pwm = getInt(strtok(NULL, " \t\r\n"), ok3);
    int speed = getInt(strtok(NULL, " \t\r\n"), ok4);

    if (!ok1 || !ok2 || !ok3 || !ok4) {
      sendReply("ERR use: M in1 in2 pwm speed");
      return;
    }

    speed = constrain(speed, -255, 255);

    pinMode(in1, OUTPUT);
    pinMode(in2, OUTPUT);
    pinMode(pwm, OUTPUT);

    if (speed > 0) {
      digitalWrite(in1, HIGH);
      digitalWrite(in2, LOW);
      analogWrite(pwm, speed);
    }
    else if (speed < 0) {
      digitalWrite(in1, LOW);
      digitalWrite(in2, HIGH);
      analogWrite(pwm, -speed);
    }
    else {
      digitalWrite(in1, LOW);
      digitalWrite(in2, LOW);
      analogWrite(pwm, 0);
    }

    sendReply("OK motor");
    return;
  }

  sendReply("ERR unknown command");
}


void setup() {
  Serial.begin(9600);
  delay(1000);

  connectToWiFi();   // from wifi_helper.h
  printWiFiInfo();    // from wifi_helper.h

  udp.begin(UDP_PORT);

  Serial.print("Listening on UDP port ");
  Serial.println(UDP_PORT);
}


void loop() {
  int packetSize = udp.parsePacket();

  if (packetSize > 0) {
    int len = udp.read(packet, sizeof(packet) - 1);

    if (len > 0) {
      packet[len] = '\0';

      Serial.print("Received: ");
      Serial.println(packet);

      handleCommand(packet);
    }
  }
}
```
| Part | What it does |
| --- | --- |
| `WiFiUDP udp;` and `udp.begin(UDP_PORT)` | Creates the UDP listener and opens port `5005` so the Arduino can receive packets. |
| `udp.parsePacket()` | In `loop()`, checks whether a packet has arrived. Returns its size, or 0 if nothing is waiting. |
| `udp.read(packet, ...)` then `packet[len] = '\0'` | Copies the bytes into a buffer and adds a string terminator so it can be treated as text. |
| `strtok(msg, " \t\r\n")` | Splits the text on spaces. The first token is the command letter; the rest are its numbers. |
| `strcmp(cmd, "D") == 0` | Compares the command letter. If it is `D`, the sketch reads a pin and value and calls `digitalWrite`. |
| `sendReply("OK digital")` | Sends a short confirmation back to whoever sent the packet, using `udp.remoteIP()` and `udp.remotePort()`. |
| `connectToWiFi()` | Connects and waits until a *valid* IP (not `0.0.0.0`) is assigned, retrying if needed. This is why your IP prints reliably in the Serial Monitor. |
!!! tip "Note the port and IP"
    Upload the sketch, open the Serial Monitor at **9600 baud,** and write down the **Arduino IP address** it prints. You will paste that into the Python script next. The port is `5005` on both sides --- they must match.

### The Python Sender (a tkinter GUI)
On the laptop, a tiny program opens a window with two buttons. Each button sends one UDP command. There is no robot yet --- just you, a socket, and the Arduino.
led_control.pyVS Code
```python
import socket
import tkinter as tk


ARDUINO_IP = "192.168.6.50" # replace this IP address with what you received in the Serial monitor window of Arduino
ARDUINO_PORT = 5005
LED_PIN = 13

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(1.0)


def send(command):
    print("Sending:", command)

    sock.sendto(command.encode(), (ARDUINO_IP, ARDUINO_PORT))

    try:
        reply, address = sock.recvfrom(1024)
        print("Arduino replied:", reply.decode())
    except socket.timeout:
        print("No reply from Arduino")


def led_on():
    send(f"D {LED_PIN} 1")


def led_off():
    send(f"D {LED_PIN} 0")


root = tk.Tk()
root.title("LED Control")
root.geometry("250x120")

on_button = tk.Button(root, text="LED ON", width=15, command=led_on)
on_button.pack(pady=10)

off_button = tk.Button(root, text="LED OFF", width=15, command=led_off)
off_button.pack(pady=5)

root.mainloop()
```
| Line | What it does |
| --- | --- |
| `socket.socket(socket.AF_INET, socket.SOCK_DGRAM)` | Creates a UDP socket. `SOCK_DGRAM` means UDP (datagrams); `SOCK_STREAM` would be TCP. |
| `sock.settimeout(1.0)` | Waits at most one second for a reply, so the program never freezes if a packet is lost. |
| `sock.sendto(command.encode(), (ARDUINO_IP, ARDUINO_PORT))` | Sends the command string (converted to bytes) to the Arduino's address and port. |
| `sock.recvfrom(1024)` | Waits for the Arduino's reply. Wrapped in `try/except socket.timeout` so a missing reply is handled gracefully. |
| `send(f"D {LED_PIN} 1")` | Builds the exact text command the Arduino expects. `LED_PIN` is 13. |
!!! warning "Set ARDUINO_IP"
    Change `ARDUINO_IP` at the top of the Python file to the IP your Arduino printed in the Serial Monitor. If it is wrong, the buttons do nothing and you will see "No reply from Arduino".

### Understanding tkinter
**tkinter** is Python's built-in toolkit for simple desktop windows --- it ships with Python, so there is nothing to install. You only need a few pieces to build a button panel like this one.
Piece What it is root = tk.Tk() Creates the main window. Everything else lives inside it. root.title(...) / root.geometry("250x120") Sets the window title and its size in pixels. tk.Button(root, text="LED ON", command=led_on) Creates a button inside the window. command is the function to run when it is clicked. .pack(pady=10) Places the widget in the window. pack stacks widgets top to bottom; pady adds vertical spacing. root.mainloop() Starts the event loop. The program waits here, watching for clicks, until you close the window.
!!! info "command=led_on, not led_on()"
    You pass the function *itself* (`command=led_on`), with no parentheses. tkinter calls it later, each time the button is clicked. Writing `command=led_on()` would call it once immediately and store its return value instead --- a very common beginner mistake.

### Run It Yourself
Create arduino_secrets.h . Add the arduino_secrets.h file shown above to the sketch folder, filling in your own SECRET_SSID and SECRET_PASS . Upload the sketch. Open udp_command_receiver.ino in the Arduino IDE and upload it to the UNO R4 WiFi. Read the IP. Open the Serial Monitor at 9600 baud. Note the printed Arduino IP address . Edit the Python file. Set ARDUINO_IP to that address. Make sure your laptop is on the same WiFi. Run it. Run python led_control.py . Click LED ON and LED OFF — the Arduino's pin 13 LED should follow, and the terminal should print Arduino replied: OK digital .
!!! tip "Where this leads"
    You just sent a one-shot command from a GUI. In **PD.2** the Arduino sketch becomes a dedicated robot receiver, and from PD.3 onward Python sends `MOTOR` commands many times per second instead of on a button press. The transport --- a small UDP text packet to an IP and port --- never changes.

---

## PD.1c Driving Motors and a Servo over UDP

*Hands-on · UDP + tkinter · 20 min*

---

In PD.1b a single packet lit an LED. Now we send the same kind of packet to do real robot motion: a slider sweeps a servo, and five buttons drive two DC motors forward, backward, and turning. Nothing about the transport changes --- only the commands we send.
!!! info "The Arduino sketch is unchanged"
    Keep running the exact `udp_command_receiver.ino` from PD.1b. It already understands the `S` (servo) and `M` (motor) commands --- we just never sent them before. Only the Python side is new.

### Recap: the S and M Commands
Command Format Example What it does Servo S pin angle S 10 90 Moves the servo on that pin to an angle from 0 to 180 degrees. Motor M in1 in2 pwm speed M 4 5 6 180 Drives one motor through a driver. speed ranges from -255 to 255; negative reverses direction.
A motor driver needs three pins per motor: two direction pins (`in1`, `in2`) that set which way it spins, and one PWM pin that sets how fast. The Arduino sets the direction pins from the sign of `speed` and writes the magnitude to the PWM pin.

### Differential Drive in One Idea
The robot has two driven wheels, one on each side. You steer it purely by spinning those two wheels at different speeds or directions --- there is no steering wheel. This is called **differential drive**.
Action Left motor Right motor Result Forward +speed +speed Both wheels push forward. Backward -speed -speed Both wheels push back. Turn left -speed +speed Wheels spin opposite ways; robot rotates left in place. Turn right +speed -speed Opposite again; robot rotates right in place. Stop 0 0 Both wheels off.

### The Python Controller
Each button or slider movement turns into one or two UDP commands. The servo slider sends an `S` command as you drag it; each motion button sends two `M` commands, one per wheel.
drive_control.pyVS Code
```python
import socket
import tkinter as tk


ARDUINO_IP = "192.168.6.50"
ARDUINO_PORT = 5005

# Servo pin
SERVO_PIN = 10

# Left motor pins
LEFT_IN1 = 4
LEFT_IN2 = 5
LEFT_PWM = 6

# Right motor pins
RIGHT_IN1 = 12
RIGHT_IN2 = 13
RIGHT_PWM = 11

MOTOR_SPEED = 180

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def send(command):
    print("Sending:", command)
    sock.sendto(command.encode(), (ARDUINO_IP, ARDUINO_PORT))


# -----------------------------
# Servo control
# -----------------------------

def servo_changed(value):
    angle = int(value)
    send(f"S {SERVO_PIN} {angle}")


# -----------------------------
# Motor control
# -----------------------------

def left_motor(speed):
    send(f"M {LEFT_IN1} {LEFT_IN2} {LEFT_PWM} {speed}")


def right_motor(speed):
    send(f"M {RIGHT_IN1} {RIGHT_IN2} {RIGHT_PWM} {speed}")


def forward():
    left_motor(MOTOR_SPEED)
    right_motor(MOTOR_SPEED)


def backward():
    left_motor(-MOTOR_SPEED)
    right_motor(-MOTOR_SPEED)


def turn_left():
    left_motor(-MOTOR_SPEED)
    right_motor(MOTOR_SPEED)


def turn_right():
    left_motor(MOTOR_SPEED)
    right_motor(-MOTOR_SPEED)


def stop_motors():
    left_motor(0)
    right_motor(0)


# -----------------------------
# UI
# -----------------------------

root = tk.Tk()
root.title("Servo and Motor Control")
root.geometry("340x320")


servo_slider = tk.Scale(
    root,
    from_=0,
    to=180,
    orient="horizontal",
    label="Servo Angle",
    command=servo_changed
)
servo_slider.pack(fill="x", padx=20, pady=15)
servo_slider.set(90)


motor_label = tk.Label(root, text="DC Motor Control")
motor_label.pack(pady=5)


forward_button = tk.Button(root, text="Forward", width=15, command=forward)
forward_button.pack(pady=5)


button_row = tk.Frame(root)
button_row.pack(pady=5)


left_button = tk.Button(button_row, text="Turn Left", width=12, command=turn_left)
left_button.grid(row=0, column=0, padx=5)


right_button = tk.Button(button_row, text="Turn Right", width=12, command=turn_right)
right_button.grid(row=0, column=1, padx=5)


backward_button = tk.Button(root, text="Backward", width=15, command=backward)
backward_button.pack(pady=5)


stop_button = tk.Button(root, text="STOP", width=15, command=stop_motors)
stop_button.pack(pady=10)


root.mainloop()
```
| Line | What it does |
| --- | --- |
| `def send(command):` | Sends one UDP packet. Note there is no `recvfrom` here --- this is fire-and-forget (see the note below). |
| `def servo_changed(value):` | The slider's callback. tkinter passes the slider's current value in automatically; we convert it to `int` and send `S 10 angle`. |
| `left_motor(speed)` / `right_motor(speed)` | Build one `M` command for a wheel from its three pins plus a speed. |
| `forward()`, `backward()`, `turn_left()`, `turn_right()` | Combine the two wheels using the differential-drive table above. |
| `stop_motors()` | Sends speed 0 to both wheels. Always give yourself a STOP button. |

### New tkinter Pieces
This window uses three widgets you have not seen yet, plus a layout trick for placing buttons side by side.
Piece What it is tk.Scale(..., from_=0, to=180, command=servo_changed) A slider. It fires its command every time the handle moves, passing the current value. servo_slider.set(90) Sets the slider's starting position (servo centered at 90 degrees). tk.Label(root, text=...) A line of static text, used here as a heading above the motor buttons. tk.Frame(root) An invisible container. Here it groups the two turn buttons so they can sit on one row. .grid(row=0, column=0) Places widgets in a row/column grid inside the frame — that is how Turn Left and Turn Right end up side by side.
!!! tip "Scale passes a value, Button does not"
    A `Button`'s `command` takes no arguments, so its callbacks (`forward`, `stop_motors`) have none. A `Scale`'s `command` is called *with* the slider value, which is why `servo_changed(value)` has a parameter. The value arrives as a string, so we call `int(value)`.
!!! info "pack vs grid"
    Do not mix `pack` and `grid` on the same parent. This program uses `pack` for the main window (stacking widgets top to bottom) and `grid` only *inside* the `button_row` frame. That separation keeps tkinter happy.
!!! tip "Fire-and-forget"
    Unlike PD.1b, this script does not wait for a reply after each command. When you drag the slider it sends dozens of packets per second, and waiting for an acknowledgement on each one would make the controls feel laggy. Dropping the occasional packet is fine --- the next one arrives a few milliseconds later. This is exactly why robot control loops use UDP.

### Run It Yourself
Keep the PD.1b sketch running. The Arduino should still have udp_command_receiver.ino uploaded and be printing its IP at 9600 baud. Wire the hardware. Connect the servo signal to pin 10, and the motor driver to the pins listed at the top of the script (left: 4, 5, 6; right: 12, 11, and IN2 on 13). Power the motors from their own supply, not the Arduino's 5V. Set ARDUINO_IP . Paste the Arduino's IP from the Serial Monitor into the Python file. Run it. Run python drive_control.py . Drag the slider to sweep the servo, and use the buttons to drive. Watch the terminal print each command as it is sent.
!!! warning "Lift the wheels for the first test"
    Put the robot on a stand so the wheels spin freely before you let it drive on the floor. Confirm Forward, the turns, and especially STOP behave as expected first.
!!! tip "Where this leads"
    You are now sending real motion commands by hand. **PD.2** moves this logic onto the robot as a dedicated sketch, and the later lessons replace your button presses with a program that decides the speeds automatically --- the same `M` packets, just sent by code instead of a click.

---

## PD.2 The Arduino Sketch: Connect, Listen, Drive

*Arduino · UDP firmware · 28 min*

---

This sketch maintains a WiFi connection, listens for UDP motor command packets from the laptop, drives the motors through the SnappyXO shield, and sends encoder telemetry back every 50 ms.
Upload this after you have confirmed Exercise C motor and encoder wiring. Leave the robot wheels lifted off the table for the first test.
This sketch uses the same `arduino_secrets.h` file you created in [PD.1b](?lesson=projD-led) for your WiFi name and password --- keep it in this sketch's folder, no need to create it again.
This sketch uses the same `wifi_helper.h` header you created in [PD.1b](?lesson=projD-led) --- it includes the header and calls `connectToWiFi()` / `printWiFiInfo()` in `setup()`. No need to recreate it.
robot_udp.inoArduino
```cpp
#include <WiFiUDP.h>
#include "wifi_helper.h"   // WiFi connect/info + arduino_secrets.h credentials

// Your laptop's IP on the same WiFi (telemetry destination). Check with ipconfig.
const char* LAPTOP_IP  = "192.168.5.245";   // CHANGE THIS
const int   CMD_PORT   = 5001;
const int   TELEM_PORT = 5002;

// Motor pins
#define LEFT_IN1_PIN    4
#define LEFT_IN2_PIN    5
#define LEFT_EN_PIN     6
#define RIGHT_IN1_PIN   12
#define RIGHT_IN2_PIN   13
#define RIGHT_EN_PIN    11

// Encoder pins
#define ENC_LEFT_A      2
#define ENC_LEFT_B      A0
#define ENC_RIGHT_A     3
#define ENC_RIGHT_B     8

volatile long ticksLeft  = 0;
volatile long ticksRight = 0;

WiFiUDP udp;
char packetBuffer[64];

unsigned long lastCmdTime = 0;
const unsigned long CMD_TIMEOUT_MS = 500;

unsigned long lastTelemTime = 0;
const unsigned long TELEM_INTERVAL_MS = 50;

void leftEncoderISR() {
  if (digitalRead(ENC_LEFT_B) == HIGH) ticksLeft++;
  else ticksLeft--;
}

void rightEncoderISR() {
  if (digitalRead(ENC_RIGHT_B) == HIGH) ticksRight++;
  else ticksRight--;
}

void setMotors(int leftSpeed, int rightSpeed) {
  leftSpeed  = constrain(leftSpeed, -255, 255);
  rightSpeed = constrain(rightSpeed, -255, 255);

  digitalWrite(LEFT_IN1_PIN, leftSpeed >= 0 ? HIGH : LOW);
  digitalWrite(LEFT_IN2_PIN, leftSpeed >= 0 ? LOW : HIGH);
  analogWrite(LEFT_EN_PIN, abs(leftSpeed));

  digitalWrite(RIGHT_IN1_PIN, rightSpeed >= 0 ? HIGH : LOW);
  digitalWrite(RIGHT_IN2_PIN, rightSpeed >= 0 ? LOW : HIGH);
  analogWrite(RIGHT_EN_PIN, abs(rightSpeed));
}

void stopMotors() {
  analogWrite(LEFT_EN_PIN, 0);
  analogWrite(RIGHT_EN_PIN, 0);
  digitalWrite(LEFT_IN1_PIN, LOW);
  digitalWrite(LEFT_IN2_PIN, LOW);
  digitalWrite(RIGHT_IN1_PIN, LOW);
  digitalWrite(RIGHT_IN2_PIN, LOW);
}

void parseCommand(char* packet) {
  if (strncmp(packet, "STOP", 4) == 0) {
    stopMotors();
    lastCmdTime = millis();
    return;
  }

  if (strncmp(packet, "MOTOR,", 6) == 0) {
    int leftSpeed = 0;
    int rightSpeed = 0;
    int matched = sscanf(packet, "MOTOR,%d,%d", &leftSpeed, &rightSpeed);

    if (matched == 2) {
      setMotors(leftSpeed, rightSpeed);
      lastCmdTime = millis();

      Serial.print("Motor command: ");
      Serial.print(leftSpeed);
      Serial.print(", ");
      Serial.println(rightSpeed);
    }
  }
}

void sendTelemetry() {
  char buf[64];

  noInterrupts();
  long l = ticksLeft;
  long r = ticksRight;
  interrupts();

  snprintf(buf, sizeof(buf), "ENC,%ld,%ld", l, r);

  udp.beginPacket(LAPTOP_IP, TELEM_PORT);
  udp.write((uint8_t*)buf, strlen(buf));
  udp.endPacket();
}

void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("Arduino UNO R4 WiFi UDP Robot Controller");

  pinMode(LEFT_IN1_PIN, OUTPUT);
  pinMode(LEFT_IN2_PIN, OUTPUT);
  pinMode(LEFT_EN_PIN, OUTPUT);
  pinMode(RIGHT_IN1_PIN, OUTPUT);
  pinMode(RIGHT_IN2_PIN, OUTPUT);
  pinMode(RIGHT_EN_PIN, OUTPUT);
  stopMotors();

  pinMode(ENC_LEFT_A, INPUT_PULLUP);
  pinMode(ENC_LEFT_B, INPUT_PULLUP);
  pinMode(ENC_RIGHT_A, INPUT_PULLUP);
  pinMode(ENC_RIGHT_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENC_LEFT_A), leftEncoderISR, RISING);
  attachInterrupt(digitalPinToInterrupt(ENC_RIGHT_A), rightEncoderISR, RISING);

  connectToWiFi();   // from wifi_helper.h
  printWiFiInfo();   // from wifi_helper.h

  udp.begin(CMD_PORT);
  lastCmdTime = millis();
  Serial.println("Ready.");
}

void loop() {
  int packetSize = udp.parsePacket();

  if (packetSize > 0) {
    int len = udp.read(packetBuffer, sizeof(packetBuffer) - 1);

    if (len > 0) {
      packetBuffer[len] = '\0';

      Serial.print("Received UDP packet: ");
      Serial.println(packetBuffer);

      parseCommand(packetBuffer);
    }
  }

  if (millis() - lastCmdTime > CMD_TIMEOUT_MS) {
    stopMotors();
  }

  if (millis() - lastTelemTime >= TELEM_INTERVAL_MS) {
    sendTelemetry();
    lastTelemTime = millis();
  }
}
```

### Upload Steps
Add arduino_secrets.h (from PD.1b) and wifi_helper.h as tabs in the sketch — your WiFi name and password come from there. Fill in LAPTOP_IP with your laptop IP address from PD.1. Upload the sketch to the UNO R4 WiFi. Open Serial Monitor at 115200 baud. Wait for Arduino IP address: 192.168.x.x to print. Copy that Arduino IP address. You need it in the Python script in PD.3.

### Anatomy Table
connectToWiFi() — in wifi_helper.h Calls WiFi.begin(SECRET_SSID, SECRET_PASS) and polls WiFi.status() until a valid IP is assigned. Moved into the header so the main sketch stays short. WiFi.localIP() Returns the IP address assigned by the router. Print this value to Serial Monitor and copy it into ARDUINO_IP in Python. udp.begin(CMD_PORT) Opens a UDP socket and tells the Arduino to listen for packets on port 5001 . udp.parsePacket() Checks whether a new packet arrived. It returns the packet size if one is waiting, or 0 if nothing has arrived. This is non-blocking, so the main loop keeps running. strncmp(packet, "MOTOR,", 6) Compares the first six characters of the received string to MOTOR, . This identifies motor packets quickly before parsing the numbers. sscanf(packet, "MOTOR,%d,%d", ...) Parses two integers from the command string. If the packet is MOTOR,150,-100 , the left speed becomes 150 and the right speed becomes -100 . CMD_TIMEOUT_MS Stops the motors if no command arrives for 500 ms. This prevents the robot from driving away if the laptop script crashes or WiFi drops mid-run. constrain(leftSpeed, -255, 255) Clamps received values into the valid PWM range. This protects the robot from malformed packets containing out-of-range numbers.
!!! warning "First test with wheels lifted"
    Wireless command bugs can make a robot move unexpectedly. Keep the wheels off the floor until PD.3 proves that `STOP` and the safety timeout both work.

---

## PD.3 The Python Script: Send Commands, Read Telemetry

*Python · UDP sockets · 24 min*

---

The Arduino is now listening for UDP packets. This Python script sends a short forward command sequence, reads encoder telemetry, then reverses and stops.
Create a new file called `robot_control.py` in your `my-detector` folder or the same folder where you keep the Project 1 files.
robot_control.pyRun locally
```python
import socket
import time

# CONFIGURATION - fill in your own values.
ARDUINO_IP = "192.168.x.x"  # IP printed by Arduino on Serial Monitor
CMD_PORT = 5001            # must match Arduino sketch
TELEM_PORT = 5002          # must match Arduino sketch

# Create UDP socket for sending commands.
cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Create UDP socket for receiving telemetry.
telem_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
telem_sock.bind(("", TELEM_PORT))
telem_sock.settimeout(0.1)

def send_command(left, right):
    msg = f"MOTOR,{int(left)},{int(right)}"
    cmd_sock.sendto(msg.encode(), (ARDUINO_IP, CMD_PORT))

def send_stop():
    cmd_sock.sendto(b"STOP", (ARDUINO_IP, CMD_PORT))

def read_telemetry():
    try:
        data, _ = telem_sock.recvfrom(128)
        parts = data.decode().strip().split(",")
        if parts[0] == "ENC" and len(parts) == 3:
            return int(parts[1]), int(parts[2])
    except socket.timeout:
        pass
    return None, None

print(f"Connecting to Arduino at {ARDUINO_IP}")
print("Sending test sequence: forward 2 seconds, stop, backward 2 seconds.")

try:
    print("Forward...")
    start = time.time()
    while time.time() - start < 2.0:
        send_command(150, 150)
        left_ticks, right_ticks = read_telemetry()
        if left_ticks is not None:
            print(f"  Ticks - L: {left_ticks:+6d}  R: {right_ticks:+6d}")
        time.sleep(0.05)

    send_stop()
    print("Stopped.")
    time.sleep(0.5)

    print("Backward...")
    start = time.time()
    while time.time() - start < 2.0:
        send_command(-150, -150)
        left_ticks, right_ticks = read_telemetry()
        if left_ticks is not None:
            print(f"  Ticks - L: {left_ticks:+6d}  R: {right_ticks:+6d}")
        time.sleep(0.05)

    send_stop()
    print("Done.")

except KeyboardInterrupt:
    send_stop()
    print("Interrupted - motors stopped.")

finally:
    cmd_sock.close()
    telem_sock.close()
```

### Run It
```bash
python robot_control.py
```

### What To Expect
The robot drives forward for two seconds while encoder tick counts print in the terminal. Then it stops, drives backward for two seconds, and stops again. If you press `Ctrl+C`, the script sends `STOP` before exiting.

### Anatomy Table
socket.AF_INET, socket.SOCK_DGRAM AF_INET means IPv4 networking. SOCK_DGRAM means UDP. These are the same socket constants used in Project 1's Commander and Telemetry classes. cmd_sock.sendto(msg.encode(), (ARDUINO_IP, CMD_PORT)) Encodes the Python string into bytes and sends it as a UDP packet to the Arduino command port. UDP is connectionless, so there is no prior connection step. telem_sock.bind(("", TELEM_PORT)) Tells the operating system that this socket wants packets arriving on TELEM_PORT . The empty string means accept packets on any local network interface. telem_sock.settimeout(0.1) Makes recvfrom return after 100 ms if no packet arrived. Without a timeout, the script could freeze while waiting for telemetry.
!!! tip "Telemetry proves the full round trip"
    If the robot moves but tick values never print, laptop-to-Arduino commands work, but Arduino-to-laptop telemetry does not. Check `LAPTOP_IP`, `TELEM_PORT`, and firewall settings.

---

## PD.4 Keyboard Control: Drive with WASD

*Python · Interactive control · 26 min*

---

The PD.3 script runs a fixed sequence. Now replace that sequence with live keyboard control so you can drive the robot interactively from your laptop.
This script sends a command about 33 times per second. The Arduino safety timeout will stop the robot if these commands stop arriving.

### Install Keyboard Input Support
```bash
pip install pygame
```
robot_keyboard.pyRun locally
```python
import socket
import time
import pygame

ARDUINO_IP = "192.168.x.x"
CMD_PORT = 5001
TELEM_PORT = 5002

# UDP sockets
cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
telem_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
telem_sock.bind(("", TELEM_PORT))
telem_sock.settimeout(0.05)


def send_command(left, right):
    msg = f"MOTOR,{int(left)},{int(right)}"
    cmd_sock.sendto(msg.encode(), (ARDUINO_IP, CMD_PORT))


def send_stop():
    cmd_sock.sendto(b"STOP", (ARDUINO_IP, CMD_PORT))


# ---------------- PYGAME INIT ----------------
pygame.init()
screen = pygame.display.set_mode((300, 200))  # required to capture input
pygame.display.set_caption("Robot Control")

clock = pygame.time.Clock()

print("Robot keyboard control (pygame)")
print("  W = forward    S = backward")
print("  A = turn left  D = turn right")
print("  ESC = quit")
print()

SPEED = 180
TURN = 150

running = True

try:
    while running:
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        left = 0
        right = 0

        # movement logic
        if keys[pygame.K_w]:
            left, right = SPEED, SPEED
        elif keys[pygame.K_s]:
            left, right = -SPEED, -SPEED

        if keys[pygame.K_a]:
            left -= TURN
            right += TURN

        if keys[pygame.K_d]:
            left += TURN
            right -= TURN

        send_command(left, right)

        # handle quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if keys[pygame.K_ESCAPE]:
            running = False

        # telemetry receive
        try:
            data, _ = telem_sock.recvfrom(128)
            parts = data.decode().strip().split(",")

            if parts[0] == "ENC" and len(parts) == 3:
                l, r = int(parts[1]), int(parts[2])
                print(
                    f"Keys: W:{keys[pygame.K_w]} A:{keys[pygame.K_a]} "
                    f"S:{keys[pygame.K_s]} D:{keys[pygame.K_d]} | "
                    f"L: {left:+4d} R: {right:+4d} | "
                    f"Ticks L: {l:+6d} R: {r:+6d}",
                    end="\r",
                )

        except socket.timeout:
            pass

        time.sleep(0.03)
        clock.tick(60)

finally:
    send_stop()
    cmd_sock.close()
    telem_sock.close()
    pygame.quit()
    print("\nStopped.")
```

### Run It
```bash
python robot_keyboard.py
```

### Controls
W forward S backward A turn left D turn right Esc quit and stop
!!! tip "Multiple keys combine"
    You can hold multiple keys at once. Holding `W` and `D` drives forward while turning right because the forward speed and steering correction are added together. This is the same idea used later when the full tracker blends forward motion with steering corrections.

### How The Steering Math Works
The code starts each loop with `left = 0` and `right = 0`. Forward and backward keys set the base speed. Left and right keys then add a turn correction by increasing one side and decreasing the other.
Key state Left command Right command Robot behavior W +180 +180 Drive forward. S -180 -180 Drive backward. A -150 +150 Turn left in place. D +150 -150 Turn right in place. W + D +330 +30 Forward arc to the right. The Arduino clamps values to the valid PWM range.

---

## PD.5 Receiving Encoder Telemetry in Python

*Telemetry · CSV logging · 22 min*

---

The telemetry printed by `robot_keyboard.py` is real encoder tick data coming from the Arduino over UDP every 50 ms. Now you will save it to a CSV file so you can plot it after the robot moves. Because only one program can receive the telemetry on port `5002`, the logging goes directly inside `robot_keyboard.py` --- you do not run a second receiver.
Run `robot_keyboard.py` and drive the robot forward in a straight line for about 50 cm. Both left and right tick counts should increase at roughly the same rate. If one grows much faster, your motors have different speeds at the same PWM value. That is normal hardware variation.
Here is the complete `robot_keyboard.py` --- the PD.4 keyboard driver with CSV logging built in. Run this one script: it drives the robot, prints live ticks, and writes `telemetry_log.csv` when you press Esc.
robot_keyboard.py (complete: drive + log)Run locally
```python
import socket
import time
import csv
import pygame

ARDUINO_IP = "192.168.x.x"
CMD_PORT = 5001
TELEM_PORT = 5002

# UDP sockets
cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
telem_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
telem_sock.bind(("", TELEM_PORT))
telem_sock.settimeout(0.05)


def send_command(left, right):
    msg = f"MOTOR,{int(left)},{int(right)}"
    cmd_sock.sendto(msg.encode(), (ARDUINO_IP, CMD_PORT))


def send_stop():
    cmd_sock.sendto(b"STOP", (ARDUINO_IP, CMD_PORT))


# ---------------- PYGAME INIT ----------------
pygame.init()
screen = pygame.display.set_mode((300, 200))  # required to capture input
pygame.display.set_caption("Robot Control")

clock = pygame.time.Clock()

print("Robot keyboard control + telemetry logging (pygame)")
print("  W = forward    S = backward")
print("  A = turn left  D = turn right")
print("  ESC = quit and save telemetry_log.csv")
print()

SPEED = 180
TURN = 150

# Telemetry log: every encoder reading is stored, then written to CSV on exit.
log_rows = []
start_time = time.time()

running = True

try:
    while running:
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        left = 0
        right = 0

        # movement logic
        if keys[pygame.K_w]:
            left, right = SPEED, SPEED
        elif keys[pygame.K_s]:
            left, right = -SPEED, -SPEED

        if keys[pygame.K_a]:
            left -= TURN
            right += TURN

        if keys[pygame.K_d]:
            left += TURN
            right -= TURN

        send_command(left, right)

        # handle quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if keys[pygame.K_ESCAPE]:
            running = False

        # telemetry receive + log
        try:
            data, _ = telem_sock.recvfrom(128)
            parts = data.decode().strip().split(",")

            if parts[0] == "ENC" and len(parts) == 3:
                l, r = int(parts[1]), int(parts[2])
                log_rows.append([f"{time.time() - start_time:.3f}", l, r])
                print(
                    f"Keys: W:{keys[pygame.K_w]} A:{keys[pygame.K_a]} "
                    f"S:{keys[pygame.K_s]} D:{keys[pygame.K_d]} | "
                    f"L: {left:+4d} R: {right:+4d} | "
                    f"Ticks L: {l:+6d} R: {r:+6d}",
                    end="\r",
                )

        except socket.timeout:
            pass

        time.sleep(0.03)
        clock.tick(60)

finally:
    send_stop()
    cmd_sock.close()
    telem_sock.close()
    pygame.quit()

    with open("telemetry_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "left_ticks", "right_ticks"])
        writer.writerows(log_rows)

    print(f"\nStopped. Saved {len(log_rows)} rows to telemetry_log.csv")
```
!!! warning "Only one telemetry receiver"
    The Arduino sends telemetry to a single port (`5002`), and only one program can bind it. Do not run a separate logger at the same time as `robot_keyboard.py` --- on Windows the second one fails with `OSError: [WinError 10048]`. That is why the logging lives inside the driving script.

### Logging Procedure
Run python robot_keyboard.py in one terminal. Drive the robot around for about 30 seconds. Press Esc to stop — the script saves telemetry_log.csv as it exits. Open telemetry_log.csv in Excel, Google Sheets, or another plotting tool. Plot left_ticks and right_ticks against time_s .

### How To Read The Plot
**Straight driving**
The left and right tick lines should climb at similar rates. Small differences are normal.
**Turning**
The two lines diverge. One side may increase while the other decreases during a turn in place.
**Stopped robot**
The tick lines should flatten. If they keep changing, the wheels are still moving or the encoder signal is noisy.
!!! info "Why logging matters"
    This is exactly what a robotics engineer does when debugging motor control. Logging raw sensor data to CSV and plotting it reveals behavior that is invisible in terminal print output. You will use this technique again in Project 1 when tuning PID gains.

---

## PD.6 Understanding the Full Communication Loop

*Architecture · Timing · 18 min*

---

The project now has a complete two-way communication loop. Python sends motor commands to the Arduino. The Arduino drives the motors and sends encoder telemetry back. These two paths are independent.
Frame 1: Command path Python script creates MOTOR,150,150 encodes to bytes sendto(ARDUINO_IP, 5001) UDP packet over WiFi Arduino UNO R4 udp.parsePacket() detects arrival udp.read() copies bytes parseCommand() extracts speeds setMotors(150, 150) Frame 2: Telemetry path Arduino UNO R4 encoder ISRs count ticks every 50 ms: ENC,847,851 udp.beginPacket(LAPTOP_IP, 5002) udp.endPacket() UDP packet over WiFi Python script recvfrom(128) receives bytes decode() converts to string split(",") extracts values prints or logs the ticks
!!! info "The timing is deliberately decoupled"
    The command rate is approximately 33 Hz, one command every 30 ms from the Python loop. The telemetry rate is 20 Hz, one packet every 50 ms from the Arduino loop. The command sender does not wait for telemetry, and the telemetry sender does not wait for commands. This decoupled timing is intentional and matches the design used in Project 1.

### Why This Architecture Works
**Python is the brain**
The laptop handles keyboard input now and will later handle vision, object detection, PID control, and state logic.
**Arduino is the body**
The microcontroller owns the hardware pins, motor driver, encoder interrupts, and safety timeout.
**UDP is the link**
Short text packets carry commands one way and telemetry the other way without blocking the control loop.

### Packet Formats
Packet Direction Example Parser MOTOR,left,right Laptop to Arduino MOTOR,180,120 Arduino sscanf STOP Laptop to Arduino STOP Arduino strncmp ENC,leftTicks,rightTicks Arduino to laptop ENC,847,851 Python split(",")

---

## PD.7 Connecting the Dots to Project 1

*Project 1 bridge · Review · 16 min*

---

Project 1's robot code includes two classes in a file called `shared.py`: `Commander` (sends motor commands) and `Telemetry` (receives encoder data). You have not started Project 1 yet, so both classes are printed below. Read through them now --- Exercise D was designed so they look familiar instead of mysterious. When you reach Project 1, this file will already make sense.
Commander and Telemetry (from Project 1's shared.py)
```python
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
```

### The Commander Class
Commander.__init__ Uses socket.AF_INET and socket.SOCK_DGRAM , identical to cmd_sock in PD.3. There is no bind() call because Commander only sends commands. Commander.motors Builds a string like MOTOR,left,right,light . This is the same format as your Exercise D motor command, extended with one extra LED field. Commander.stop Sends a stop packet, just like send_stop() in Exercise D. Project 1 extends it with LED state.

### The Telemetry Class
Telemetry.__init__ Creates a UDP socket, calls bind(("", port)) , and sets a timeout. That is the same receive setup used in robot_control.py and robot_keyboard.py . Background thread Project 1 uses threading.Thread(target=self._run) so telemetry keeps arriving without blocking the main detection loop. This is the same reason Exercise B used a background stream thread. Telemetry._run Uses data.decode().strip().split(",") , the same parsing pattern from PD.3. Project 1 validates more fields because it sends more information.

### What Changes In Project 1?
Part Exercise D Project 1 Motor command MOTOR,left,right MOTOR,left,right,light Stop command STOP STOP,OFF Telemetry packet ENC,leftTicks,rightTicks ENC,leftTicks,rightTicks,leftCmd,rightCmd Command source Keyboard keys YOLO detection plus PID controllers Architecture Python sends, Arduino drives, Arduino reports The same architecture with vision and state logic added
!!! tip "Closing the chapter"
    You have built a wirelessly controlled robot with real-time sensor feedback. The laptop is the brain, the Arduino is the body, and UDP carries signals between them in both directions at robotics control-loop speed. Project 1 adds camera input and PID logic on top of the same communication foundation.

### Final Checklist
Arduino connects to WiFi and prints its IP address in Serial Monitor. robot_control.py runs and the robot drives forward, stops, then drives backward. Encoder ticks print in the terminal and increase or decrease correctly. robot_keyboard.py allows live WASD driving. telemetry_log.csv was generated and opened successfully. You can explain the difference between bind() and sendto() . You can explain what CMD_TIMEOUT_MS does and why it matters. You can recognize the Exercise D patterns in the Commander and Telemetry classes shown above.
When all boxes are checked, you are ready for Project 1.

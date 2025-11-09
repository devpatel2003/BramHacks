#include <Servo.h>

Servo servoPan, servoTilt;

const int PAN_PIN  = 9;   // check wiring
const int TILT_PIN = 10;  // check wiring

// Start centered (inside your ±15° window if you use it on the PC side)
float panDeg  = 85.0;
float tiltDeg = 43.0;

void setup() {
  Serial.begin(115200);

  // Attach once and never detach
  servoPan.attach(PAN_PIN, 1000, 2000);
  servoTilt.attach(TILT_PIN, 1000, 2000);

  // Go to start pose
  servoPan.write((int)panDeg);
  servoTilt.write((int)tiltDeg);

  Serial.println(F("Ready. Commands:"));
  Serial.println(F("  PAN=<deg>,TILT=<deg>   e.g., PAN=90,TILT=101"));
  Serial.println(F("  TEST                   (tilt sweep 75..105)"));
  Serial.println(F("  ?                      (report current)"));
}

void loop() {
  static String buf;

  // Read a full line
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      buf.trim();
      if (buf.length()) handleCommand(buf);
      buf = "";
    } else if (c != '\r') {
      buf += c;
    }
  }
}

void handleCommand(const String &cmd) {
  // Helpers
  if (cmd.equalsIgnoreCase("TEST")) {
    Serial.println(F("Tilt sweep 75..105"));
    for (int t = 75; t <= 105; ++t) { servoTilt.write(t); delay(18); }
    for (int t = 105; t >= 75; --t) { servoTilt.write(t); delay(18); }
    tiltDeg = 90; servoTilt.write(90);
    Serial.println(F("Done."));
    return;
  }
  if (cmd == "?") {
    Serial.print(F("PAN="));  Serial.print(panDeg,1);
    Serial.print(F(", TILT=")); Serial.println(tiltDeg,1);
    return;
  }

  // Parse PAN / TILT (order-agnostic, commas optional)
  float pCmd = panDeg, tCmd = tiltDeg;
  bool have = false;

  int pIdx = cmd.indexOf("PAN=");
  if (pIdx >= 0) {
    int end = cmd.indexOf(',', pIdx);
    String v = (end >= 0) ? cmd.substring(pIdx + 4, end) : cmd.substring(pIdx + 4);
    v.trim();
    if (v.length()) { pCmd = v.toFloat(); have = true; }
  }

  int tIdx = cmd.indexOf("TILT=");
  if (tIdx >= 0) {
    int end = cmd.indexOf(',', tIdx);
    String v = (end >= 0) ? cmd.substring(tIdx + 5, end) : cmd.substring(tIdx + 5);
    v.trim();
    if (v.length()) { tCmd = v.toFloat(); have = true; }
  }

  if (!have) {
    Serial.print(F("Unrecognized: "));
    Serial.println(cmd);
    return;
  }

  // Constrain to the servo’s physical 0..180 range (PC side enforces your tighter limits)
  pCmd = constrain(pCmd, 0, 180);
  tCmd = constrain(tCmd, 0, 180);

  // Write and cache
  servoPan.write((int)pCmd);
  servoTilt.write((int)tCmd);
  panDeg = pCmd;
  tiltDeg = tCmd;

  Serial.print(F("Set -> PAN="));  Serial.print(panDeg,1);
  Serial.print(F(", TILT="));       Serial.println(tiltDeg,1);
}

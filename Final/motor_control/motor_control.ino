// arduino.ino
#include <Servo.h>

Servo servoPan;
Servo servoTilt;

// -------- PINS ----------
const int PAN_PIN  = 9;
const int TILT_PIN = 10;
// -----------------------------

// Mechanical limits (deg)
const float PAN_MIN  = 0.0;
const float PAN_MAX  = 180.0;
const float TILT_MIN = 0.0;
const float TILT_MAX = 180.0;

// Smoothing factor
const float ALPHA = 0.5;

// Home position
float panDeg  = 80.0;
float tiltDeg = 43.0;

void drainSerial() {
  while (Serial.available()) (void)Serial.read();
}

void setup() {
  Serial.begin(115200);

  pinMode(PAN_PIN, OUTPUT);
  pinMode(TILT_PIN, OUTPUT);
  digitalWrite(PAN_PIN, LOW);
  digitalWrite(TILT_PIN, LOW);


  servoPan.attach(PAN_PIN);
  servoTilt.attach(TILT_PIN);

  delay(200);

  servoPan.write((int)panDeg);
  servoTilt.write((int)tiltDeg);
  Serial.println("READY");
}

bool parseAngles(const String &line, float &panOut, float &tiltOut) {
  // Expected: "PAN=<deg>,TILT=<deg>"
  int pIdx = line.indexOf("PAN=");
  int tIdx = line.indexOf("TILT=");
  if (pIdx < 0 || tIdx < 0) return false;

  int comma = line.indexOf(',', pIdx);
  if (comma < 0) return false;

  String pStr = line.substring(pIdx + 4, comma);
  String tStr = line.substring(tIdx + 5);

  panOut  = pStr.toFloat();
  tiltOut = tStr.toFloat();
  if (isnan(panOut) || isnan(tiltOut)) return false;

  // clamp
  if (panOut  < PAN_MIN)  panOut  = PAN_MIN;
  if (panOut  > PAN_MAX)  panOut  = PAN_MAX;
  if (tiltOut < TILT_MIN) tiltOut = TILT_MIN;
  if (tiltOut > TILT_MAX) tiltOut = TILT_MAX;

  return true;
}

void loop() {
  static String buf;

  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      float pCmd, tCmd;
      if (parseAngles(buf, pCmd, tCmd)) {
        // simple smoothing
        panDeg  = panDeg  + ALPHA * (pCmd - panDeg);
        tiltDeg = tiltDeg + ALPHA * (tCmd - tiltDeg);

        servoPan.write((int)panDeg);
        servoTilt.write((int)tiltDeg);
        Serial.print("OK ");
        Serial.print(panDeg, 1);
        Serial.print(",");
        Serial.println(tiltDeg, 1);
      } else {
        Serial.println("ERR");
      }
      buf = "";
    } else if (c != '\r') {
      buf += c;
      if (buf.length() > 64) buf = ""; // safety
    }
  }

  delay(5);
}

# DriverGuard AI

Real-time driver drowsiness monitoring from a camera. It tracks the driver's face landmarks, detects **eye closure, yawning and head deviation**, turns them into an explainable **risk score**, sounds an alarm, and records every warning to CSV files and a PDF session report.

There are two ways to run it:

- **`dashboard.py`**: a Streamlit web dashboard that uses the browser's camera (WebRTC). This is the main version and the one you can deploy.
- **`app.py`**: a standalone OpenCV desktop window (press `Q` to quit).

> The risk score is a heuristic indicator, not a validated measure of fitness to drive.

## What it detects

All of it comes from MediaPipe's Face Landmarker (`models/face_landmarker.task`):

| Signal | How it is measured | Default threshold (`config.py`) |
|---|---|---|
| **Eyes closed / drowsiness** | Eye Aspect Ratio (EAR), smoothed over 3 frames | EAR below 0.21; "drowsy" after the eyes stay closed for 1.5 s |
| **Yawning** | Mouth Aspect Ratio (MAR) | MAR above 0.55 for 1.0 s (2 s cooldown) |
| **Head deviation** | Head pose estimated from the landmarks | Tilt above 28 degrees for 1.5 s (3 s cooldown) |
| **No face** | Face not found in the frame | Counted as a warning signal |

### Risk score

`monitoring/state_manager.py` adds up the active signals and caps the result at 100:

| Signal | Points |
|---|---|
| Prolonged eye closure (drowsy) | 60 |
| Eyes closed (not yet prolonged) | 20 |
| Possible yawn | 20 |
| Head deviation | 20 |
| Face not detected | 10 |

0 to 29 is **LOW RISK**, 30 to 59 is **MODERATE RISK**, and 60 or more is **HIGH RISK**. Each result also lists the reasons, and a warning event is recorded only when a signal *starts*, not on every frame.

## Dashboard features

- Live camera feed with clean eye and lip outlines drawn on the face
- Driver status, risk score, risk trend chart and a list of recent events
- Browser alarm sound with an on/off switch and a "test alarm" button
- Last-session summary and event history
- **PDF session report** generated with ReportLab
- **CSV logs** in `logs/`: `events.csv` (each warning when it starts) and `risk_history.csv` (the risk score over time)

## Run it locally

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

For the desktop version: `python app.py`.

Python 3.11 is recommended. The thresholds and risk points are all in `config.py`; `python create_alarm.py` regenerates the alarm sound.

## Deploy on Streamlit Cloud

- Main file: `dashboard.py`; Python version: 3.11 (system packages are in `packages.txt`).
- The browser camera needs a TURN server to connect from the cloud. Add one in the app's Secrets. Easiest is a free Metered account (metered.ca), then:

```toml
[metered]
domain = "yourapp.metered.live"
api_key = "your-metered-api-key"
```

  Or set a fixed server:

```toml
[turn]
urls = ["turn:YOUR_HOST:443?transport=tcp"]
username = "..."
credential = "..."
```

The cloud version keeps the load low on purpose: the camera runs at 320 x 240 and 15 fps, and the dashboard refreshes in short fragments.

## Project layout

```text
dashboard.py            Streamlit dashboard (WebRTC video processor, charts, reports)
app.py                  Standalone OpenCV desktop version
config.py               Thresholds, risk points and folders
vision/                 eye_analyzer.py (EAR), mouth_analyzer.py (MAR), head_pose.py, face_detector.py
monitoring/             state_manager.py (risk score), alert_manager.py (alarm),
                        event_logger.py (CSV logs), pdf_report.py (PDF reports)
models/                 MediaPipe face landmarker model
assets/alert.wav        Alarm sound
tests/                  Unit tests for the monitoring logic
```

## Status and limitations

- The thresholds are general defaults. Glasses, lighting, camera angle and different faces change the EAR and MAR values, so tune `config.py` for your setup.
- It does not detect distraction by phone, looking away for long periods beyond the head-tilt rule, or alcohol or medical conditions.
- Warnings are an aid for the driver, not a safety system. Do not rely on it while driving.

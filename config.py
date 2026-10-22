
from pathlib import Path

# Project folders
ROOT_DIR = Path(__file__).resolve().parent

MODEL_PATH = ROOT_DIR / "models" / "face_landmarker.task"
ALERT_SOUND_PATH = ROOT_DIR / "assets" / "alert.wav"

LOG_DIR = ROOT_DIR / "logs"
DRIVERS_DIR = LOG_DIR / "drivers"
REPORTS_DIR = ROOT_DIR / "reports"

# Eye monitoring
EAR_THRESHOLD = 0.21
CLOSED_DURATION = 1.5
SMOOTHING_FRAMES = 3

# Yawning detection
YAWN_THRESHOLD = 0.55
YAWN_DURATION = 1.0
YAWN_COOLDOWN = 2.0

# Head pose detection
HEAD_TILT_THRESHOLD = 28.0
HEAD_TILT_DURATION = 1.5
HEAD_TILT_COOLDOWN = 3.0

# Experimental risk scoring
RISK_DROWSINESS = 60
RISK_EYES_CLOSED = 20
RISK_YAWNING = 20
RISK_HEAD_TILT = 20
RISK_NO_FACE = 10

LOW_RISK_LIMIT = 29
MODERATE_RISK_LIMIT = 59

# Camera settings
CAMERA_INDEX = 0
WINDOW_NAME = "DriverGuard AI | Driver Monitoring"

# Create required folders automatically
for folder in (LOG_DIR, DRIVERS_DIR, REPORTS_DIR):
    folder.mkdir(parents=True, exist_ok=True)


from pathlib import Path
import glob
import os
import tempfile
import time

import cv2
import mediapipe as mp
import pandas as pd
import streamlit as st

from config import (
    MODEL_PATH,
    ALERT_SOUND_PATH,
    LOG_DIR,
    CAMERA_INDEX,
    EAR_THRESHOLD,
    CLOSED_DURATION,
    SMOOTHING_FRAMES,
    YAWN_THRESHOLD,
    YAWN_DURATION,
    YAWN_COOLDOWN,
    HEAD_TILT_THRESHOLD,
    HEAD_TILT_DURATION,
    HEAD_TILT_COOLDOWN,
)
from monitoring.pdf_report import generate_session_report
from vision.eye_analyzer import (
    EyeAnalyzer,
    LEFT_EYE,
    RIGHT_EYE,
    eye_aspect_ratio,
)
from vision.mouth_analyzer import YawnAnalyzer, mouth_aspect_ratio
from vision.head_pose import HeadPoseAnalyzer, estimate_head_pose
from monitoring.alert_manager import AlertManager
from monitoring.state_manager import DriverStateManager
from monitoring.event_logger import EventLogger


# =========================================================
# PAGE CONFIGURATION & PROFESSIONAL THEME
# =========================================================

st.set_page_config(
    page_title="DriverGuard AI",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Orbitron:wght@500;600;700;800&display=swap');

:root {
    --dg-bg: #070b14;
    --dg-panel: #0e1726;
    --dg-panel-2: #121f31;
    --dg-border: rgba(81, 190, 220, 0.20);
    --dg-cyan: #43d9f5;
    --dg-teal: #3ce6c2;
    --dg-text: #e8f4fc;
    --dg-muted: #91a6ba;
}

.stApp {
    background:
        radial-gradient(ellipse at 15% 0%, rgba(17, 101, 139, .19), transparent 35%),
        radial-gradient(ellipse at 95% 15%, rgba(16, 174, 160, .10), transparent 30%),
        linear-gradient(145deg, #070b14 0%, #0a1220 55%, #071019 100%);
    color: var(--dg-text);
    font-family: 'Inter', sans-serif;
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    opacity: .14;
    background-image:
        linear-gradient(rgba(67,217,245,.06) 1px, transparent 1px),
        linear-gradient(90deg, rgba(67,217,245,.06) 1px, transparent 1px);
    background-size: 42px 42px;
    mask-image: linear-gradient(to bottom, black, transparent 85%);
}

.block-container {
    max-width: 1600px;
    padding-top: 1.7rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #f1f8ff !important;
    letter-spacing: -0.035em;
}

h1 {
    font-family: 'Orbitron', sans-serif !important;
    font-size: clamp(1.55rem, 3vw, 2.25rem) !important;
    font-weight: 700 !important;
}

h2, h3 {
    font-weight: 700 !important;
}

p, label, [data-testid="stCaptionContainer"] {
    color: var(--dg-muted);
}

[data-testid="stMetric"] {
    position: relative;
    overflow: hidden;
    background: linear-gradient(145deg, #132236, #0d1726);
    border: 1px solid var(--dg-border);
    border-radius: 16px;
    padding: 18px 18px 15px;
    box-shadow: 0 8px 28px rgba(0,0,0,.17);
    transition: border-color .25s ease, transform .25s ease,
                box-shadow .25s ease;
}

[data-testid="stMetric"]:hover {
    border-color: rgba(67,217,245,.55);
    transform: translateY(-2px);
    box-shadow: 0 10px 32px rgba(18,174,206,.09);
}

[data-testid="stMetricLabel"] {
    color: #9ab0c5 !important;
    font-size: .82rem !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: #eafaff !important;
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(1.15rem, 2vw, 1.8rem) !important;
}

[data-testid="stMetricDelta"] {
    font-size: .78rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1422, #080e18);
    border-right: 1px solid var(--dg-border);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

div.stButton > button,
div.stDownloadButton > button {
    min-height: 44px;
    border: 1px solid rgba(67,217,245,.35);
    border-radius: 11px;
    background: linear-gradient(115deg, #123d54, #12534f);
    color: #effcff;
    font-weight: 700;
    transition: transform .2s ease, border-color .2s ease,
                filter .2s ease, box-shadow .2s ease;
}

div.stButton > button:hover,
div.stDownloadButton > button:hover {
    border-color: var(--dg-cyan);
    filter: brightness(1.15);
    transform: translateY(-1px);
    box-shadow: 0 5px 20px rgba(67,217,245,.12);
}

div.stButton > button:focus,
div.stDownloadButton > button:focus {
    box-shadow: 0 0 0 2px rgba(67,217,245,.22);
}

[data-testid="stAlert"] {
    border-radius: 12px;
    border: 1px solid var(--dg-border);
}

[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid var(--dg-border);
    border-radius: 12px;
    overflow: hidden;
}

[data-testid="stImage"] img {
    border: 1px solid rgba(67,217,245,.23);
    border-radius: 15px;
    box-shadow: 0 10px 32px rgba(0,0,0,.24);
}

hr {
    border-color: rgba(81,190,220,.17) !important;
}

.stProgress > div > div > div {
    background: linear-gradient(90deg, #22b9e8, #3ce6c2);
}

@keyframes dg-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(60,230,194,.20); }
    50% { box-shadow: 0 0 0 6px rgba(60,230,194,0); }
}

.dg-status {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    padding: 9px 13px;
    border: 1px solid rgba(60,230,194,.25);
    border-radius: 999px;
    background: rgba(20,110,100,.13);
    color: #aaf9e7;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: .13em;
}

.dg-status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--dg-teal);
    animation: dg-pulse 2s infinite;
}

.dg-hero {
    position: relative;
    padding: 18px 22px;
    margin: 4px 0 22px;
    border: 1px solid var(--dg-border);
    border-radius: 18px;
    background: linear-gradient(110deg, rgba(17,39,59,.93), rgba(9,24,36,.72));
}

.dg-kicker {
    color: var(--dg-cyan);
    font-size: 10px;
    font-weight: 800;
    letter-spacing: .22em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.dg-hero-title {
    color: #f0f9ff;
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(19px, 2.5vw, 27px);
    font-weight: 700;
    line-height: 1.4;
}

.dg-hero-copy {
    color: #91aabd;
    font-size: 13px;
    line-height: 1.7;
    margin-top: 7px;
}

@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    [data-testid="stMetric"] {
        padding: 12px;
        border-radius: 12px;
    }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: .01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: .01ms !important;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================

DEFAULTS = {
    "monitoring": False,
    "camera": None,
    "landmarker": None,
    "eye_analyzer": None,
    "yawn_analyzer": None,
    "head_analyzer": None,
    "state_manager": None,
    "alert_manager": None,
    "logger": None,
    "session_start": None,
    "last_frame": None,
    "last_metrics": {},
    "last_session_id": None,
    "last_summary": None,
    "last_timestamp_ms": -1,
    "previous_frame_time": None,
    "fps": 0.0,
    "dashboard_error": None,
    "risk_history": [],
    "last_risk_sample_time": 0.0,
    "pdf_report_bytes": None,
    "pdf_report_name": None,
    "eye_closure_count": 0,
    "eye_closure_total_seconds": 0.0,
    "previous_eye_closed": False,
    "last_eye_state_time": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# CLEANUP
# =========================================================

def cleanup_monitoring():
    """Stop monitoring and safely release hardware resources."""

    was_monitoring = st.session_state.monitoring
    start = st.session_state.session_start
    logger = st.session_state.logger

    # Complete an eye-closure interval if the session stops
    # while the driver's eyes are still classified as closed.
    if (
        was_monitoring
        and st.session_state.previous_eye_closed
        and st.session_state.last_eye_state_time is not None
    ):
        closed_for = max(
            0.0,
            time.monotonic() - st.session_state.last_eye_state_time,
        )
        st.session_state.eye_closure_total_seconds += closed_for
        st.session_state.eye_closure_count += 1

    if was_monitoring and start is not None and logger is not None:
        duration = time.monotonic() - start

        st.session_state.last_session_id = logger.session_id
        st.session_state.last_summary = str(logger.summary_path)

        try:
            logger.finish(duration)
        except Exception as exc:
            print("Session logging error:", exc)

    alert = st.session_state.alert_manager
    if alert is not None:
        try:
            alert.close()
        except Exception as exc:
            print("Alert cleanup error:", exc)

    landmarker = st.session_state.landmarker
    if landmarker is not None:
        try:
            landmarker.close()
        except Exception as exc:
            print("Landmarker cleanup error:", exc)

    camera = st.session_state.camera
    if camera is not None:
        try:
            camera.release()
        except Exception as exc:
            print("Camera cleanup error:", exc)

    st.session_state.monitoring = False
    st.session_state.camera = None
    st.session_state.landmarker = None
    st.session_state.eye_analyzer = None
    st.session_state.yawn_analyzer = None
    st.session_state.head_analyzer = None
    st.session_state.state_manager = None
    st.session_state.alert_manager = None
    st.session_state.logger = None
    st.session_state.session_start = None
    st.session_state.previous_frame_time = None
    st.session_state.last_timestamp_ms = -1
    st.session_state.previous_eye_closed = False
    st.session_state.last_eye_state_time = None


# =========================================================
# START MONITORING
# =========================================================

def start_monitoring():
    """Initialize the camera, MediaPipe and analysis modules."""

    if st.session_state.monitoring:
        return

    if not Path(MODEL_PATH).is_file():
        st.session_state.dashboard_error = (
            f"Face Landmarker model not found: {MODEL_PATH}"
        )
        return

    camera = None
    landmarker = None
    alert = None
    logger = None

    try:
        # CAP_DSHOW is appropriate on Windows. Use the default
        # backend on Linux/macOS.
        if os.name == "nt":
            camera = cv2.VideoCapture(
                CAMERA_INDEX,
                cv2.CAP_DSHOW,
            )
        else:
            camera = cv2.VideoCapture(CAMERA_INDEX)

        if camera is None or not camera.isOpened():
            if camera is not None:
                camera.release()

            st.session_state.dashboard_error = (
                "Cannot open webcam. Check camera permissions, "
                "CAMERA_INDEX in config.py, and whether another "
                "application is using the camera."
            )
            return

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(MODEL_PATH)
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        landmarker = (
            mp.tasks.vision.FaceLandmarker.create_from_options(options)
        )

        alert = AlertManager(ALERT_SOUND_PATH)
        logger = EventLogger(LOG_DIR)

        # Assign resources to session state after initialization.
        st.session_state.camera = camera
        st.session_state.landmarker = landmarker
        st.session_state.alert_manager = alert
        st.session_state.logger = logger

        st.session_state.eye_analyzer = EyeAnalyzer(
            ear_threshold=EAR_THRESHOLD,
            closed_duration=CLOSED_DURATION,
            smoothing_frames=SMOOTHING_FRAMES,
        )
        st.session_state.yawn_analyzer = YawnAnalyzer(
            threshold=YAWN_THRESHOLD,
            duration=YAWN_DURATION,
            cooldown=YAWN_COOLDOWN,
        )
        st.session_state.head_analyzer = HeadPoseAnalyzer(
            tilt_threshold=HEAD_TILT_THRESHOLD,
            duration=HEAD_TILT_DURATION,
            cooldown=HEAD_TILT_COOLDOWN,
        )
        st.session_state.state_manager = DriverStateManager()

        st.session_state.session_start = time.monotonic()
        st.session_state.previous_frame_time = None
        st.session_state.last_timestamp_ms = -1
        st.session_state.last_frame = None
        st.session_state.last_metrics = {}
        st.session_state.risk_history = []
        st.session_state.last_risk_sample_time = 0.0
        st.session_state.fps = 0.0
        st.session_state.pdf_report_bytes = None
        st.session_state.pdf_report_name = None
        st.session_state.eye_closure_count = 0
        st.session_state.eye_closure_total_seconds = 0.0
        st.session_state.previous_eye_closed = False
        st.session_state.last_eye_state_time = None
        st.session_state.dashboard_error = None
        st.session_state.monitoring = True

    except Exception as exc:
        if logger is not None:
            try:
                logger.finish(0)
            except Exception:
                pass

        if alert is not None:
            try:
                alert.close()
            except Exception:
                pass

        if landmarker is not None:
            try:
                landmarker.close()
            except Exception:
                pass

        if camera is not None:
            try:
                camera.release()
            except Exception:
                pass

        st.session_state.camera = None
        st.session_state.landmarker = None
        st.session_state.alert_manager = None
        st.session_state.logger = None
        st.session_state.monitoring = False
        st.session_state.dashboard_error = (
            f"Could not start monitoring: {exc}"
        )


# =========================================================
# PROCESS ONE CAMERA FRAME
# =========================================================

def process_frame(frame):
    """Analyze one frame and update session metrics."""

    if st.session_state.session_start is None:
        raise RuntimeError("Monitoring session has not started.")

    now = time.monotonic()
    elapsed = now - st.session_state.session_start

    previous = st.session_state.previous_frame_time
    if previous is not None and now > previous:
        instant_fps = 1.0 / (now - previous)
        old_fps = st.session_state.fps
        st.session_state.fps = (
            instant_fps
            if old_fps == 0
            else 0.85 * old_fps + 0.15 * instant_fps
        )

    st.session_state.previous_frame_time = now

    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )

    timestamp_ms = max(
        int(elapsed * 1000),
        st.session_state.last_timestamp_ms + 1,
    )
    st.session_state.last_timestamp_ms = timestamp_ms

    result = st.session_state.landmarker.detect_for_video(
        image,
        timestamp_ms,
    )
    face_present = bool(result.face_landmarks)

    eyes = {
        "ear": None,
        "eyes_closed": False,
        "drowsy": False,
        "closed_seconds": 0.0,
    }
    mouth = {
        "mar": None,
        "yawning": False,
        "yawn_event": False,
        "yawn_seconds": 0.0,
    }
    head = {
        "yaw": None,
        "pitch": None,
        "roll": None,
        "head_tilt": False,
        "tilt_event": False,
        "tilt_seconds": 0.0,
    }

    if face_present:
        landmarks = result.face_landmarks[0]

        left_ear = eye_aspect_ratio(
            landmarks, LEFT_EYE, width, height
        )
        right_ear = eye_aspect_ratio(
            landmarks, RIGHT_EYE, width, height
        )

        eyes = st.session_state.eye_analyzer.update(
            left_ear,
            right_ear,
            now,
        )

        mar = mouth_aspect_ratio(landmarks, width, height)
        mouth = st.session_state.yawn_analyzer.update(mar, now)

        pose = estimate_head_pose(landmarks, width, height)
        head = st.session_state.head_analyzer.update(pose, now)

        for index in LEFT_EYE + RIGHT_EYE:
            point = landmarks[index]
            cv2.circle(
                frame,
                (int(point.x * width), int(point.y * height)),
                3,
                (60, 240, 200),
                -1,
            )

        for index in (13, 14, 78, 308):
            point = landmarks[index]
            cv2.circle(
                frame,
                (int(point.x * width), int(point.y * height)),
                3,
                (255, 190, 60),
                -1,
            )

    else:
        st.session_state.eye_analyzer.reset()
        st.session_state.yawn_analyzer.reset()
        st.session_state.head_analyzer.reset()

    # Track completed eye-closure intervals for analytics.
    eyes_closed = bool(face_present and eyes["eyes_closed"])
    was_closed = st.session_state.previous_eye_closed

    if eyes_closed and not was_closed:
        st.session_state.last_eye_state_time = now

    elif was_closed and not eyes_closed:
        start_closed = st.session_state.last_eye_state_time

        if start_closed is not None:
            closed_for = max(0.0, now - start_closed)
            st.session_state.eye_closure_total_seconds += closed_for
            st.session_state.eye_closure_count += 1

        st.session_state.last_eye_state_time = None

    st.session_state.previous_eye_closed = eyes_closed

    signals = {
        "face_present": face_present,
        "eyes_closed": eyes["eyes_closed"],
        "drowsy": eyes["drowsy"],
        "yawning": mouth["yawning"],
        "head_tilt": head["head_tilt"],
    }

    state = st.session_state.state_manager.update(signals, now)
    risk = state["risk_score"]

    # Sample the risk score approximately twice per second.
    last_sample = st.session_state.last_risk_sample_time

    if elapsed - last_sample >= 0.5 or not st.session_state.risk_history:
        st.session_state.risk_history.append({
            "elapsed_seconds": round(elapsed, 2),
            "risk_score": float(risk),
            "status": state["status"],
        })
        st.session_state.last_risk_sample_time = elapsed

    # Bound memory use during long sessions.
    if len(st.session_state.risk_history) > 7200:
        st.session_state.risk_history = (
            st.session_state.risk_history[-7200:]
        )

    st.session_state.alert_manager.update(eyes["drowsy"])

    logger = st.session_state.logger

    for event in state["events"]:
        logger.log_event(
            event,
            risk,
            elapsed,
            state["reasons"],
        )

    if mouth["yawn_event"]:
        logger.log_event(
            "YAWN_DETECTED",
            risk,
            elapsed,
            f"MAR={mouth['mar']:.3f}",
        )

    if head["tilt_event"]:
        logger.log_event(
            "PROLONGED_HEAD_DEVIATION",
            risk,
            elapsed,
            f"Yaw={head['yaw']:.1f}, Pitch={head['pitch']:.1f}",
        )

    if not face_present:
        banner = "FACE NOT DETECTED"
        color = (0, 165, 255)
    elif eyes["drowsy"]:
        banner = "DROWSINESS ALERT!"
        color = (0, 0, 255)
    elif eyes["eyes_closed"]:
        banner = "EYES CLOSED"
        color = (0, 165, 255)
    elif mouth["yawning"]:
        banner = "POSSIBLE YAWN"
        color = (0, 165, 255)
    elif head["head_tilt"]:
        banner = "HEAD DEVIATION"
        color = (0, 165, 255)
    else:
        banner = "MONITORING ACTIVE"
        color = (50, 210, 100)

    cv2.rectangle(
        frame,
        (0, 0),
        (width, 48),
        (15, 25, 40),
        -1,
    )
    cv2.putText(
        frame,
        f"DRIVERGUARD AI | {banner}",
        (14, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )

    metrics = {
        "risk": risk,
        "status": state["status"],
        "face_present": face_present,
        "ear": eyes["ear"],
        "mar": mouth["mar"],
        "closed_seconds": eyes["closed_seconds"],
        "yaw": head["yaw"],
        "pitch": head["pitch"],
        "roll": head["roll"],
        "warning": banner,
        "elapsed": elapsed,
        "fps": st.session_state.fps,
        "event_count": logger.event_count,
        "eye_closure_count": st.session_state.eye_closure_count,
        "eye_closure_total_seconds": (
            st.session_state.eye_closure_total_seconds
        ),
    }

    st.session_state.last_metrics = metrics
    st.session_state.last_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )


# =========================================================
# HISTORY AND EVENT TABLES
# =========================================================

def load_event_files():
    return sorted(
        glob.glob(str(Path(LOG_DIR) / "events_*.csv")),
        reverse=True,
    )


def load_session_events():
    frames = []

    for file in load_event_files()[:20]:
        try:
            data = pd.read_csv(file)
            required = {
                "elapsed_seconds",
                "risk_score",
                "event",
            }

            if data.empty or not required.issubset(data.columns):
                continue

            data["session"] = Path(file).stem.replace(
                "events_", ""
            )
            frames.append(data)

        except (
            OSError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
        ):
            continue

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def get_current_session_events():
    all_events = load_session_events()

    if all_events.empty:
        return all_events

    if (
        st.session_state.monitoring
        and st.session_state.logger is not None
    ):
        session_id = st.session_state.logger.session_id
    else:
        session_id = st.session_state.last_session_id

    if session_id:
        data = all_events[
            all_events["session"] == session_id
        ].copy()
    else:
        newest = all_events.iloc[0]["session"]
        data = all_events[
            all_events["session"] == newest
        ].copy()

    data["elapsed_seconds"] = pd.to_numeric(
        data["elapsed_seconds"],
        errors="coerce",
    )
    data["risk_score"] = pd.to_numeric(
        data["risk_score"],
        errors="coerce",
    )

    return data.dropna(
        subset=["elapsed_seconds", "risk_score"]
    )


def render_history():
    """Display live risk trends or recorded event history."""

    history = st.session_state.risk_history

    if history:
        chart = pd.DataFrame(history)
        chart = chart[
            ["elapsed_seconds", "risk_score"]
        ].copy()
        chart = chart.dropna(
            subset=["elapsed_seconds", "risk_score"]
        )
        chart = chart.sort_values("elapsed_seconds")
        chart = chart.set_index("elapsed_seconds")
        chart.index.name = "Elapsed time (seconds)"
        chart.columns = ["Driver Risk Score"]

        st.line_chart(chart, height=300)

        st.caption(
            "Live session risk trend · Samples collected "
            "approximately every 0.5 seconds"
        )
        return

    data = get_current_session_events()

    if data.empty:
        st.info(
            "Start monitoring to begin collecting risk history."
        )
        return

    chart = data[
        ["elapsed_seconds", "risk_score"]
    ].sort_values("elapsed_seconds")
    chart = chart.set_index("elapsed_seconds")
    chart.index.name = "Elapsed time (seconds)"
    chart.columns = ["Recorded Event Risk"]

    st.line_chart(chart, height=300)
    st.caption(
        "Showing recorded event risks because live samples "
        "are not available."
    )


def render_recent_events():
    data = get_current_session_events()

    if data.empty:
        st.info("No events recorded yet.")
        return

    recent = data.sort_values(
        "elapsed_seconds",
        ascending=False,
    ).head(8)

    columns = [
        column
        for column in [
            "elapsed_seconds",
            "event",
            "risk_score",
            "details",
        ]
        if column in recent.columns
    ]

    st.dataframe(
        recent[columns],
        use_container_width=True,
        hide_index=True,
    )

    csv_data = data.to_csv(index=False).encode("utf-8")

    if (
        st.session_state.monitoring
        and st.session_state.logger is not None
    ):
        session_id = st.session_state.logger.session_id
    else:
        session_id = (
            st.session_state.last_session_id or "session"
        )

    st.download_button(
        "⬇️ Download Session Events",
        data=csv_data,
        file_name=f"driverguard_{session_id}.csv",
        mime="text/csv",
        key="download_events",
    )


# =========================================================
# PDF SESSION REPORT
# =========================================================

def create_pdf_report_bytes():
    """Build a PDF for the active or last completed session."""

    logger = st.session_state.logger

    if st.session_state.monitoring and logger is not None:
        session_id = logger.session_id
        duration = (
            time.monotonic() - st.session_state.session_start
            if st.session_state.session_start is not None
            else 0
        )
        total_events = logger.event_count
    else:
        session_id = st.session_state.last_session_id
        metrics = st.session_state.last_metrics
        duration = metrics.get("elapsed", 0)
        total_events = 0

    if not session_id:
        raise ValueError(
            "Pehle monitoring session start karo."
        )

    log_dir = Path(LOG_DIR)
    events_path = log_dir / f"events_{session_id}.csv"

    events = []
    if events_path.exists():
        try:
            events = pd.read_csv(events_path).fillna("").to_dict(
                orient="records"
            )
        except (
            OSError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
        ):
            events = []

    history = list(st.session_state.risk_history)
    risk_values = [
        float(row.get("risk_score", 0))
        for row in history
        if row.get("risk_score") is not None
    ]

    summary = {
        "duration_seconds": duration,
        "average_risk": (
            sum(risk_values) / len(risk_values)
            if risk_values
            else 0
        ),
        "maximum_risk": max(risk_values) if risk_values else 0,
        "total_events": total_events or len(events),
    }

    summary_path = st.session_state.last_summary

    if not st.session_state.monitoring and summary_path:
        saved_path = Path(summary_path)

        if saved_path.exists():
            try:
                saved_data = pd.read_csv(saved_path)
                if not saved_data.empty:
                    summary.update(saved_data.iloc[0].to_dict())
            except (
                OSError,
                pd.errors.EmptyDataError,
                pd.errors.ParserError,
            ):
                pass

    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = (
            Path(temp_dir) / f"driverguard_{session_id}.pdf"
        )

        generate_session_report(
            output_path=pdf_path,
            session_summary=summary,
            risk_history=history,
            events=events,
            driver_name="Driver",
            eye_closure_seconds=(
                st.session_state.eye_closure_total_seconds
            ),
            eye_closure_count=(
                st.session_state.eye_closure_count
            ),
        )

        return (
            pdf_path.read_bytes(),
            f"DriverGuard_Report_{session_id}.pdf",
        )


# =========================================================
# LIVE DASHBOARD
# =========================================================

@st.fragment(run_every=0.15)
def live_dashboard():
    camera_col, metrics_col = st.columns(
        [1.45, 1],
        gap="large",
    )

    with camera_col:
        st.subheader("📹 Live Camera")

        if st.session_state.monitoring:
            try:
                camera = st.session_state.camera

                if camera is None or not camera.isOpened():
                    st.error("Webcam is unavailable.")
                    return

                ok, frame = camera.read()

                if not ok or frame is None:
                    st.error(
                        "Webcam stopped responding. "
                        "Stop monitoring and try again."
                    )
                    return

                process_frame(frame)

                st.image(
                    st.session_state.last_frame,
                    channels="RGB",
                    use_container_width=True,
                )
                st.caption(
                    "Live webcam · Facial landmarks enabled"
                )

            except Exception as exc:
                st.session_state.dashboard_error = str(exc)
                cleanup_monitoring()
                st.error(f"Monitoring stopped: {exc}")
                return

        elif st.session_state.last_frame is not None:
            st.image(
                st.session_state.last_frame,
                channels="RGB",
                use_container_width=True,
            )
            st.caption("Last captured frame · Camera is off.")
        else:
            st.info(
                "📷 Camera is off. Click Start Monitoring "
                "in the sidebar to begin."
            )

    with metrics_col:
        st.subheader("📊 Driver Status")
        metrics = st.session_state.last_metrics

        if not metrics:
            st.metric("Driver Risk Score", "—")
            st.info(
                "Start a session to view live measurements."
            )
        else:
            st.metric(
                "Driver Risk Score",
                f"{metrics['risk']}/100",
            )

            if metrics["status"] == "HIGH RISK":
                st.error("🔴 HIGH RISK")
            elif metrics["status"] == "MODERATE RISK":
                st.warning("🟠 MODERATE RISK")
            else:
                st.success("🟢 LOW RISK")

            ear = metrics["ear"]
            mar = metrics["mar"]
            yaw = metrics["yaw"]
            pitch = metrics["pitch"]

            a, b = st.columns(2)
            a.metric(
                "Eye Aspect Ratio",
                f"{ear:.3f}" if ear is not None else "—",
            )
            b.metric(
                "Mouth Aspect Ratio",
                f"{mar:.3f}" if mar is not None else "—",
            )

            c, d = st.columns(2)
            c.metric(
                "Eyes Closed",
                f"{metrics['closed_seconds']:.1f}s",
            )
            d.metric(
                "Session Time",
                f"{int(metrics['elapsed'] // 60):02d}:"
                f"{int(metrics['elapsed'] % 60):02d}",
            )

            e, f = st.columns(2)
            e.metric(
                "Head Yaw",
                f"{yaw:.1f}°" if yaw is not None else "—",
            )
            f.metric(
                "Head Pitch",
                f"{pitch:.1f}°" if pitch is not None else "—",
            )

            st.info(
                f"**Alert:** {metrics['warning']}  \n"
                f"**Face detected:** "
                f"{'Yes' if metrics['face_present'] else 'No'}  \n"
                f"**FPS:** {metrics['fps']:.1f}  \n"
                f"**Logged events:** {metrics['event_count']}"
            )

    st.divider()

    chart_col, events_col = st.columns(
        [1.15, 1],
        gap="large",
    )

    with chart_col:
        st.subheader("📈 Recorded Risk History")
        render_history()

    with events_col:
        st.subheader("⚠️ Recent Events")
        render_recent_events()


# =========================================================
# SIDEBAR CONTROLS
# =========================================================

with st.sidebar:
    st.markdown("## 🛡️ DriverGuard AI")
    st.caption("Intelligent Driver Monitoring")
    st.divider()

    st.markdown("### Control Center")

    if not st.session_state.monitoring:
        if st.button(
            "▶ Start Monitoring",
            type="primary",
            use_container_width=True,
        ):
            start_monitoring()
            st.rerun()
    else:
        st.success("Camera monitoring is active")

        if st.button(
            "■ Stop Monitoring",
            type="primary",
            use_container_width=True,
        ):
            cleanup_monitoring()
            st.rerun()

    if st.session_state.dashboard_error:
        st.error(st.session_state.dashboard_error)

    st.divider()
    st.markdown("### Detection Modules")
    st.write("🟢 Eye closure / drowsiness")
    st.write("🟢 Possible yawning")
    st.write("🟢 Head-pose deviation")
    st.write("🟢 CSV event logging")
    st.divider()
    st.caption("DRIVERGUARD AI  •  SAFETY ANALYTICS")


# =========================================================
# MAIN PAGE
# =========================================================

st.markdown(
    """
<div class="dg-hero">
    <div class="dg-kicker">Intelligent mobility · Driver safety</div>
    <div class="dg-hero-title">DRIVERGUARD AI</div>
    <div class="dg-hero-copy">
        Real-time driver awareness · Visual analytics ·
        Session intelligence
    </div>
    <div style="margin-top:15px;">
        <span class="dg-status">
            <span class="dg-status-dot"></span>
            DRIVER MONITORING SYSTEM
        </span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.subheader("Live Operations")
live_dashboard()

st.divider()
st.subheader("📋 Last Session Summary")

if st.session_state.last_summary:
    summary_path = Path(st.session_state.last_summary)

    if summary_path.exists():
        try:
            summary = pd.read_csv(summary_path)
            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True,
            )
        except Exception as exc:
            st.warning(
                f"Could not read session summary: {exc}"
            )
    else:
        st.caption(
            "The last session summary is not available."
        )
else:
    st.caption(
        "Stop a monitoring session to save and view its summary."
    )


st.subheader("📄 PDF Session Report")

has_session = (
    st.session_state.monitoring
    or st.session_state.last_session_id is not None
)

if not has_session:
    st.caption(
        "PDF report generate karne ke liye "
        "monitoring session start karo."
    )
else:
    if st.button(
        "🧾 Generate PDF Report",
        type="primary",
        key="generate_pdf_report",
    ):
        try:
            pdf_bytes, pdf_name = create_pdf_report_bytes()
            st.session_state.pdf_report_bytes = pdf_bytes
            st.session_state.pdf_report_name = pdf_name
            st.success("PDF report ready hai.")
        except Exception as exc:
            st.error(
                f"PDF report generate nahi hui: {exc}"
            )

    if st.session_state.get("pdf_report_bytes"):
        st.download_button(
            "⬇️ Download PDF Report",
            data=st.session_state.pdf_report_bytes,
            file_name=(
                st.session_state.pdf_report_name
                or "DriverGuard_Report.pdf"
            ),
            mime="application/pdf",
            key="download_pdf_report",
        )

st.divider()
st.caption(
    "DriverGuard AI provides driver-awareness indicators. "
    "Risk scores are estimates, not validated measures of "
    "fitness to drive. Always follow safe-driving practices."
)

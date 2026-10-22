
from pathlib import Path
import glob
import queue
import threading
import tempfile
import time

import av
import cv2
import mediapipe as mp
import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from config import (
    MODEL_PATH,
    ALERT_SOUND_PATH,
    LOG_DIR,
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
from vision.mouth_analyzer import (
    YawnAnalyzer,
    mouth_aspect_ratio,
)
from vision.head_pose import (
    HeadPoseAnalyzer,
    estimate_head_pose,
)
from monitoring.alert_manager import AlertManager
from monitoring.state_manager import DriverStateManager
from monitoring.event_logger import EventLogger


# =========================================================
# PAGE CONFIGURATION
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

    .stApp {
        background:
          radial-gradient(ellipse at 15% 0%,rgba(17,101,139,.19),transparent 35%),
          radial-gradient(ellipse at 95% 15%,rgba(16,174,160,.10),transparent 30%),
          linear-gradient(145deg,#070b14,#0a1220 55%,#071019);
        color:#e8f4fc;
        font-family:Inter,sans-serif;
    }

    .block-container {
        max-width:1600px;
        padding-top:1.5rem;
        padding-bottom:3rem;
    }

    h1,h2,h3 {
        color:#f1f8ff!important;
        letter-spacing:-.035em;
    }

    h1 {
        font-family:Orbitron,sans-serif!important;
    }

    p,label,[data-testid="stCaptionContainer"] {
        color:#91a6ba;
    }

    [data-testid="stMetric"] {
        background:linear-gradient(145deg,#132236,#0d1726);
        border:1px solid rgba(81,190,220,.2);
        border-radius:16px;
        padding:16px;
    }

    [data-testid="stMetricLabel"] {
        color:#9ab0c5!important;
    }

    [data-testid="stMetricValue"] {
        color:#eafaff!important;
        font-family:Orbitron,sans-serif;
    }

    section[data-testid="stSidebar"] {
        background:linear-gradient(180deg,#0b1422,#080e18);
        border-right:1px solid rgba(81,190,220,.2);
    }

    div.stButton>button,
    div.stDownloadButton>button {
        min-height:43px;
        border:1px solid rgba(67,217,245,.35);
        border-radius:11px;
        background:linear-gradient(115deg,#123d54,#12534f);
        color:#effcff;
        font-weight:700;
    }

    div.stButton>button:hover,
    div.stDownloadButton>button:hover {
        border-color:#43d9f5;
        filter:brightness(1.12);
    }

    hr {
        border-color:rgba(81,190,220,.17)!important;
    }

    .dg-hero {
        padding:22px;
        margin:4px 0 22px;
        border:1px solid rgba(81,190,220,.2);
        border-radius:18px;
        background:linear-gradient(
            110deg,
            rgba(17,39,59,.93),
            rgba(9,24,36,.72)
        );
    }

    .dg-kicker {
        color:#43d9f5;
        font-size:10px;
        font-weight:800;
        letter-spacing:.22em;
    }

    .dg-title {
        color:#f0f9ff;
        font-family:Orbitron,sans-serif;
        font-size:28px;
        font-weight:800;
    }

    .dg-copy {
        color:#91aabd;
        font-size:13px;
        line-height:1.7;
        margin-top:8px;
    }

    .dg-status {
        border:1px solid rgba(81,190,220,.24);
        background:rgba(15,35,52,.75);
        padding:12px 15px;
        border-radius:12px;
        color:#b9d2e4;
        font-size:13px;
        line-height:1.6;
        margin:8px 0 12px;
    }

    /* Keep the WebRTC video clean and responsive. */
    div[data-testid="stWebRtc"] video {
        width:100%;
        border-radius:16px;
        object-fit:cover;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# EYE AND LIP LANDMARK DOTS ONLY
# =========================================================

# No full-face mesh and no connecting lines.

OUTER_LIP_CONTOUR = [
    61, 146, 91, 181, 84, 17, 314, 405,
    321, 375, 291, 409, 270, 269, 267, 0,
    37, 39, 40, 185,
]

INNER_LIP_CONTOUR = [
    78, 95, 88, 178, 87, 14, 317, 402,
    318, 324, 308, 415, 310, 311, 312, 13,
    82, 81, 80, 191,
]


def draw_eye_lip_landmarks(frame, landmarks):
    """Draw green dots only around the eyes and lips."""

    if not landmarks:
        return frame

    height, width = frame.shape[:2]

    landmark_indices = (
        set(LEFT_EYE)
        | set(RIGHT_EYE)
        | set(OUTER_LIP_CONTOUR)
        | set(INNER_LIP_CONTOUR)
    )

    for index in landmark_indices:
        if index < 0 or index >= len(landmarks):
            continue

        landmark = landmarks[index]

        x = int(round(landmark.x * width))
        y = int(round(landmark.y * height))

        if 0 <= x < width and 0 <= y < height:
            cv2.circle(
                frame,
                (x, y),
                2,
                (0, 255, 0),
                -1,
                cv2.LINE_AA,
            )

    return frame


# =========================================================
# SESSION STATE
# =========================================================

DEFAULTS = {
    "monitoring": False,
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
    "video_callback": None,
    "callback_output_cache": None,
    "frame_processing_lock": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

if "incoming_frames" not in st.session_state:
    st.session_state.incoming_frames = queue.Queue(maxsize=1)

if "processed_frames" not in st.session_state:
    st.session_state.processed_frames = queue.Queue(maxsize=1)

if st.session_state.frame_processing_lock is None:
    st.session_state.frame_processing_lock = threading.Lock()

if "frame_lock" not in st.session_state:
    st.session_state.frame_lock = threading.Lock()

if st.session_state.callback_output_cache is None:
    st.session_state.callback_output_cache = {"frame": None}


def put_latest(target_queue, item):
    """Keep the newest frame and discard stale queued frames."""

    try:
        target_queue.put_nowait(item)
        return
    except queue.Full:
        pass

    try:
        target_queue.get_nowait()
    except queue.Empty:
        pass

    try:
        target_queue.put_nowait(item)
    except queue.Full:
        pass


def drain_queue(target_queue):
    while True:
        try:
            target_queue.get_nowait()
        except queue.Empty:
            break


# =========================================================
# WEBRTC VIDEO CALLBACK
# =========================================================

def browser_video_callback(
    incoming_queue,
    processed_queue,
    output_cache,
):
    """Receive browser camera frames and return processed frames."""

    def callback(video_frame):
        try:
            frame_bgr = video_frame.to_ndarray(
                format="bgr24"
            )

            put_latest(incoming_queue, frame_bgr)

            try:
                output_bgr = processed_queue.get_nowait()
                output_cache["frame"] = output_bgr
            except queue.Empty:
                output_bgr = output_cache.get("frame")

            if output_bgr is None:
                return video_frame

            return av.VideoFrame.from_ndarray(
                output_bgr,
                format="bgr24",
            )

        except Exception:
            return video_frame

    return callback


# =========================================================
# CLEANUP MONITORING
# =========================================================

def cleanup_monitoring():
    was_monitoring = st.session_state.monitoring
    start = st.session_state.session_start
    logger = st.session_state.logger

    if (
        was_monitoring
        and st.session_state.previous_eye_closed
        and st.session_state.last_eye_state_time is not None
    ):
        closed_for = max(
            0,
            time.monotonic()
            - st.session_state.last_eye_state_time,
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

    for key, method in (
        ("alert_manager", "close"),
        ("landmarker", "close"),
    ):
        obj = st.session_state.get(key)

        if obj is not None:
            try:
                getattr(obj, method)()
            except Exception as exc:
                print(f"{key} cleanup error:", exc)

    for key in (
        "landmarker",
        "eye_analyzer",
        "yawn_analyzer",
        "head_analyzer",
        "state_manager",
        "alert_manager",
        "logger",
        "session_start",
        "video_callback",
    ):
        st.session_state[key] = None

    st.session_state.monitoring = False
    st.session_state.previous_frame_time = None
    st.session_state.last_timestamp_ms = -1
    st.session_state.previous_eye_closed = False
    st.session_state.last_eye_state_time = None
    st.session_state.callback_output_cache = {"frame": None}

    drain_queue(st.session_state.incoming_frames)
    drain_queue(st.session_state.processed_frames)


# =========================================================
# START MONITORING
# =========================================================

def start_monitoring():
    if st.session_state.monitoring:
        return

    if not Path(MODEL_PATH).is_file():
        st.session_state.dashboard_error = (
            f"Face Landmarker model not found: {MODEL_PATH}"
        )
        return

    landmarker = None
    alert = None
    logger = None

    try:
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
            mp.tasks.vision.FaceLandmarker.create_from_options(
                options
            )
        )

        alert = AlertManager(ALERT_SOUND_PATH)
        logger = EventLogger(LOG_DIR)

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

        drain_queue(st.session_state.incoming_frames)
        drain_queue(st.session_state.processed_frames)

        output_cache = {"frame": None}
        st.session_state.callback_output_cache = output_cache

        st.session_state.video_callback = browser_video_callback(
            st.session_state.incoming_frames,
            st.session_state.processed_frames,
            output_cache,
        )

        st.session_state.monitoring = True

    except Exception as exc:
        for obj, method in (
            (logger, "finish"),
            (alert, "close"),
            (landmarker, "close"),
        ):
            if obj is not None:
                try:
                    if method == "finish":
                        obj.finish(0)
                    else:
                        getattr(obj, method)()
                except Exception:
                    pass

        st.session_state.monitoring = False
        st.session_state.dashboard_error = (
            f"Could not start monitoring: {exc}"
        )


# =========================================================
# FRAME ANALYSIS
# =========================================================

def process_frame(frame):
    if (
        st.session_state.session_start is None
        or st.session_state.landmarker is None
    ):
        return

    processing_lock = st.session_state.frame_processing_lock

    if not processing_lock.acquire(blocking=False):
        return

    try:
        _process_frame_locked(frame)
    finally:
        processing_lock.release()


def _process_frame_locked(frame):
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

    # Mirror the webcam image.
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

        # Green landmark dots only; no connecting lines.
        draw_eye_lip_landmarks(frame, landmarks)

        left_ear = eye_aspect_ratio(
            landmarks,
            LEFT_EYE,
            width,
            height,
        )

        right_ear = eye_aspect_ratio(
            landmarks,
            RIGHT_EYE,
            width,
            height,
        )

        eyes = st.session_state.eye_analyzer.update(
            left_ear,
            right_ear,
            now,
        )

        mar = mouth_aspect_ratio(
            landmarks,
            width,
            height,
        )

        mouth = st.session_state.yawn_analyzer.update(
            mar,
            now,
        )

        # Head pose analytics work independently of the overlay.
        pose = estimate_head_pose(
            landmarks,
            width,
            height,
        )

        head = st.session_state.head_analyzer.update(
            pose,
            now,
        )

    else:
        st.session_state.eye_analyzer.reset()
        st.session_state.yawn_analyzer.reset()
        st.session_state.head_analyzer.reset()

    # Eye-closure duration analytics.
    eyes_closed = bool(
        face_present and eyes["eyes_closed"]
    )

    was_closed = st.session_state.previous_eye_closed

    if eyes_closed and not was_closed:
        st.session_state.last_eye_state_time = now

    elif was_closed and not eyes_closed:
        closed_start = st.session_state.last_eye_state_time

        if closed_start is not None:
            st.session_state.eye_closure_total_seconds += max(
                0,
                now - closed_start,
            )
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

    state = st.session_state.state_manager.update(
        signals,
        now,
    )

    risk = state["risk_score"]

    # Sample risk history approximately twice per second.
    if (
        elapsed - st.session_state.last_risk_sample_time >= 0.5
        or not st.session_state.risk_history
    ):
        st.session_state.risk_history.append({
            "elapsed_seconds": round(elapsed, 2),
            "risk_score": float(risk),
            "status": state["status"],
        })

        st.session_state.last_risk_sample_time = elapsed

    st.session_state.risk_history = (
        st.session_state.risk_history[-7200:]
    )

    st.session_state.alert_manager.update(
        eyes["drowsy"]
    )

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
            f"Yaw={head.get('yaw')}, Pitch={head.get('pitch')}",
        )

    # Determine the status shown in the camera overlay.
    if not face_present:
        banner, color = "FACE NOT DETECTED", (0, 165, 255)
    elif eyes["drowsy"]:
        banner, color = "DROWSINESS ALERT", (0, 0, 255)
    elif eyes["eyes_closed"]:
        banner, color = "EYES CLOSED", (0, 165, 255)
    elif mouth["yawning"]:
        banner, color = "POSSIBLE YAWN", (0, 165, 255)
    elif head["head_tilt"]:
        banner, color = "HEAD DEVIATION", (0, 165, 255)
    else:
        banner, color = "MONITORING ACTIVE", (50, 210, 100)

    # Professional camera-feed heading.
    header_height = 76

    cv2.rectangle(
        frame,
        (0, 0),
        (width, header_height),
        (10, 20, 32),
        -1,
    )

    # Cyan separator.
    cv2.line(
        frame,
        (0, header_height),
        (width, header_height),
        (70, 210, 245),
        2,
        cv2.LINE_AA,
    )

    # Main title.
    cv2.putText(
        frame,
        "DRIVERGUARD AI",
        (18, 29),
        cv2.FONT_HERSHEY_DUPLEX,
        0.72,
        (245, 250, 255),
        2,
        cv2.LINE_AA,
    )

    # Subtitle.
    cv2.putText(
        frame,
        "INTELLIGENT DRIVER MONITORING",
        (19, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (160, 190, 210),
        1,
        cv2.LINE_AA,
    )

    # Status badge; keep it within the frame on narrow screens.
    badge_width = min(250, max(150, width - 36))
    badge_x = max(18, width - badge_width - 18)

    cv2.rectangle(
        frame,
        (badge_x, 17),
        (width - 18, 56),
        (25, 40, 54),
        -1,
    )

    cv2.rectangle(
        frame,
        (badge_x, 17),
        (width - 18, 56),
        color,
        1,
    )

    cv2.putText(
        frame,
        banner,
        (badge_x + 9, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        color,
        1,
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
        "eye_closure_total_seconds":
            st.session_state.eye_closure_total_seconds,
    }

    with st.session_state.frame_lock:
        st.session_state.last_metrics = metrics
        st.session_state.last_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

    put_latest(
        st.session_state.processed_frames,
        frame.copy(),
    )


# =========================================================
# EVENT AND HISTORY DATA
# =========================================================

def load_event_files():
    return sorted(
        glob.glob(
            str(Path(LOG_DIR) / "events_*.csv")
        ),
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
                "events_",
                "",
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

    return pd.concat(
        frames,
        ignore_index=True,
    )


def get_current_session_events():
    data = load_session_events()

    if data.empty:
        return data

    if (
        st.session_state.monitoring
        and st.session_state.logger is not None
    ):
        session_id = st.session_state.logger.session_id
    else:
        session_id = st.session_state.last_session_id

    if session_id:
        data = data[
            data["session"] == session_id
        ].copy()
    else:
        newest = data.iloc[0]["session"]
        data = data[
            data["session"] == newest
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
        subset=[
            "elapsed_seconds",
            "risk_score",
        ]
    )


def render_history():
    history = st.session_state.risk_history

    if history:
        chart = pd.DataFrame(history)[
            [
                "elapsed_seconds",
                "risk_score",
            ]
        ].dropna().sort_values(
            "elapsed_seconds"
        )

        chart = chart.set_index(
            "elapsed_seconds"
        )

        chart.index.name = "Elapsed time (seconds)"
        chart.columns = ["Driver Risk Score"]

        st.line_chart(
            chart,
            height=300,
        )

        st.caption(
            "Live risk trend · Samples collected approximately "
            "every 0.5 seconds"
        )
        return

    data = get_current_session_events()

    if data.empty:
        st.info(
            "Risk history will appear here once monitoring begins."
        )
        return

    chart = data[
        [
            "elapsed_seconds",
            "risk_score",
        ]
    ].sort_values(
        "elapsed_seconds"
    ).set_index(
        "elapsed_seconds"
    )

    chart.index.name = "Elapsed time (seconds)"
    chart.columns = ["Recorded Event Risk"]

    st.line_chart(
        chart,
        height=300,
    )


def render_recent_events():
    data = get_current_session_events()

    if data.empty:
        st.info(
            "No events recorded for this session yet."
        )
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
        data=data.to_csv(index=False).encode("utf-8"),
        file_name=f"driverguard_{session_id}.csv",
        mime="text/csv",
        key="download_events",
    )


# =========================================================
# PDF SESSION REPORT
# =========================================================

def create_pdf_report_bytes():
    logger = st.session_state.logger

    if (
        st.session_state.monitoring
        and logger is not None
    ):
        session_id = logger.session_id

        duration = (
            time.monotonic()
            - st.session_state.session_start
            if st.session_state.session_start is not None
            else 0
        )

        total_events = logger.event_count

    else:
        session_id = st.session_state.last_session_id

        duration = st.session_state.last_metrics.get(
            "elapsed",
            0,
        )

        total_events = 0

    if not session_id:
        raise ValueError(
            "Generate a report after starting a monitoring session."
        )

    events_path = (
        Path(LOG_DIR)
        / f"events_{session_id}.csv"
    )

    events = []

    if events_path.exists():
        try:
            events = pd.read_csv(
                events_path
            ).fillna("").to_dict("records")
        except (
            OSError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
        ):
            pass

    history = list(st.session_state.risk_history)

    values = [
        float(row.get("risk_score", 0))
        for row in history
        if row.get("risk_score") is not None
    ]

    summary = {
        "duration_seconds": duration,
        "average_risk": (
            sum(values) / len(values)
            if values
            else 0
        ),
        "maximum_risk": (
            max(values)
            if values
            else 0
        ),
        "total_events": total_events or len(events),
    }

    summary_path = st.session_state.last_summary

    if (
        not st.session_state.monitoring
        and summary_path
    ):
        saved_path = Path(summary_path)

        if saved_path.exists():
            try:
                saved = pd.read_csv(saved_path)

                if not saved.empty:
                    summary.update(
                        saved.iloc[0].to_dict()
                    )
            except (
                OSError,
                pd.errors.EmptyDataError,
                pd.errors.ParserError,
            ):
                pass

    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = (
            Path(temp_dir)
            / f"driverguard_{session_id}.pdf"
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

@st.fragment(run_every=0.1)
def live_dashboard():
    camera_col, metrics_col = st.columns(
        [1.45, 1],
        gap="large",
    )

    with camera_col:
        st.subheader("📹 Live Camera Feed")

        if st.session_state.monitoring:
            ctx = webrtc_streamer(
                key="driverguard-browser-camera",
                mode=WebRtcMode.SENDRECV,
                video_frame_callback=(
                    st.session_state.video_callback
                ),
                media_stream_constraints={
                    "video": True,
                    "audio": False,
                },
                video_html_attrs={
                    "controls": False,
                    "autoPlay": True,
                    "playsInline": True,
                    "muted": True,
                    "style": {
                        "width": "100%",
                        "borderRadius": "16px",
                    },
                },
                async_processing=True,
                desired_playing_state=True,
            )

            if ctx.state.playing:
                try:
                    frame = (
                        st.session_state.incoming_frames.get_nowait()
                    )
                    process_frame(frame)

                except queue.Empty:
                    pass

                except Exception as exc:
                    st.session_state.dashboard_error = (
                        f"Frame processing error: {exc}"
                    )

                st.success(
                    "Camera connected · Live monitoring active"
                )

                st.caption(
                    "Eye and lip landmark dots · Eye-closure analytics "
                    "· Head-pose tracking"
                )

            else:
                st.markdown(
                    """
                    <div class="dg-status">
                        <strong>Camera connection pending</strong><br>
                        Allow browser camera permission if prompted.
                        The live feed will appear when the connection
                        becomes active.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:
            st.markdown(
                """
                <div class="dg-status">
                    <strong>Monitoring is stopped</strong><br>
                    Use the Start Monitoring control in the sidebar
                    to begin a new session.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with metrics_col:
        st.subheader("📊 Driver Status")

        with st.session_state.frame_lock:
            metrics = dict(st.session_state.last_metrics)

        if not metrics:
            st.metric(
                "Driver Risk Score",
                "—",
            )

            st.caption(
                "Live measurements will appear when the camera "
                "starts processing frames."
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

            a, b = st.columns(2)

            a.metric(
                "Eye Aspect Ratio",
                (
                    f"{metrics['ear']:.3f}"
                    if metrics["ear"] is not None
                    else "—"
                ),
            )

            b.metric(
                "Mouth Aspect Ratio",
                (
                    f"{metrics['mar']:.3f}"
                    if metrics["mar"] is not None
                    else "—"
                ),
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
                (
                    f"{metrics['yaw']:.1f}°"
                    if metrics["yaw"] is not None
                    else "—"
                ),
            )

            f.metric(
                "Head Pitch",
                (
                    f"{metrics['pitch']:.1f}°"
                    if metrics["pitch"] is not None
                    else "—"
                ),
            )

            st.metric(
                "Head Roll",
                (
                    f"{metrics['roll']:.1f}°"
                    if metrics["roll"] is not None
                    else "—"
                ),
            )

            st.info(
                f"**Alert:** {metrics['warning']}  \n"
                f"**Face detected:** "
                f"{'Yes' if metrics['face_present'] else 'No'}  \n"
                f"**FPS:** {metrics['fps']:.1f}  \n"
                f"**Logged events:** {metrics['event_count']}  \n"
                f"**Eye closures:** {metrics['eye_closure_count']}  \n"
                f"**Total closed time:** "
                f"{metrics['eye_closure_total_seconds']:.1f}s"
            )

    st.divider()

    chart_col, events_col = st.columns(
        [1.15, 1],
        gap="large",
    )

    with chart_col:
        st.subheader("📈 Risk Trend")
        render_history()

    with events_col:
        st.subheader("⚠️ Recent Events")
        render_recent_events()


# =========================================================
# SIDEBAR CONTROL CENTER
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
        st.success("Monitoring session initialized")

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

    st.write("🟢 Eye landmark tracking")
    st.write("🟢 Lip landmark tracking")
    st.write("🟢 Eye closure / drowsiness")
    st.write("🟢 Possible yawning")
    st.write("🟢 Head-pose deviation")
    st.write("🟢 Risk trend tracking")
    st.write("🟢 CSV event logging")
    st.write("🟢 PDF session reports")

    st.divider()
    st.caption("DRIVERGUARD AI • SAFETY ANALYTICS")


# =========================================================
# MAIN PAGE
# =========================================================

st.markdown(
    """
    <div class="dg-hero">
        <div class="dg-kicker">
            INTELLIGENT MOBILITY · DRIVER SAFETY
        </div>
        <div class="dg-title">
            DRIVERGUARD AI
        </div>
        <div class="dg-copy">
            Real-time driver awareness · Visual analytics ·
            Session intelligence
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
            "Last session summary is not available."
        )
else:
    st.caption(
        "Stop a monitoring session to save its summary."
    )


# =========================================================
# PDF REPORT DOWNLOAD
# =========================================================

st.subheader("📄 PDF Session Report")

has_session = (
    st.session_state.monitoring
    or st.session_state.last_session_id is not None
)

if not has_session:
    st.caption(
        "Start a monitoring session to generate its PDF report."
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

            st.success(
                "PDF report generated successfully."
            )

        except Exception as exc:
            st.error(
                f"PDF report generation failed: {exc}"
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

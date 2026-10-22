
from pathlib import Path
import glob
import queue
import threading
import time
import tempfile

import av
import cv2
import mediapipe as mp
import pandas as pd
import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoProcessorBase,
)

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
    @import url(
      'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Orbitron:wght@500;600;700;800&display=swap'
    );

    .stApp {
        background:
            radial-gradient(
                ellipse at 15% 0%,
                rgba(17,101,139,.19),
                transparent 35%
            ),
            radial-gradient(
                ellipse at 95% 15%,
                rgba(16,174,160,.10),
                transparent 30%
            ),
            linear-gradient(
                145deg,
                #070b14,
                #0a1220 55%,
                #071019
            );
        color: #e8f4fc;
        font-family: Inter, sans-serif;
    }

    .block-container {
        max-width: 1600px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #f1f8ff !important;
        letter-spacing: -.035em;
    }

    h1, .dg-title {
        font-family: Orbitron, sans-serif !important;
    }

    p, label, [data-testid="stCaptionContainer"] {
        color: #91a6ba;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #132236, #0d1726);
        border: 1px solid rgba(81,190,220,.2);
        border-radius: 16px;
        padding: 16px;
    }

    [data-testid="stMetricLabel"] {
        color: #9ab0c5 !important;
    }

    [data-testid="stMetricValue"] {
        color: #eafaff !important;
        font-family: Orbitron, sans-serif;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1422, #080e18);
        border-right: 1px solid rgba(81,190,220,.2);
    }

    div.stButton > button,
    div.stDownloadButton > button {
        min-height: 43px;
        border: 1px solid rgba(67,217,245,.35);
        border-radius: 11px;
        background: linear-gradient(115deg, #123d54, #12534f);
        color: #effcff;
        font-weight: 700;
        transition: filter .2s ease, border-color .2s ease;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        border-color: #43d9f5;
        filter: brightness(1.12);
    }

    .dg-hero {
        padding: 22px;
        margin: 4px 0 22px;
        border: 1px solid rgba(81,190,220,.2);
        border-radius: 18px;
        background: linear-gradient(
            110deg,
            rgba(17,39,59,.93),
            rgba(9,24,36,.72)
        );
        animation: dg-enter .5s ease-out;
    }

    .dg-kicker {
        color: #43d9f5;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: .22em;
    }

    .dg-title {
        color: #f0f9ff;
        font-size: 28px;
        font-weight: 800;
        margin-top: 5px;
    }

    .dg-copy {
        color: #91aabd;
        font-size: 13px;
        line-height: 1.7;
        margin-top: 8px;
    }

    .dg-status {
        border: 1px solid rgba(81,190,220,.24);
        background: rgba(15,35,52,.75);
        padding: 12px 15px;
        border-radius: 12px;
        color: #b9d2e4;
        font-size: 13px;
        line-height: 1.6;
        margin: 8px 0 12px;
    }

    div[data-testid="stWebRtc"] video {
        width: 100%;
        max-height: 620px;
        border-radius: 16px;
        object-fit: contain;
        background: #050a10;
    }

    @keyframes dg-enter {
        from {
            opacity: 0;
            transform: translateY(7px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: .01ms !important;
            transition-duration: .01ms !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LANDMARKS
# =========================================================

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
    if not landmarks:
        return frame

    height, width = frame.shape[:2]

    indices = (
        set(LEFT_EYE)
        | set(RIGHT_EYE)
        | set(OUTER_LIP_CONTOUR)
        | set(INNER_LIP_CONTOUR)
    )

    for index in indices:
        if not 0 <= index < len(landmarks):
            continue

        point = landmarks[index]
        x = int(point.x * width)
        y = int(point.y * height)

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


def put_latest(target_queue, item):
    """Keep the most recent frame without blocking the callback."""
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
# THREAD-SAFE SHARED FRAME STATE
# =========================================================

class SharedFrameState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_metrics = {}
        self.latest_risk_history = []
        self.last_error = None
        self.started_at = None
        self.event_count = 0
        self.eye_closure_count = 0
        self.eye_closure_total_seconds = 0.0
        self.running = False

    def update_metrics(self, metrics):
        with self.lock:
            self.latest_metrics = dict(metrics)

    def get_metrics(self):
        with self.lock:
            return dict(self.latest_metrics)

    def set_error(self, error):
        with self.lock:
            self.last_error = str(error)

    def get_error(self):
        with self.lock:
            return self.last_error


# =========================================================
# VIDEO PROCESSOR
# =========================================================

class DriverGuardVideoProcessor(VideoProcessorBase):
    """
    Process camera frames in the WebRTC worker thread.

    The processor owns its MediaPipe model and analyzers.
    It does not call Streamlit UI functions.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.shared = SharedFrameState()
        self.started_at = time.monotonic()
        self.previous_frame_time = None
        self.last_timestamp_ms = -1
        self.fps = 0.0
        self.closed = False

        self.risk_history = []
        self.last_risk_sample_time = 0.0
        self.previous_eye_closed = False
        self.last_eye_state_time = None

        self.eye_closure_count = 0
        self.eye_closure_total_seconds = 0.0

        self.landmarker = None
        self.eye_analyzer = None
        self.yawn_analyzer = None
        self.head_analyzer = None
        self.state_manager = None
        self.alert_manager = None
        self.logger = None

        try:
            if not Path(MODEL_PATH).is_file():
                raise FileNotFoundError(
                    f"Face Landmarker model not found: {MODEL_PATH}"
                )

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

            self.landmarker = (
                mp.tasks.vision.FaceLandmarker
                .create_from_options(options)
            )

            self.eye_analyzer = EyeAnalyzer(
                ear_threshold=EAR_THRESHOLD,
                closed_duration=CLOSED_DURATION,
                smoothing_frames=SMOOTHING_FRAMES,
            )

            self.yawn_analyzer = YawnAnalyzer(
                threshold=YAWN_THRESHOLD,
                duration=YAWN_DURATION,
                cooldown=YAWN_COOLDOWN,
            )

            self.head_analyzer = HeadPoseAnalyzer(
                tilt_threshold=HEAD_TILT_THRESHOLD,
                duration=HEAD_TILT_DURATION,
                cooldown=HEAD_TILT_COOLDOWN,
            )

            self.state_manager = DriverStateManager()
            self.alert_manager = AlertManager(ALERT_SOUND_PATH)
            self.logger = EventLogger(LOG_DIR)

            self.shared.started_at = self.started_at
            self.shared.running = True

        except Exception as exc:
            self.shared.set_error(
                f"Could not initialize video processor: {exc}"
            )
            self.close()

    def recv(self, video_frame):
        """
        Return a valid video frame on every invocation.

        If initialization or processing fails, return the original
        frame rather than a blank image.
        """
        original = video_frame

        try:
            frame = video_frame.to_ndarray(format="bgr24")

            with self.lock:
                if self.closed or self.landmarker is None:
                    return original

                processed = self.process_frame(frame)

            if processed is None:
                return original

            return av.VideoFrame.from_ndarray(
                processed,
                format="bgr24",
            )

        except Exception as exc:
            self.shared.set_error(
                f"Video processing error: {exc}"
            )
            return original

    def process_frame(self, frame):
        now = time.monotonic()
        elapsed = now - self.started_at

        if self.previous_frame_time is not None:
            delta = now - self.previous_frame_time
            if delta > 0:
                instant_fps = 1.0 / delta
                self.fps = (
                    instant_fps
                    if self.fps == 0
                    else 0.85 * self.fps + 0.15 * instant_fps
                )

        self.previous_frame_time = now

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        timestamp_ms = max(
            int(elapsed * 1000),
            self.last_timestamp_ms + 1,
        )
        self.last_timestamp_ms = timestamp_ms

        result = self.landmarker.detect_for_video(
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
            draw_eye_lip_landmarks(frame, landmarks)

            left_ear = eye_aspect_ratio(
                landmarks, LEFT_EYE, width, height
            )
            right_ear = eye_aspect_ratio(
                landmarks, RIGHT_EYE, width, height
            )

            eyes = self.eye_analyzer.update(
                left_ear, right_ear, now
            )

            mar = mouth_aspect_ratio(
                landmarks, width, height
            )
            mouth = self.yawn_analyzer.update(mar, now)

            pose = estimate_head_pose(
                landmarks, width, height
            )
            head = self.head_analyzer.update(pose, now)

        else:
            self.eye_analyzer.reset()
            self.yawn_analyzer.reset()
            self.head_analyzer.reset()

        eyes_closed = bool(
            face_present and eyes["eyes_closed"]
        )

        if eyes_closed and not self.previous_eye_closed:
            self.last_eye_state_time = now

        elif self.previous_eye_closed and not eyes_closed:
            if self.last_eye_state_time is not None:
                self.eye_closure_total_seconds += max(
                    0, now - self.last_eye_state_time
                )
                self.eye_closure_count += 1

            self.last_eye_state_time = None

        self.previous_eye_closed = eyes_closed

        signals = {
            "face_present": face_present,
            "eyes_closed": eyes["eyes_closed"],
            "drowsy": eyes["drowsy"],
            "yawning": mouth["yawning"],
            "head_tilt": head["head_tilt"],
        }

        state = self.state_manager.update(signals, now)
        risk = state["risk_score"]

        if (
            elapsed - self.last_risk_sample_time >= 0.5
            or not self.risk_history
        ):
            self.risk_history.append({
                "elapsed_seconds": round(elapsed, 2),
                "risk_score": float(risk),
                "status": state["status"],
            })
            self.last_risk_sample_time = elapsed

        self.risk_history = self.risk_history[-7200:]

        self.alert_manager.update(eyes["drowsy"])

        for event in state["events"]:
            self.logger.log_event(
                event, risk, elapsed, state["reasons"]
            )

        if mouth["yawn_event"]:
            self.logger.log_event(
                "YAWN_DETECTED",
                risk,
                elapsed,
                f"MAR={mouth['mar']:.3f}",
            )

        if head["tilt_event"]:
            self.logger.log_event(
                "PROLONGED_HEAD_DEVIATION",
                risk,
                elapsed,
                f"Yaw={head.get('yaw')}, "
                f"Pitch={head.get('pitch')}",
            )

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

        cv2.rectangle(
            frame, (0, 0), (width, 76), (10, 20, 32), -1
        )
        cv2.line(
            frame, (0, 76), (width, 76),
            (70, 210, 245), 2, cv2.LINE_AA
        )

        cv2.putText(
            frame, "DRIVERGUARD AI", (18, 29),
            cv2.FONT_HERSHEY_DUPLEX, 0.72,
            (245, 250, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            frame, "INTELLIGENT DRIVER MONITORING", (19, 54),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42,
            (160, 190, 210), 1, cv2.LINE_AA
        )

        badge_width = min(250, max(150, width - 36))
        badge_x = max(18, width - badge_width - 18)

        cv2.rectangle(
            frame, (badge_x, 17), (width - 18, 56),
            (25, 40, 54), -1
        )
        cv2.rectangle(
            frame, (badge_x, 17), (width - 18, 56),
            color, 1
        )
        cv2.putText(
            frame, banner, (badge_x + 9, 42),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40,
            color, 1, cv2.LINE_AA
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
            "fps": self.fps,
            "event_count": self.logger.event_count,
            "eye_closure_count": self.eye_closure_count,
            "eye_closure_total_seconds":
                self.eye_closure_total_seconds,
        }

        self.shared.update_metrics(metrics)
        self.shared.latest_risk_history = list(self.risk_history)
        self.shared.event_count = self.logger.event_count
        self.shared.eye_closure_count = self.eye_closure_count
        self.shared.eye_closure_total_seconds = (
            self.eye_closure_total_seconds
        )

        return frame

    def on_ended(self):
        self.close()

    def close(self):
        with self.lock:
            if self.closed:
                return

            self.closed = True

            if (
                self.previous_eye_closed
                and self.last_eye_state_time is not None
            ):
                self.eye_closure_total_seconds += max(
                    0,
                    time.monotonic() - self.last_eye_state_time,
                )
                self.eye_closure_count += 1
                self.previous_eye_closed = False
                self.last_eye_state_time = None

            if self.logger is not None:
                try:
                    duration = time.monotonic() - self.started_at
                    self.logger.finish(duration)
                except Exception as exc:
                    self.shared.set_error(
                        f"Session logging error: {exc}"
                    )

            for obj in (self.alert_manager, self.landmarker):
                if obj is not None:
                    try:
                        obj.close()
                    except Exception:
                        pass

            self.shared.running = False


def get_rtc_configuration():
    """STUN by default; add a TURN server from secrets when configured."""
    servers = [{"urls": ["stun:stun.l.google.com:19302"]}]

    try:
        turn = st.secrets.get("turn")
    except Exception:
        turn = None

    if turn:
        servers.append({
            "urls": list(turn["urls"]),
            "username": turn["username"],
            "credential": turn["credential"],
        })
    else:
        # Shared public relay (Open Relay Project): fine for demos,
        # rate-limited. Configure [turn] secrets for reliable use.
        servers.append({
            "urls": [
                "turn:openrelay.metered.ca:80",
                "turn:openrelay.metered.ca:443",
                "turn:openrelay.metered.ca:443?transport=tcp",
            ],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        })

    return {"iceServers": servers}


# =========================================================
# SESSION STATE
# =========================================================

DEFAULTS = {
    "monitoring": False,
    "last_session_id": None,
    "last_summary": None,
    "last_metrics": {},
    "risk_history": [],
    "pdf_report_bytes": None,
    "pdf_report_name": None,
    "dashboard_error": None,
    "processor_ref": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# SESSION DATA HELPERS
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
    session_id = st.session_state.last_session_id

    if not session_id:
        return pd.DataFrame()

    data = load_session_events()

    if data.empty:
        return data

    data = data[data["session"] == session_id].copy()

    for column in ("elapsed_seconds", "risk_score"):
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    return data.dropna(
        subset=["elapsed_seconds", "risk_score"]
    )


def render_history():
    history = st.session_state.risk_history

    if history:
        chart = pd.DataFrame(history)[
            ["elapsed_seconds", "risk_score"]
        ].dropna().sort_values("elapsed_seconds")

        chart = chart.set_index("elapsed_seconds")
        chart.index.name = "Elapsed time (seconds)"
        chart.columns = ["Driver Risk Score"]

        st.line_chart(chart, height=300)
        st.caption(
            "Risk trend · Samples collected approximately "
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
        ["elapsed_seconds", "risk_score"]
    ].sort_values("elapsed_seconds").set_index(
        "elapsed_seconds"
    )

    chart.index.name = "Elapsed time (seconds)"
    chart.columns = ["Recorded Event Risk"]
    st.line_chart(chart, height=300)


def render_recent_events():
    data = get_current_session_events()

    if data.empty:
        st.info("No events recorded for this session yet.")
        return

    recent = data.sort_values(
        "elapsed_seconds", ascending=False
    ).head(8)

    columns = [
        column for column in (
            "elapsed_seconds", "event", "risk_score", "details"
        )
        if column in recent.columns
    ]

    st.dataframe(
        recent[columns],
        use_container_width=True,
        hide_index=True,
    )

    session_id = st.session_state.last_session_id or "session"

    st.download_button(
        "⬇️ Download Session Events",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name=f"driverguard_{session_id}.csv",
        mime="text/csv",
        key="download_events",
    )


# =========================================================
# LIVE DASHBOARD
# =========================================================

def live_dashboard():
    camera_col, metrics_col = st.columns(
        [1.45, 1], gap="large"
    )

    with camera_col:
        st.subheader("📹 Live Camera Feed")
        ctx = webrtc_streamer(
            key="driverguard-browser-camera",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=get_rtc_configuration(),
            video_processor_factory=DriverGuardVideoProcessor,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 15},
                },
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
        )

        @st.fragment(run_every=0.5)
        def camera_status(ctx):
            processor = ctx.video_processor

            if processor is not None:
                st.session_state.processor_ref = processor
                st.session_state.last_session_id = processor.logger.session_id

                error = processor.shared.get_error()
                if error:
                    st.error(error)

                metrics = processor.shared.get_metrics()

                if metrics:
                    st.session_state.last_metrics = metrics
                    st.session_state.risk_history = list(
                        processor.risk_history
                    )

                if ctx.state.playing:
                    st.success("Camera connected · Monitoring active")
                    st.caption(
                        "Eye and lip landmarks · Eye-closure analytics "
                        "· Yawning detection · Head-pose tracking"
                    )
                else:
                    st.info(
                        "Camera is initialized. Allow camera permission "
                        "and wait for the video connection."
                    )
            else:
                st.info(
                    "Allow browser camera access to start live monitoring."
                )

        camera_status(ctx)


    @st.fragment(run_every=0.5)
    def metrics_panel():
        st.subheader("📊 Driver Status")
        metrics = st.session_state.last_metrics

        if not metrics:
            st.metric("Driver Risk Score", "—")
            st.caption(
                "Live measurements appear when the first frame "
                "has been processed."
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
                f"{metrics['ear']:.3f}"
                if metrics["ear"] is not None else "—",
            )
            b.metric(
                "Mouth Aspect Ratio",
                f"{metrics['mar']:.3f}"
                if metrics["mar"] is not None else "—",
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
                f"{metrics['yaw']:.1f}°"
                if metrics["yaw"] is not None else "—",
            )
            f.metric(
                "Head Pitch",
                f"{metrics['pitch']:.1f}°"
                if metrics["pitch"] is not None else "—",
            )

            st.metric(
                "Head Roll",
                f"{metrics['roll']:.1f}°"
                if metrics["roll"] is not None else "—",
            )

            if metrics["warning"] == "DROWSINESS ALERT":
                st.audio(
                    str(ALERT_SOUND_PATH),
                    format="audio/wav",
                    autoplay=True,
                    loop=True,
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


    with metrics_col:
        metrics_panel()


    @st.fragment(run_every=0.5)
    def history_panel():
        st.divider()

        chart_col, events_col = st.columns(
            [1.15, 1], gap="large"
        )

        with chart_col:
            st.subheader("📈 Risk Trend")
            render_history()

        with events_col:
            st.subheader("⚠️ Recent Events")
            render_recent_events()


    history_panel()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## 🛡️ DriverGuard AI")
    st.caption("Intelligent Driver Monitoring")
    st.divider()

    st.markdown("### Control Center")

    st.write(
        "The WebRTC processor starts when the camera connects. "
        "Use the camera's Stop control to end the stream."
    )

    if st.button(
        "↻ Refresh Dashboard",
        use_container_width=True,
    ):
        st.rerun()

    st.divider()
    st.markdown("### Detection Modules")

    for item in (
        "Eye landmark tracking",
        "Lip landmark tracking",
        "Eye closure / drowsiness",
        "Possible yawning",
        "Head-pose deviation",
        "Risk trend tracking",
        "CSV event logging",
        "PDF session reports",
    ):
        st.write(f"🟢 {item}")

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
        <div class="dg-title">DRIVERGUARD AI</div>
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

summary_path_text = st.session_state.last_summary

if summary_path_text:
    summary_path = Path(summary_path_text)

    if summary_path.exists():
        try:
            summary = pd.read_csv(summary_path)
            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True,
            )
        except Exception as exc:
            st.warning(f"Could not read session summary: {exc}")
    else:
        st.caption("Last session summary is unavailable.")
else:
    st.caption(
        "A session summary will appear when your existing "
        "event logger saves one."
    )


# =========================================================
# PDF REPORT
# =========================================================

st.subheader("📄 PDF Session Report")

processor = st.session_state.processor_ref

if processor is None:
    st.caption(
        "Connect the camera and start monitoring before generating "
        "a report."
    )
else:
    if st.button(
        "🧾 Generate PDF Report",
        type="primary",
        key="generate_pdf_report",
    ):
        try:
            session_id = processor.logger.session_id
            duration = time.monotonic() - processor.started_at

            events_path = (
                Path(LOG_DIR) / f"events_{session_id}.csv"
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

            history = list(processor.risk_history)
            values = [
                float(row["risk_score"])
                for row in history
                if row.get("risk_score") is not None
            ]

            summary = {
                "duration_seconds": duration,
                "average_risk": (
                    sum(values) / len(values) if values else 0
                ),
                "maximum_risk": max(values) if values else 0,
                "total_events": processor.logger.event_count,
            }

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
                        processor.eye_closure_total_seconds
                    ),
                    eye_closure_count=(
                        processor.eye_closure_count
                    ),
                )

                st.session_state.pdf_report_bytes = (
                    pdf_path.read_bytes()
                )
                st.session_state.pdf_report_name = (
                    f"DriverGuard_Report_{session_id}.pdf"
                )

            st.session_state.last_session_id = session_id
            st.success("PDF report generated successfully.")

        except Exception as exc:
            st.error(f"PDF report generation failed: {exc}")

    if st.session_state.pdf_report_bytes:
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
    "Risk scores are estimates, not validated measures of fitness "
    "to drive. Always follow safe-driving practices."
)

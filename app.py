
import time
import cv2
import mediapipe as mp

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
    CAMERA_INDEX,
    WINDOW_NAME,
)

from vision.eye_analyzer import EyeAnalyzer, LEFT_EYE, RIGHT_EYE, eye_aspect_ratio
from vision.mouth_analyzer import mouth_aspect_ratio, YawnAnalyzer
from vision.head_pose import estimate_head_pose, HeadPoseAnalyzer
from monitoring.alert_manager import AlertManager
from monitoring.state_manager import DriverStateManager
from monitoring.event_logger import EventLogger


def draw_text(frame, text, position, color=(235, 235, 235), scale=0.58):
    cv2.putText(
        frame,
        str(text),
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_risk_bar(frame, score, x=25, y=160, width=300, height=18):
    score = max(0, min(100, int(score)))
    cv2.rectangle(frame, (x, y), (x + width, y + height), (65, 65, 65), -1)

    if score <= 29:
        color = (60, 200, 60)
    elif score <= 59:
        color = (0, 190, 255)
    else:
        color = (0, 0, 255)

    fill_width = int(width * score / 100)
    if fill_width > 0:
        cv2.rectangle(
            frame,
            (x, y),
            (x + fill_width, y + height),
            color,
            -1,
        )

    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        (220, 220, 220),
        1,
    )


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Face Landmarker model not found: {MODEL_PATH}\n"
            "Check that models/face_landmarker.task exists."
        )

    eye_analyzer = EyeAnalyzer(
        ear_threshold=EAR_THRESHOLD,
        closed_duration=CLOSED_DURATION,
        smoothing_frames=SMOOTHING_FRAMES,
    )

    yawn_analyzer = YawnAnalyzer(
        threshold=YAWN_THRESHOLD,
        duration=YAWN_DURATION,
        cooldown=YAWN_COOLDOWN,
    )

    head_analyzer = HeadPoseAnalyzer(
        tilt_threshold=HEAD_TILT_THRESHOLD,
        duration=HEAD_TILT_DURATION,
        cooldown=HEAD_TILT_COOLDOWN,
    )

    state_manager = DriverStateManager()
    alert_manager = AlertManager(ALERT_SOUND_PATH)
    logger = EventLogger(LOG_DIR)

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        alert_manager.close()
        logger.finish(0)
        raise RuntimeError(
            "Could not open the webcam. Close other camera apps "
            "or check CAMERA_INDEX."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mp_vision = mp.tasks.vision

    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    start_time = time.monotonic()
    previous_frame_time = start_time
    fps = 0.0

    print("DriverGuard AI started. Press Q to quit.")

    try:
        with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = camera.read()

                if not success:
                    print("Unable to read a frame from the webcam.")
                    break

                now = time.monotonic()
                elapsed = now - start_time
                delta = now - previous_frame_time
                previous_frame_time = now

                if delta > 0:
                    current_fps = 1.0 / delta
                    fps = (
                        current_fps
                        if fps == 0
                        else 0.9 * fps + 0.1 * current_fps
                    )

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                timestamp_ms = int(elapsed * 1000)
                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                face_present = bool(result.face_landmarks)

                eye_data = {
                    "ear": None,
                    "eyes_closed": False,
                    "drowsy": False,
                    "closed_seconds": 0.0,
                }

                yawn_data = {
                    "mar": None,
                    "yawning": False,
                    "yawn_event": False,
                    "yawn_seconds": 0.0,
                }

                head_data = {
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

                    eye_data = eye_analyzer.update(
                        left_ear,
                        right_ear,
                        now,
                    )

                    mar = mouth_aspect_ratio(
                        landmarks,
                        width,
                        height,
                    )

                    yawn_data = yawn_analyzer.update(mar, now)

                    pose = estimate_head_pose(
                        landmarks,
                        width,
                        height,
                    )

                    head_data = head_analyzer.update(pose, now)

                    # Draw facial landmark points.
                    for index in LEFT_EYE + RIGHT_EYE:
                        point = landmarks[index]
                        px = int(point.x * width)
                        py = int(point.y * height)

                        cv2.circle(
                            frame,
                            (px, py),
                            2,
                            (0, 255, 255),
                            -1,
                        )

                    for index in (13, 14, 78, 308):
                        point = landmarks[index]
                        px = int(point.x * width)
                        py = int(point.y * height)

                        cv2.circle(
                            frame,
                            (px, py),
                            2,
                            (255, 180, 0),
                            -1,
                        )

                else:
                    eye_analyzer.reset()
                    yawn_analyzer.reset()
                    head_analyzer.reset()

                signals = {
                    "face_present": face_present,
                    "eyes_closed": eye_data["eyes_closed"],
                    "drowsy": eye_data["drowsy"],
                    "yawning": yawn_data["yawning"],
                    "head_tilt": head_data["head_tilt"],
                }

                state = state_manager.update(signals, now)
                risk_score = state["risk_score"]

                # Sound the alarm for detected drowsiness.
                alert_manager.update(eye_data["drowsy"])

                # Save each newly detected state transition.
                for event in state["events"]:
                    logger.log_event(
                        event_type=event,
                        risk_score=risk_score,
                        elapsed_seconds=elapsed,
                        details=state["reasons"],
                    )

                if yawn_data["yawn_event"]:
                    logger.log_event(
                        event_type="YAWN_DETECTED",
                        risk_score=risk_score,
                        elapsed_seconds=elapsed,
                        details=f"MAR={yawn_data['mar']:.3f}",
                    )

                if head_data["tilt_event"]:
                    logger.log_event(
                        event_type="PROLONGED_HEAD_DEVIATION",
                        risk_score=risk_score,
                        elapsed_seconds=elapsed,
                        details=(
                            f"Yaw={head_data['yaw']:.1f}, "
                            f"Pitch={head_data['pitch']:.1f}"
                        ),
                    )

                # Dashboard-style overlay.
                overlay = frame.copy()

                cv2.rectangle(
                    overlay,
                    (0, 0),
                    (365, height),
                    (20, 20, 20),
                    -1,
                )

                frame = cv2.addWeighted(
                    overlay,
                    0.82,
                    frame,
                    0.18,
                    0,
                )

                draw_text(
                    frame,
                    "DRIVERGUARD AI",
                    (25, 35),
                    (255, 220, 80),
                    0.8,
                )

                draw_text(
                    frame,
                    "DRIVER MONITORING SYSTEM",
                    (25, 62),
                    scale=0.43,
                )

                draw_text(
                    frame,
                    f"SESSION: {int(elapsed // 60):02d}:{int(elapsed % 60):02d}",
                    (25, 95),
                )

                draw_text(frame, f"FPS: {fps:.1f}", (25, 125))
                draw_text(frame, "DRIVER RISK SCORE", (25, 148))

                draw_risk_bar(frame, risk_score)

                draw_text(
                    frame,
                    f"{risk_score}/100",
                    (25, 202),
                    scale=0.65,
                )

                if state["status"] == "HIGH RISK":
                    status_color = (0, 0, 255)
                elif state["status"] == "MODERATE RISK":
                    status_color = (0, 190, 255)
                else:
                    status_color = (60, 220, 60)

                draw_text(
                    frame,
                    state["status"],
                    (25, 235),
                    status_color,
                    0.7,
                )

                ear_value = eye_data["ear"]
                mar_value = yawn_data["mar"]

                draw_text(
                    frame,
                    f"EAR: {ear_value:.3f}"
                    if ear_value is not None
                    else "EAR: --",
                    (25, 280),
                )

                draw_text(
                    frame,
                    f"Eyes closed: {eye_data['closed_seconds']:.1f}s",
                    (25, 310),
                )

                draw_text(
                    frame,
                    f"MAR: {mar_value:.3f}"
                    if mar_value is not None
                    else "MAR: --",
                    (25, 345),
                )

                yaw = head_data["yaw"]
                pitch = head_data["pitch"]
                roll = head_data["roll"]

                if yaw is not None:
                    draw_text(frame, f"Yaw: {yaw:.1f}", (25, 380))
                    draw_text(frame, f"Pitch: {pitch:.1f}", (25, 410))
                    draw_text(frame, f"Roll: {roll:.1f}", (25, 440))
                else:
                    draw_text(frame, "Head pose: --", (25, 380))

                draw_text(
                    frame,
                    f"Logged events: {logger.event_count}",
                    (25, 485),
                )

                if not face_present:
                    warning = "FACE NOT DETECTED"
                    warning_color = (0, 165, 255)
                elif eye_data["drowsy"]:
                    warning = "DROWSINESS ALERT!"
                    warning_color = (0, 0, 255)
                elif eye_data["eyes_closed"]:
                    warning = "EYES CLOSED"
                    warning_color = (0, 165, 255)
                elif yawn_data["yawning"]:
                    warning = "POSSIBLE YAWN"
                    warning_color = (0, 165, 255)
                elif head_data["head_tilt"]:
                    warning = "HEAD DEVIATION"
                    warning_color = (0, 165, 255)
                else:
                    warning = "MONITORING ACTIVE"
                    warning_color = (60, 220, 60)

                draw_text(
                    frame,
                    warning,
                    (25, 535),
                    warning_color,
                    0.65,
                )

                draw_text(
                    frame,
                    "Press Q to quit",
                    (25, height - 25),
                    scale=0.5,
                )

                cv2.imshow(WINDOW_NAME, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("Stopping DriverGuard AI...")

    finally:
        alert_manager.close()
        camera.release()
        cv2.destroyAllWindows()

        duration = time.monotonic() - start_time
        logger.finish(duration)

        print(f"Session ended. Duration: {duration:.1f} seconds.")
        print(f"Events logged: {logger.event_count}")


if __name__ == "__main__":
    main()

import cv2
import numpy as np


# Approximate 3D facial reference points.
# These are a simplified face model, not a
# personalized 3D reconstruction.

MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye outer
        (225.0, 170.0, -135.0),   # Right eye outer
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0),  # Right mouth corner
    ],
    dtype=np.float64,
)

LANDMARK_INDICES = (
    1,    # Nose tip
    152,  # Chin
    33,   # Left eye outer
    263,  # Right eye outer
    61,   # Left mouth corner
    291,  # Right mouth corner
)


def estimate_head_pose(landmarks, width, height):
    """
    Estimate head pitch, yaw and roll.

    Returns a dictionary of approximate angles in
    degrees, or None if estimation fails.
    """

    if len(landmarks) <= max(LANDMARK_INDICES):
        return None

    image_points = np.array(
        [
            (
                landmarks[index].x * width,
                landmarks[index].y * height,
            )
            for index in LANDMARK_INDICES
        ],
        dtype=np.float64,
    )

    focal_length = float(width)

    camera_matrix = np.array(
        [
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    distortion = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, _ = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(
        rotation_vector
    )

    angles = cv2.RQDecomp3x3(
        rotation_matrix
    )[0]

    pitch, yaw, roll = (float(value) for value in angles)
    
    def normalize_angle(angle):
        angle = (angle + 180.0) % 360.0 - 180.0
        if angle > 90.0:
            angle -= 180.0
        elif angle < -90.0:
            angle += 180.0
            
        return angle
    pitch = normalize_angle(pitch)
    yaw = normalize_angle(yaw)
    roll = normalize_angle(roll)

    return {
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
    }


class HeadPoseAnalyzer:
    """Detect sustained head-direction deviations."""

    def __init__(
        self,
        tilt_threshold=28.0,
        duration=1.5,
        cooldown=3.0,
    ):
        self.tilt_threshold = tilt_threshold
        self.duration = duration
        self.cooldown = cooldown

        self.tilt_since = None
        self.last_event = float("-inf")

    def update(self, pose, now):

        if pose is None:
            self.reset()

            return {
                "yaw": None,
                "pitch": None,
                "roll": None,
                "head_tilt": False,
                "tilt_seconds": 0.0,
                "tilt_event": False,
            }

        yaw = pose["yaw"]
        pitch = pose["pitch"]
        roll = pose["roll"]

        deviation = (
            abs(yaw) >= self.tilt_threshold
            or abs(pitch) >= self.tilt_threshold
        )

        if deviation:
            if self.tilt_since is None:
                self.tilt_since = now
        else:
            self.tilt_since = None

        seconds = (
            now - self.tilt_since
            if self.tilt_since is not None
            else 0.0
        )

        head_tilt = (
            deviation
            and seconds >= self.duration
        )

        event = (
            head_tilt
            and now - self.last_event >= self.cooldown
        )

        if event:
            self.last_event = now

        return {
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "head_tilt": head_tilt,
            "tilt_seconds": seconds,
            "tilt_event": event,
        }

    def reset(self):
        self.tilt_since = None
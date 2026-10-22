import math

#MediaPipe Face Landmarker indices for the eye contours
#Six representative points are selected for each eye
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def distance(point_a, point_b):
    """Euclidean distance between two 2D points"""
    return math.hypot(
        point_a[0] - point_b[0],
        point_a[1] - point_b[1],
    )

def eye_aspect_ratio(landmarks, eye_indices, width, height):
    """Compute the EARE-style measurment from normalized
    MediaPipe Landmarks
    
    Returns None if landmarks are unavailable or invalid
    """

    if len(landmarks) <= max(eye_indices):
        return None

    points = []

    for index in eye_indices:
        landmark = landmarks[index]
        points.append(
            (landmark.x * width, landmark.y * height)
        )

    p1, p2, p3, p4, p5, p6 = points

    horizontal = distance(p1, p4)

    if horizontal < 1e-6:
        return None

    vertical_one = distance(p2, p6)
    vertical_two = distance(p3, p5)

    return (vertical_one + vertical_two) / (2.0 * horizontal)

class EyeAnalyzer:
    def __init__(
        self,
        ear_threshold=0.21,
        closed_duration=1.5,
        smoothing_frames=3,
    ):
        self.ear_threshold = ear_threshold
        self.closed_duration = closed_duration
        self.smoothing_frames = smoothing_frames

        self.closed_since = None
        self.closed_frames = 0
        self.ear_history = []

    def update(self, left_ear, right_ear, now):
        """
        Update eye state using the current frame.

        now must be a monotonic timestamp in seconds.
        """
        if left_ear is None or right_ear is None:
            self.reset()
            return {
                "ear": None,
                "eyes_closed": False,
                "drowsy": False,
                "closed_seconds": 0.0,
            }

        ear = (left_ear + right_ear) / 2.0

        self.ear_history.append(ear)
        self.ear_history = self.ear_history[
            -self.smoothing_frames:
        ]

        smoothed_ear = sum(self.ear_history) / len(self.ear_history)

        eyes_closed = smoothed_ear < self.ear_threshold

        if eyes_closed:
            if self.closed_since is None:
                self.closed_since = now

            self.closed_frames += 1
        else:
            self.closed_since = None
            self.closed_frames = 0

        closed_seconds = (
            now - self.closed_since
            if self.closed_since is not None
            else 0.0
        )

        drowsy = (
            eyes_closed
            and self.closed_frames >= 3
            and closed_seconds >= self.closed_duration
        )

        return {
            "ear": smoothed_ear,
            "eyes_closed": eyes_closed,
            "drowsy": drowsy,
            "closed_seconds": closed_seconds,
        }

    def reset(self):
        self.closed_since = None
        self.closed_frames = 0
        self.ear_history.clear()

        

     
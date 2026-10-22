import math

# MediaPipe Face Landmarker mouth landmark indices
MOUTH_TOP = 13
MOUTH_BOTTOM = 14

MOUTH_LEFT = 78
MOUTH_RIGHT = 308

MOUTH_UPPER_SIDE = 82
MOUTH_LOWER_SIDE = 312


def distance(point_a, point_b):
    """Calculate Euclidean distance between two points."""
    return math.hypot(
        point_a[0] - point_b[0],
        point_a[1] - point_b[1],
    )


def mouth_aspect_ratio(landmarks, width, height):
    """
    Calculate a mouth-opening ratio from facial landmarks.

    Returns None if the landmarks are unavailable.
    """

    required = [
        MOUTH_TOP,
        MOUTH_BOTTOM,
        MOUTH_LEFT,
        MOUTH_RIGHT,
        MOUTH_UPPER_SIDE,
        MOUTH_LOWER_SIDE,
    ]

    if len(landmarks) <= max(required):
        return None

    points = {}

    for index in required:
        landmark = landmarks[index]

        points[index] = (
            landmark.x * width,
            landmark.y * height,
        )

    horizontal = distance(
        points[MOUTH_LEFT],
        points[MOUTH_RIGHT],
    )

    if horizontal < 1e-6:
        return None

    vertical_one = distance(
        points[MOUTH_TOP],
        points[MOUTH_BOTTOM],
    )

    vertical_two = distance(
        points[MOUTH_UPPER_SIDE],
        points[MOUTH_LOWER_SIDE],
    )

    return (
        vertical_one + vertical_two
    ) / (2.0 * horizontal)


class YawnAnalyzer:
    """Detect sustained mouth opening with a cooldown."""

    def __init__(
        self,
        threshold=0.55,
        duration=1.0,
        cooldown=2.0,
    ):
        self.threshold = threshold
        self.duration = duration
        self.cooldown = cooldown

        self.open_since = None
        self.last_event = float("-inf")

    def update(self, mar, now):
        """
        Update the yawn detector.

        mar: mouth aspect ratio
        now: monotonic timestamp in seconds
        """

        if mar is None:
            self.reset()

            return {
                "mar": None,
                "yawning": False,
                "yawn_seconds": 0.0,
                "yawn_event": False,
            }

        mouth_open = mar >= self.threshold

        if mouth_open:
            if self.open_since is None:
                self.open_since = now
        else:
            self.open_since = None

        open_seconds = (
            now - self.open_since
            if self.open_since is not None
            else 0.0
        )

        yawning = (
            mouth_open
            and open_seconds >= self.duration
        )

        event = (
            yawning
            and now - self.last_event >= self.cooldown
        )

        if event:
            self.last_event = now

        return {
            "mar": mar,
            "yawning": yawning,
            "yawn_seconds": open_seconds,
            "yawn_event": event,
        }

    def reset(self):
        self.open_since = None
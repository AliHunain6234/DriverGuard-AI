class DriverStateManager:
    """
    Explainable, experimental driver risk scoring.

    This score is a heuristic indicator, not a validated
    measurement of driver fitness or road safety.
    """

    def __init__(self):
        self.previous_flags = {}

    def update(self, signals, now=None):
        risk = 0
        reasons = []

        # ------------------------------------------
        # EYE CLOSURE AND DROWSINESS
        # ------------------------------------------

        if signals.get("drowsy", False):
            risk += 60
            reasons.append("Prolonged eye closure")

        elif signals.get("eyes_closed", False):
            risk += 20
            reasons.append("Eyes closed")

        # ------------------------------------------
        # YAWNING
        # ------------------------------------------

        if signals.get("yawning", False):
            risk += 20
            reasons.append("Possible prolonged yawn")

        # ------------------------------------------
        # HEAD POSITION
        # ------------------------------------------

        if signals.get("head_tilt", False):
            risk += 20
            reasons.append("Prolonged head deviation")

        # ------------------------------------------
        # FACE DETECTION
        # ------------------------------------------

        if not signals.get("face_present", True):
            risk += 10
            reasons.append("Face not detected")

        risk = min(100, risk)

        # ------------------------------------------
        # DRIVER STATUS
        # ------------------------------------------

        if risk <= 29:
            status = "LOW RISK"

        elif risk <= 59:
            status = "MODERATE RISK"

        else:
            status = "HIGH RISK"

        # ------------------------------------------
        # RECORD WARNING TRANSITIONS
        # ------------------------------------------

        flags = {
            "DROWSINESS_WARNING": signals.get(
                "drowsy", False
            ),
            "EYES_CLOSED_WARNING": signals.get(
                "eyes_closed", False
            ),
            "YAWNING_WARNING": signals.get(
                "yawning", False
            ),
            "HEAD_TILT_WARNING": signals.get(
                "head_tilt", False
            ),
            "FACE_NOT_DETECTED": not signals.get(
                "face_present", True
            ),
        }

        events = []

        for event_name, active in flags.items():

            was_active = self.previous_flags.get(
                event_name, False
            )

            if active and not was_active:
                events.append(event_name)

            self.previous_flags[event_name] = active

        return {
            "risk_score": risk,
            "status": status,
            "reasons": (
                ", ".join(reasons)
                if reasons
                else "No active warning"
            ),
            "events": events,
        }
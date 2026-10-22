
import csv
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from config import LOG_DIR


class EventLogger:
    """Log monitoring events and maintain session-compatible CSV files."""

    def __init__(self, log_dir=None):
        self.log_dir = Path(log_dir) if log_dir else Path(LOG_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Unique ID for the current monitoring session.
        self.session_id = (
            datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            + "_"
            + uuid4().hex[:6]
        )

        # Files used by the existing logging system.
        self.events_file = self.log_dir / "events.csv"
        self.risk_file = self.log_dir / "risk_history.csv"
        self.sessions_file = self.log_dir / "sessions.csv"

        # Session-specific file used by dashboard history.
        self.session_events_file = (
            self.log_dir / f"events_{self.session_id}.csv"
        )

        # Session summary file displayed by dashboard.py.
        self.summary_path = (
            self.log_dir / f"summary_{self.session_id}.csv"
        )

        self.event_count = 0
        self.finished = False
        self.start_time = datetime.now().astimezone()
        self.risk_scores = []

        self._ensure_file(
            self.events_file,
            [
                "timestamp",
                "driver_id",
                "event_type",
                "message",
                "risk_score",
            ],
        )

        self._ensure_file(
            self.risk_file,
            [
                "timestamp",
                "driver_id",
                "risk_score",
                "risk_level",
                "drowsy",
                "eyes_closed",
                "yawning",
                "head_tilt",
                "face_detected",
            ],
        )

        self._ensure_file(
            self.sessions_file,
            [
                "session_id",
                "driver_id",
                "start_time",
                "end_time",
                "duration_seconds",
                "average_risk",
                "maximum_risk",
                "total_events",
                "status",
            ],
        )

        self._ensure_file(
            self.session_events_file,
            [
                "elapsed_seconds",
                "risk_score",
                "event",
                "details",
            ],
        )

    @staticmethod
    def _ensure_file(file_path, headers):
        if not file_path.exists() or file_path.stat().st_size == 0:
            with file_path.open(
                "w", newline="", encoding="utf-8"
            ) as file:
                csv.writer(file).writerow(headers)

    @staticmethod
    def _timestamp(value=None):
        if value is None:
            return datetime.now().astimezone().isoformat(
                timespec="seconds"
            )

        if isinstance(value, datetime):
            return value.isoformat(timespec="seconds")

        return str(value)

    @staticmethod
    def _append_row(file_path, row):
        with file_path.open(
            "a", newline="", encoding="utf-8"
        ) as file:
            csv.writer(file).writerow(row)

    def log_event(
        self,
        event_type,
        risk_score=None,
        elapsed_seconds=None,
        details="",
        driver_id="unknown",
        timestamp=None,
        **kwargs,
    ):
        """Record events in both global and session-specific CSV files."""

        if self.finished:
            return

        # Support alternative keyword names.
        if "event_name" in kwargs:
            event_type = kwargs["event_name"]

        if "message" in kwargs and not details:
            details = kwargs["message"]

        if "elapsed" in kwargs and elapsed_seconds is None:
            elapsed_seconds = kwargs["elapsed"]

        if "details" in kwargs:
            details = kwargs["details"]

        # Convert numeric values safely.
        try:
            score = (
                float(risk_score)
                if risk_score is not None
                else 0.0
            )
        except (TypeError, ValueError):
            score = 0.0

        try:
            elapsed = (
                float(elapsed_seconds)
                if elapsed_seconds is not None
                else 0.0
            )
        except (TypeError, ValueError):
            elapsed = 0.0

        event_name = str(event_type)
        detail_text = str(details)
        timestamp_text = self._timestamp(timestamp)

        # Preserve the existing global events.csv format.
        self._append_row(
            self.events_file,
            [
                timestamp_text,
                driver_id,
                event_name,
                detail_text,
                score,
            ],
        )

        # Write the format expected by dashboard.py.
        self._append_row(
            self.session_events_file,
            [
                round(elapsed, 2),
                score,
                event_name,
                detail_text,
            ],
        )

        self.event_count += 1
        self.risk_scores.append(score)

    def log_risk(
        self,
        driver_id="unknown",
        risk_score=0,
        risk_level="Low",
        drowsy=False,
        eyes_closed=False,
        yawning=False,
        head_tilt=False,
        face_detected=True,
        timestamp=None,
    ):
        """Record a risk sample in the existing risk-history CSV."""

        self._append_row(
            self.risk_file,
            [
                self._timestamp(timestamp),
                driver_id,
                risk_score,
                risk_level,
                bool(drowsy),
                bool(eyes_closed),
                bool(yawning),
                bool(head_tilt),
                bool(face_detected),
            ],
        )

        try:
            self.risk_scores.append(float(risk_score))
        except (TypeError, ValueError):
            pass

    def log_session_summary(
        self,
        session_id,
        driver_id="unknown",
        start_time=None,
        end_time=None,
        duration_seconds=0,
        average_risk=0,
        maximum_risk=0,
        total_events=0,
        status="completed",
    ):
        """Append a completed session to sessions.csv."""

        row = [
            session_id,
            driver_id,
            self._timestamp(start_time),
            self._timestamp(end_time),
            round(float(duration_seconds), 2),
            round(float(average_risk), 2),
            round(float(maximum_risk), 2),
            int(total_events),
            status,
        ]

        self._append_row(self.sessions_file, row)

    def log_session(self, *args, **kwargs):
        """Compatibility alias for existing callers."""
        return self.log_session_summary(*args, **kwargs)

    def finish(self, duration_seconds=0):
        """Finish the session and write a summary CSV."""

        if self.finished:
            return

        end_time = datetime.now().astimezone()

        try:
            duration = max(float(duration_seconds), 0.0)
        except (TypeError, ValueError):
            duration = 0.0

        scores = self.risk_scores

        average_risk = (
            sum(scores) / len(scores) if scores else 0.0
        )
        maximum_risk = max(scores) if scores else 0.0

        summary = {
            "session_id": [self.session_id],
            "start_time": [self._timestamp(self.start_time)],
            "end_time": [self._timestamp(end_time)],
            "duration_seconds": [round(duration, 2)],
            "average_risk": [round(average_risk, 2)],
            "maximum_risk": [round(maximum_risk, 2)],
            "total_events": [self.event_count],
            "status": ["completed"],
        }

        pd.DataFrame(summary).to_csv(
            self.summary_path,
            index=False,
        )

        self.log_session_summary(
            session_id=self.session_id,
            start_time=self.start_time,
            end_time=end_time,
            duration_seconds=duration,
            average_risk=average_risk,
            maximum_risk=maximum_risk,
            total_events=self.event_count,
            status="completed",
        )

        self.finished = True

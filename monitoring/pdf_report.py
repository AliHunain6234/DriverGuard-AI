
import csv
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)


def _read_csv(file_path):
    """Read a CSV file safely and return its rows as dictionaries."""
    if not file_path:
        return []

    path = Path(file_path)

    if not path.exists() or not path.is_file():
        return []

    try:
        with path.open("r", newline="", encoding="utf-8-sig") as file:
            return list(csv.DictReader(file))
    except (OSError, csv.Error, UnicodeError):
        return []


def _number(value, default=0.0):
    """Convert a value to a number without crashing the report."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_duration(seconds):
    seconds = max(0, int(_number(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m {seconds}s"

    return f"{minutes}m {seconds}s"


def generate_session_report(
    output_path,
    session_summary=None,
    risk_history=None,
    events=None,
    driver_name="Driver",
    eye_closure_seconds=0,
    eye_closure_count=0,
):
    """
    Generate a PDF session report.

    Parameters:
        output_path: Destination PDF path.
        session_summary: Dictionary of session metrics or a CSV file path.
        risk_history: List of dictionaries or a CSV file path.
        events: List of event dictionaries or a CSV file path.
        driver_name: Driver's display name.
        eye_closure_seconds: Total eye-closure duration in seconds.
        eye_closure_count: Number of eye-closure events.
    """

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Load session summary if a CSV path was supplied.
    if isinstance(session_summary, (str, Path)):
        summary_rows = _read_csv(session_summary)
        session_summary = summary_rows[-1] if summary_rows else {}

    if not isinstance(session_summary, dict):
        session_summary = {}

    # Load risk history and events if file paths were supplied.
    if isinstance(risk_history, (str, Path)):
        risk_history = _read_csv(risk_history)

    if isinstance(events, (str, Path)):
        events = _read_csv(events)

    risk_history = risk_history or []
    events = events or []

    # Support a single summary row passed as a list.
    if isinstance(session_summary, list):
        session_summary = (
            session_summary[-1] if session_summary else {}
        )

    average_risk = _number(
        session_summary.get("average_risk", 0)
    )
    maximum_risk = _number(
        session_summary.get("maximum_risk", 0)
    )
    duration = _number(
        session_summary.get("duration_seconds", 0)
    )

    total_events = int(
        _number(
            session_summary.get(
                "total_events",
                len(events),
            )
        )
    )

    if risk_history:
        risk_values = [
            _number(row.get("risk_score", 0))
            for row in risk_history
        ]

        if not session_summary.get("average_risk"):
            average_risk = sum(risk_values) / len(risk_values)

        if not session_summary.get("maximum_risk"):
            maximum_risk = max(risk_values)

    generated_at = datetime.now().strftime("%d %B %Y, %I:%M %p")

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=23,
        leading=29,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#12304A"),
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=20,
    )

    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#12304A"),
        spaceBefore=12,
        spaceAfter=8,
    )

    normal_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
    )

    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="DriverGuard AI Session Report",
        author="DriverGuard AI",
    )

    story = []

    story.append(Paragraph("DriverGuard AI", title_style))
    story.append(
        Paragraph("DRIVER MONITORING SESSION REPORT", subtitle_style)
    )

    story.append(
        Paragraph(
            f"<b>Driver:</b> {driver_name} &nbsp;&nbsp; "
            f"<b>Generated:</b> {generated_at}",
            normal_style,
        )
    )

    story.append(Spacer(1, 14))
    story.append(Paragraph("Session Overview", section_style))

    overview_data = [
        ["Metric", "Result"],
        [
            "Session Duration",
            _format_duration(duration),
        ],
        [
            "Average Risk Score",
            f"{average_risk:.1f} / 100",
        ],
        [
            "Maximum Risk Score",
            f"{maximum_risk:.1f} / 100",
        ],
        [
            "Total Logged Events",
            str(total_events),
        ],
        [
            "Eye Closure Events",
            str(int(_number(eye_closure_count))),
        ],
        [
            "Total Eye Closure Duration",
            _format_duration(eye_closure_seconds),
        ],
    ]

    overview_table = Table(
        overview_data,
        colWidths=[3.1 * inch, 3.0 * inch],
        repeatRows=1,
    )

    overview_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#12304A"),
                ),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D8E1EA")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                    colors.white,
                    colors.HexColor("#F3F7FA"),
                ]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )

    story.append(overview_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Risk Trend Samples", section_style))

    if risk_history:
        trend_data = [["Elapsed Time (s)", "Risk Score", "Status"]]

        # Limit the table size to keep the PDF readable.
        selected_rows = risk_history[-30:]

        for row in selected_rows:
            elapsed = _number(
                row.get("elapsed_seconds", row.get("elapsed", 0))
            )
            score = _number(row.get("risk_score", 0))
            status = str(row.get("status", row.get("risk_level", "N/A")))

            trend_data.append([
                f"{elapsed:.1f}",
                f"{score:.1f} / 100",
                status[:40],
            ])

        trend_table = Table(
            trend_data,
            colWidths=[1.7 * inch, 1.7 * inch, 2.7 * inch],
            repeatRows=1,
        )

        trend_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#176B87")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D8E1EA")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                        colors.white,
                        colors.HexColor("#F3F7FA"),
                    ]),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        story.append(trend_table)
        story.append(
            Paragraph(
                "The table shows up to the latest 30 available risk samples.",
                styles["Italic"],
            )
        )
    else:
        story.append(
            Paragraph(
                "No risk-history samples were available for this session.",
                normal_style,
            )
        )

    story.append(Spacer(1, 12))
    story.append(Paragraph("Session Events", section_style))

    if events:
        event_data = [["Event", "Risk Score", "Details"]]

        for event in events[-25:]:
            event_name = str(
                event.get("event", event.get("event_type", "Event"))
            )
            score = event.get("risk_score", "N/A")
            details = str(
                event.get("details", event.get("message", ""))
            )

            event_data.append([
                event_name[:45],
                str(score)[:20],
                details[:90],
            ])

        event_table = Table(
            event_data,
            colWidths=[1.65 * inch, 1.05 * inch, 3.4 * inch],
            repeatRows=1,
        )

        event_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12304A")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D8E1EA")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        story.append(event_table)
    else:
        story.append(
            Paragraph(
                "No session events were recorded.",
                normal_style,
            )
        )

    story.append(Spacer(1, 18))
    story.append(
        Paragraph(
            "<b>Disclaimer:</b> This report is intended to support "
            "driver-awareness monitoring. It is not a medical diagnosis "
            "or a substitute for safe driving practices.",
            normal_style,
        )
    )

    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(
            0.55 * inch,
            0.35 * inch,
            "DriverGuard AI | Confidential Session Report",
        )
        canvas.drawRightString(
            A4[0] - 0.55 * inch,
            0.35 * inch,
            f"Page {doc.page}",
        )
        canvas.restoreState()

    document.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )

    return str(output)

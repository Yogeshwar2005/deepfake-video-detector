from pathlib import Path

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


def generate_report(
    result_id,
    filename,
    prediction,
    probability,
    frames_analyzed,
    faces_detected,
    threshold
):
    output_dir = Path("media/reports")

    pdf_path = output_dir / f"{result_id}.pdf"

    doc = SimpleDocTemplate(str(pdf_path))
    styles = getSampleStyleSheet()

    elements = []

    elements.append(
        Paragraph(
            "Deepfake Detection Report",
            styles["Title"]
        )
    )

    elements.append(Spacer(1, 20))

    data = [
        ["File", filename],
        ["Prediction", prediction],
        ["Fake Probability", f"{probability:.2f}%"],
        ["Frames Analyzed", frames_analyzed],
        ["Faces Detected", faces_detected],
        ["Threshold", f"{threshold:.2f}%"]
    ]

    table = Table(data)

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), colors.whitesmoke),
            ("BOX", (0,0), (-1,-1), 1, colors.black),
            ("GRID", (0,0), (-1,-1), 1, colors.black),
            ("PADDING", (0,0), (-1,-1), 8),
        ])
    )

    elements.append(table)

    doc.build(elements)

    return f"reports/{result_id}.pdf"
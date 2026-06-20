from pathlib import Path
from django.utils import timezone
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
)
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER


def generate_report(
    result_id,
    filename,
    prediction,
    probability,
    frames_analyzed,
    faces_detected,
    threshold,
    gradcam_results=None,
):
    gradcam_results = gradcam_results or []

    output_dir = Path("media/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / f"{result_id}.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path),
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=26,
        alignment=TA_CENTER,
        textColor=HexColor("#222222"),
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=9,
        alignment=TA_CENTER,
        textColor=HexColor("#666666"),
        spaceAfter=22,
    )

    section_style = ParagraphStyle(
        "SectionTitle",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=HexColor("#222222"),
        spaceBefore=18,
        spaceAfter=10,
    )

    verdict_color = HexColor("#b02a37") if prediction == "Fake" else HexColor("#146c43")

    verdict_style = ParagraphStyle(
        "Verdict",
        parent=styles["Heading1"],
        fontSize=20,
        alignment=TA_CENTER,
        textColor=verdict_color,
        spaceAfter=18,
    )

    elements = []

    elements.append(Paragraph("Deepfake Detection Report", title_style))
    elements.append(
        Paragraph(
            timezone.localtime(timezone.now()).strftime("Generated on %d %B %Y at %I:%M %p"),
            subtitle_style,
        )
    )

    elements.append(Paragraph(f"Result: {prediction}", verdict_style))

    probability_text = "N/A" if probability is None else f"{probability:.2f}%"

    summary_data = [
        ["File name", filename],
        ["Prediction", prediction],
        ["Fake probability", probability_text],
        ["Decision threshold", f"{threshold:.2f}%"],
        ["Frames analyzed", str(frames_analyzed)],
        ["Faces detected", str(faces_detected)],
    ]

    summary_table = Table(summary_data, colWidths=[150, 340])

    summary_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), HexColor("#f5f5f5")),
            ("TEXTCOLOR", (0, 0), (-1, -1), HexColor("#222222")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#dddddd")),
            ("BOX", (0, 0), (-1, -1), 0.8, HexColor("#cccccc")),
            ("PADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )

    elements.append(summary_table)

    elements.append(Paragraph("Model Details", section_style))

    model_data = [
        ["Model", "EfficientNet-B0"],
        ["Face detector", "MTCNN"],
        ["Input size", "224 × 224"],
        ["Frames sampled", "16"],
        ["Aggregation", "Top-5 mean logits"],
        ["Explanation method", "Grad-CAM"],
    ]

    model_table = Table(model_data, colWidths=[150, 340])

    model_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), HexColor("#f5f5f5")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#dddddd")),
            ("BOX", (0, 0), (-1, -1), 0.8, HexColor("#cccccc")),
            ("PADDING", (0, 0), (-1, -1), 8),
        ])
    )

    elements.append(model_table)

    if gradcam_results:
        elements.append(Paragraph("Suspicious Frames", section_style))

        for index, result in enumerate(gradcam_results, start=1):
            original_path = Path("media") / result["original_path"]
            gradcam_path = Path("media") / result["gradcam_path"]

            if original_path.exists() and gradcam_path.exists():
                elements.append(Paragraph(f"Frame Rank {index}", styles["Heading3"]))

                original_img = Image(str(original_path), width=155, height=155)
                gradcam_img = Image(str(gradcam_path), width=155, height=155)

                image_table = Table(
                    [
                        ["Original Face", "Grad-CAM"],
                        [original_img, gradcam_img],
                    ],
                    colWidths=[245, 245],
                )

                image_table.setStyle(
                    TableStyle([
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 10),
                        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#f5f5f5")),
                        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#dddddd")),
                        ("BOX", (0, 0), (-1, -1), 0.8, HexColor("#cccccc")),
                        ("PADDING", (0, 0), (-1, -1), 8),
                    ])
                )

                elements.append(image_table)
                elements.append(Spacer(1, 14))

    footer_style = ParagraphStyle(
        "Footer",
        parent=styles["Normal"],
        fontSize=8,
        alignment=TA_CENTER,
        textColor=HexColor("#777777"),
        spaceBefore=20,
    )

    elements.append(
        Paragraph(
            "This report was generated automatically from the submitted video analysis.",
            footer_style,
        )
    )

    doc.build(elements)

    return f"reports/{result_id}.pdf"
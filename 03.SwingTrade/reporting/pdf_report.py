from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import os


def generate_pdf_report(output_path="reports/final_report.pdf"):
    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(output_path, pagesize=A4)
    elements = []

    elements.append(Paragraph("Swing Trading Strategy Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    images = [
        "reports/portfolio_vs_nifty.png",
        "reports/drawdown.png",
        "reports/rolling_returns.png"
    ]

    for img in images:
        if os.path.exists(img):
            elements.append(Image(img, width=500, height=280))
            elements.append(Spacer(1, 20))

    doc.build(elements)

    print(f"PDF report generated: {output_path}")

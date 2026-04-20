import base64, io
from datetime import datetime

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors


def create_pdf_report(
    dpcm2=None,
    resemblance=None,
    utility=None,
    xai=None,
    datasets=None,
    models=None,
    assignments=None
):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        title="Synthetic Health Data Evaluation Report",
        pagesize=(8.5 * inch, 11 * inch)
    )

    styles = getSampleStyleSheet()
    elements = []

    # --------------------------------------------------
    # Title
    # --------------------------------------------------

    elements.append(Paragraph(
        "Synthetic Health Data Evaluation Report",
        styles["Title"]
    ))

    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph(
        f"Generated on: {datetime.now().strftime('%B %d, %Y')}",
        styles["Normal"]
    ))

    elements.append(Spacer(1, 0.4 * inch))

    # --------------------------------------------------
    # Dataset Information
    # --------------------------------------------------

    if datasets:

        elements.append(Paragraph("Datasets Used in Evaluation", styles["Heading2"]))
        elements.append(Spacer(1, 0.2 * inch))

        table_data = [["Type", "Filename"]]

        for i, ds in enumerate(datasets):

            fname = ds.get("filename", f"Dataset {i+1}")
            label = f"Dataset {i+1}"

            if assignments:
                if assignments.get("synthetic") == i:
                    label += " (Synthetic)"
                elif assignments.get("real") == i:
                    label += " (Real)"

            table_data.append([label, fname])

        table = Table(table_data)

        table.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
            ("GRID",(0,0),(-1,-1),1,colors.black),
            ("ALIGN",(0,0),(-1,-1),"CENTER")
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.3 * inch))

    # --------------------------------------------------
    # Model Information
    # --------------------------------------------------

    if models:
        elements.append(Paragraph("Predictive Models Used", styles["Heading2"]))
        elements.append(Spacer(1, 0.2 * inch))

        model_table = [["Model File"]]

        # If models is a dict
        if isinstance(models, dict):
            model_table.append([models.get("filename", "Unknown")])
        # If models is a list of dicts
        elif isinstance(models, list):
            for m in models:
                model_table.append([m.get("filename", "Unknown")])

        table = Table(model_table)
        table.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
            ("GRID",(0,0),(-1,-1),1,colors.black),
            ("ALIGN",(0,0),(-1,-1),"CENTER")
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.3 * inch))


    # --------------------------------------------------
    # Helper Table Generator
    # --------------------------------------------------

    def metric_table(metrics):

        data = [["Metric", "Score"]]

        for key, value in metrics.items():
            data.append([key, f"{value:.3f}"])

        table = Table(data)

        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0),(-1,0),colors.lightgrey),
            ("GRID",(0,0),(-1,-1),1,colors.black),
            ("ALIGN",(1,1),(-1,-1),"CENTER")
        ]))

        return table

    # --------------------------------------------------
    # DPCM2 Section
    # --------------------------------------------------

    if dpcm2:

        elements.append(Paragraph("1. DPCM2 Evaluation", styles["Heading2"]))
        elements.append(Spacer(1, 0.15 * inch))

        overall = dpcm2.get("dpcm2_final_score")
        feature_scores = dpcm2.get("attributes")

        if overall is not None:

            elements.append(
                Paragraph(f"DPCM2 Overall Score: {overall:.2f}%", styles["Normal"])
            )

            elements.append(Spacer(1, 0.2 * inch))

        if feature_scores:

            data = [["Feature", "Score"]]

            for feature, score in feature_scores.items():
                data.append([feature, f"{score:.3f}"])

            table = Table(data)

            table.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
                ("GRID",(0,0),(-1,-1),1,colors.black),
                ("ALIGN",(1,1),(-1,-1),"CENTER")
            ]))

            elements.append(table)
            elements.append(Spacer(1, 0.2 * inch))

        elements.append(Paragraph(
            "Higher DPCM2 scores indicate stronger similarity between "
            "original and synthetic datasets.",
            styles["Italic"]
        ))

        elements.append(Spacer(1, 0.3 * inch))

    # --------------------------------------------------
    # Resemblance Section
    # --------------------------------------------------

    if resemblance:

        js = resemblance.get("js")
        ks = resemblance.get("ks")
        w = resemblance.get("wasserstein")

        table_data = [["Test", "Result"]]

        if js is not None:
            table_data.append(["Jensen-Shannon Divergence", f"{js:.4f}"])

        if ks is not None:
            table_data.append(["Kolmogorov-Smirnov D-Statistic", f"{ks:.4f}"])

        if w is not None:
            table_data.append(["Wasserstein Distance", f"{w:.4f}"])

        if len(table_data) > 1:

            elements.append(Paragraph("2. Resemblance Analysis", styles["Heading2"]))
            elements.append(Spacer(1, 0.2 * inch))

            table = Table(table_data)

            table.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
                ("GRID",(0,0),(-1,-1),1,colors.black),
                ("ALIGN",(1,1),(-1,-1),"CENTER")
            ]))

            elements.append(table)

            elements.append(Spacer(1, 0.2 * inch))

            elements.append(Paragraph(
                "Lower values indicate closer similarity between the datasets.",
                styles["Italic"]
            ))

            elements.append(Spacer(1, 0.3 * inch))

    # --------------------------------------------------
    # Utility Section
    # --------------------------------------------------

    if utility:

        elements.append(Paragraph("3. Utility Evaluation", styles["Heading2"]))
        elements.append(Spacer(1, 0.2 * inch))

        tstr_metrics = utility.get("tstr")
        trtr_metrics = utility.get("trtr")

        if tstr_metrics:

            elements.append(
                Paragraph("TSTR (Train Synthetic, Test Real)", styles["Heading3"])
            )

            elements.append(Spacer(1, 0.1 * inch))
            elements.append(metric_table(tstr_metrics))
            elements.append(Spacer(1, 0.2 * inch))

        if trtr_metrics:

            elements.append(
                Paragraph("TRTR (Train Real, Test Real)", styles["Heading3"])
            )

            elements.append(Spacer(1, 0.1 * inch))
            elements.append(metric_table(trtr_metrics))
            elements.append(Spacer(1, 0.2 * inch))

        elements.append(Paragraph(
            "TSTR measures whether synthetic data can train models that "
            "generalize to real-world cases. TRTR provides the real-data baseline.",
            styles["Italic"]
        ))

        elements.append(Spacer(1, 0.3 * inch))

    # --------------------------------------------------
    # XAI Section
    # --------------------------------------------------

    if xai:

        elements.append(Paragraph("4. Explainable AI (SHAP)", styles["Heading2"]))
        elements.append(Spacer(1, 0.2 * inch))

        images = xai.get("images", [])

        for img_b64 in images:

            img_bytes = base64.b64decode(img_b64)
            img_buffer = io.BytesIO(img_bytes)

            elements.append(
                Image(img_buffer, width=5*inch, height=3*inch)
            )

            elements.append(Spacer(1, 0.2 * inch))

        elements.append(Paragraph(
            "SHAP plots show feature influence on model predictions.",
            styles["Italic"]
        ))

        elements.append(Spacer(1, 0.3 * inch))

    doc.build(elements)

    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes
from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def add_first_page_elements(elements, filename, grand_weighted, qtr_to_avg):
    styles = getSampleStyleSheet()
    # Title: centered, large, bold, with filename
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Title'],
        alignment=1,  # center
        fontSize=22,
        spaceAfter=16,
        leading=26,
    )
    elements.append(Paragraph(f"{filename} Weighted Average Days to Pay Report", title_style))
    elements.append(Spacer(1, 20))
    # Grand weighted average: centered, bold
    grand_style = ParagraphStyle(
        'Grand',
        parent=styles['Heading2'],
        alignment=1,  # center
        fontSize=16,
        textColor=colors.HexColor("#003366"),
        leading=20,
    )
    elements.append(Paragraph(f"Grand Weighted Average Days Late: <b>{grand_weighted}</b>", grand_style))
    elements.append(Spacer(1, 18))
    # Quarterly averages as a grid
    qtrs = sorted(qtr_to_avg.keys())
    data = []
    row = []
    for idx, q in enumerate(qtrs):
        row.append(f"<b>{q}</b>: {qtr_to_avg[q]}")
        # 3 quarters per row, change to 4 for a wider page
        if (idx + 1) % 3 == 0:
            data.append(row)
            row = []
    if row:
        data.append(row)
    qtr_table = Table(data, style=[
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 12),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 8),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.HexColor("#003366")),
    ])
    elements.append(qtr_table)
    elements.append(Spacer(1, 20))

# USAGE EXAMPLE (to be included at the start of your generate_pdf_report_grouped function):

# elements = []
# add_first_page_elements(elements, filename, grand_weighted, qtr_to_avg)
# ... then proceed with the rest of your PDF content as before ...

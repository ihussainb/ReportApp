import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
from datetime import timedelta
from io import BytesIO
from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import matplotlib.pyplot as plt

DATE_FMT = '%d-%b-%y'
CREDIT_PERIOD_DAYS = 30
EXCLUDE_TYPES = {'CREDIT NOTE - C25', 'JOURNAL - C25'}

QUARTER_MONTHS = {1: "Apr–Jun", 2: "Jul–Sep", 3: "Oct–Dec", 4: "Jan–Mar"}

def robust_parse_dates(df, date_col="Date"):
    dt = pd.to_datetime(df[date_col], format=DATE_FMT, errors='coerce')
    mask = dt.isna()
    if mask.any():
        dt_alt = pd.to_datetime(df.loc[mask, date_col], errors='coerce')
        dt.loc[mask] = dt_alt
    failed = df[date_col][dt.isna()]
    return dt, failed

def parse_float(s):
    try:
        return float(str(s).replace(',', '').strip()) if s and str(s).strip() else 0.0
    except Exception:
        return 0.0

def get_fiscal_quarter_label(dt):
    month = dt.month
    year = dt.year
    if month >= 4 and month <= 6:
        quarter = 1
        months = QUARTER_MONTHS[quarter]
    elif month >= 7 and month <= 9:
        quarter = 2
        months = QUARTER_MONTHS[quarter]
    elif month >= 10 and month <= 12:
        quarter = 3
        months = QUARTER_MONTHS[quarter]
    else:  # Jan-Mar
        quarter = 4
        year -= 1  # assign to previous FY
        months = QUARTER_MONTHS[quarter]
    return f"{year} Q{quarter} {months}", year, quarter

@st.cache_data(show_spinner=False)
def analyze_ledger(df):
    sales = []
    payments = []
    problematic_rows = []

    df["Parsed_Date"], parse_failures = robust_parse_dates(df, "Date")

    for idx, row in df.iterrows():
        vch_type = str(row.get('Vch Type', '')).strip()
        if vch_type in EXCLUDE_TYPES:
            continue

        parsed_date = row.get("Parsed_Date")
        if pd.isna(parsed_date):
            problematic_rows.append(row)
            continue

        debit = parse_float(row.get('Debit', ''))
        credit = parse_float(row.get('Credit', ''))
        vch_no = str(row.get('Vch No.', '')).strip()
        particulars = str(row.get('Particulars', '')).strip()
        if particulars.lower() == "opening balance":
            sales.append({
                'date': parsed_date,
                'vch_no': 'Opening Balance',
                'amount': debit,
                'due_date': parsed_date + timedelta(days=CREDIT_PERIOD_DAYS),
                'remaining': debit,
                'payments': []
            })
        elif debit > 0:
            sales.append({
                'date': parsed_date,
                'vch_no': vch_no,
                'amount': debit,
                'due_date': parsed_date + timedelta(days=CREDIT_PERIOD_DAYS),
                'remaining': debit,
                'payments': []
            })
        elif credit > 0:
            payments.append({
                'date': parsed_date,
                'amount': credit
            })

    sale_idx = 0
    for payment in payments:
        amt_left = payment['amount']
        while amt_left > 0 and sale_idx < len(sales):
            sale = sales[sale_idx]
            to_apply = min(amt_left, sale['remaining'])
            if to_apply > 0:
                sale['payments'].append({
                    'pay_date': payment['date'],
                    'pay_amt': to_apply
                })
                sale['remaining'] -= to_apply
                amt_left -= to_apply
            if sale['remaining'] <= 0.01:
                sale['remaining'] = 0
                sale_idx += 1
            elif amt_left == 0:
                break

    rows = []
    total_impact = 0.0
    total_amount = 0.0
    for sale in sales:
        remaining = sale['remaining']
        total_per_invoice = 0.0
        for pay in sale['payments']:
            days_late = (pay['pay_date'] - sale['due_date']).days
            total_per_invoice += pay['pay_amt'] * days_late
        weighted_days = total_per_invoice / sale['amount'] if sale['amount'] else 0
        rows.append({
            'Sale_Date': sale['date'],
            'Invoice_No': sale['vch_no'],
            'Sale_Amount': sale['amount'],
            'Weighted_Days_Late': round(weighted_days, 2),
            'Amount_Remaining': round(remaining, 2),
            'Due_Date': sale['due_date']
        })
        if abs(remaining) < 1e-2:
            total_impact += total_per_invoice
            total_amount += sale['amount']

    df_rows = pd.DataFrame(rows)
    # Assign concise quarter label, fiscal year and quarter number
    q_labels = df_rows['Sale_Date'].apply(lambda d: get_fiscal_quarter_label(pd.to_datetime(d)))
    df_rows[['Quarter_Label', 'Fiscal_Year', 'Fiscal_Quarter']] = pd.DataFrame(q_labels.tolist(), index=df_rows.index)
    paid = df_rows[df_rows['Amount_Remaining'] < 0.01]

    # Sort by fiscal year and quarter
    paid = paid.sort_values(['Fiscal_Year', 'Fiscal_Quarter'])
    qtr_to_avg = {}
    for _, group in paid.groupby(['Fiscal_Year', 'Fiscal_Quarter', 'Quarter_Label']):
        label = group['Quarter_Label'].iloc[0]
        qtr_avg = np.average(group['Weighted_Days_Late'], weights=group['Sale_Amount'])
        qtr_to_avg[label] = round(qtr_avg, 2)

    # Quarterly payment weightage (% of total paid in each quarter)
    quarter_amounts = paid.groupby('Quarter_Label')['Sale_Amount'].sum()
    total_paid = quarter_amounts.sum()
    quarter_weightage = (quarter_amounts / total_paid * 100).round(1)

    df_rows['Sale_Date'] = df_rows['Sale_Date'].dt.strftime('%d-%b-%y')
    df_rows['Due_Date'] = df_rows['Due_Date'].dt.strftime('%d-%b-%y')

    grand_weighted = round(total_impact / total_amount, 2) if total_amount else 0.0

    return df_rows, grand_weighted, qtr_to_avg, quarter_weightage, problematic_rows

def add_first_page_elements(elements, report_title, grand_weighted, qtr_to_avg, quarter_weightage):
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    import os

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'Title', parent=styles['Title'], alignment=1, fontSize=20,
        spaceAfter=10, leading=24,
    )
    subtitle_style = ParagraphStyle(
        'Subtitle', parent=styles['Title'], alignment=1, fontSize=15,
        spaceAfter=2, leading=18,
    )
    grand_style = ParagraphStyle(
        'Grand', parent=styles['Heading2'], alignment=1,
        fontSize=14, textColor=colors.HexColor("#003366"), leading=18,
    )

    clean_filename = os.path.splitext(os.path.basename(report_title))[0]
    elements.append(Paragraph(f"{clean_filename}", title_style))
    elements.append(Paragraph("By Fiscal Year — 30 Days Credit Period", subtitle_style))
    elements.append(Paragraph(f"Grand Weighted Avg Days Late: <b>{grand_weighted}</b>", grand_style))
    elements.append(Spacer(1, 12))

    data = [["Quarter", "Weighted Avg Days Late", "% Paid Amount"]]
    for q in qtr_to_avg.keys():
        weight = f"{quarter_weightage.get(q, 0.0):.1f}%" if q in quarter_weightage else ""
        data.append([q, f"{qtr_to_avg[q]:.2f}", weight])
    table = Table(data, colWidths=[220, 170, 110], hAlign='CENTER')
    table.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 13),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#003366")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f0f0f0")]),
        ("GRID", (0,0), (-1,-1), 0.3, colors.gray),
        ("FONTSIZE", (0,1), (-1,-1), 12),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 6),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 18))

def generate_pdf_report_grouped(df_rows, grand_weighted, qtr_to_avg, quarter_weightage, buffer, report_title, chart_path=None):
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(LETTER),
        rightMargin=30, leftMargin=30,
        topMargin=30, bottomMargin=30,
    )
    elements = []
    add_first_page_elements(elements, report_title, grand_weighted, qtr_to_avg, quarter_weightage)
    if chart_path:
        elements.append(Image(chart_path, width=500, height=250))
        elements.append(Spacer(1, 18))
    styles = getSampleStyleSheet()
    styleQ = ParagraphStyle(
        'QuarterStyle', parent=styles['Heading2'],
        spaceAfter=4, spaceBefore=12, textColor=colors.HexColor("#003366")
    )
    unique_quarters = df_rows.drop_duplicates(subset=["Quarter_Label"])[
        ["Fiscal_Year", "Fiscal_Quarter", "Quarter_Label"]
    ].sort_values(["Fiscal_Year", "Fiscal_Quarter"])
    table_header = ["Sale Date", "Invoice No", "Sale Amount", "Weighted Days Late", "Amount Remaining"]
    for _, quarter_row in unique_quarters.iterrows():
        qlabel = quarter_row["Quarter_Label"]
        q_rows = df_rows[df_rows["Quarter_Label"] == qlabel]
        elements.append(Paragraph(f"{qlabel}: Weighted Avg Days Late = {qtr_to_avg.get(qlabel, '')}", styleQ))
        data = [table_header]
        for _, row in q_rows.iterrows():
            data.append([
                row["Sale_Date"],
                row["Invoice_No"],
                f"{row['Sale_Amount']:,.2f}",
                f"{row['Weighted_Days_Late']:.2f}",
                f"{row['Amount_Remaining']:,.2f}",
            ])
        t = Table(data, colWidths=[90, 140, 100, 120, 120])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003366")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (2, 1), (4, -1), "RIGHT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 11),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f6f6f6")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#e6e6e6")]),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.gray),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 18))
    doc.build(elements)

st.title("Ledger Weighted Average Days to Pay Report (Fiscal Year Quarterly Grouped)")
st.markdown("""
Upload your Tally ledger CSV file (must have columns: Date, Particulars, Vch Type, Vch No., Debit, Credit).
""")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if "Date" not in df.columns:
            st.error("Date column not found in uploaded CSV!")
            st.stop()
        st.success("File uploaded and read successfully.")
        st.write("Preview:", df.head())
        df_rows, grand_weighted, qtr_to_avg, quarter_weightage, problematic_rows = analyze_ledger(df)
        if not df_rows.empty:
            st.markdown(f"### Grand Weighted Avg Days Late: **{grand_weighted}**")
            st.markdown("#### By Fiscal Year — 30 Days Credit Period")
            summary_df = pd.DataFrame([
                {
                    "Quarter": q,
                    "Weighted Avg Days Late": qtr_to_avg[q],
                    "% Paid Amount": f"{quarter_weightage.get(q, 0.0):.1f}%"
                }
                for q in qtr_to_avg.keys()
            ])
            st.dataframe(summary_df)
            # Display grouped DataFrame table
            unique_quarters = df_rows.drop_duplicates(subset=["Quarter_Label"])[
                ["Fiscal_Year", "Fiscal_Quarter", "Quarter_Label"]
            ].sort_values(["Fiscal_Year", "Fiscal_Quarter"])
            for _, quarter_row in unique_quarters.iterrows():
                qlabel = quarter_row["Quarter_Label"]
                st.markdown(f"### {qlabel}: Weighted Avg Days Late = {qtr_to_avg.get(qlabel, '')}")
                q_rows = df_rows[df_rows['Quarter_Label'] == qlabel]
                st.dataframe(q_rows[
                    ["Sale_Date", "Invoice_No", "Sale_Amount", "Weighted_Days_Late", "Amount_Remaining"]
                ])
            # Chart: Only plot actual data, no quarter averages or extra legend items!
            df_rows["Sale_Date_dt"] = pd.to_datetime(df_rows["Sale_Date"], format="%d-%b-%y", errors="coerce")
            df_rows = df_rows.dropna(subset=["Sale_Date_dt"])
            df_rows = df_rows.sort_values("Sale_Date_dt")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(
                df_rows["Sale_Date_dt"],
                df_rows["Weighted_Days_Late"],
                marker="o",
                label="Weighted Days Late"
            )
            ax.set_xlabel("Sale Date")
            ax.set_ylabel("Days Late")
            ax.set_title("Weighted Days Late Over Time")
            ax.legend(["Weighted Days Late"])
            plt.grid(True)
            st.pyplot(fig, use_container_width=True)
            chart_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig.savefig(chart_temp.name, bbox_inches='tight')
            plt.close(fig)
            pdf_buffer = BytesIO()
            generate_pdf_report_grouped(
                df_rows,
                grand_weighted,
                qtr_to_avg,
                quarter_weightage,
                pdf_buffer,
                uploaded_file.name,
                chart_path=chart_temp.name
            )
            pdf_buffer.seek(0)
            st.download_button(
                label="Download PDF Report",
                data=pdf_buffer,
                file_name="WADP_Report.pdf",
                mime="application/pdf"
            )
        else:
            st.warning("No sales found in this data.")
        if len(problematic_rows) > 0:
            st.markdown("#### ⚠️ The following rows have problematic or missing dates and were skipped:")
            st.dataframe(pd.DataFrame(problematic_rows))
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Awaiting CSV file upload.")

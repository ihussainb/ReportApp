import streamlit as st
import pandas as pd
import numpy as np
import tempfile
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
import matplotlib.pyplot as plt

DATE_FMT = '%d-%b-%y'
CREDIT_PERIOD_DAYS = 30
EXCLUDE_TYPES = {'CREDIT NOTE - C25', 'JOURNAL - C25'}

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

def get_quarter(dt):
    return f"{dt.year}-Q{((dt.month-1)//3)+1}"

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
        status = (
            "Paid" if abs(remaining) < 1e-2 else
            ("Partial" if sale['payments'] else "Unpaid")
        )
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
            'Status': status,
            'Payments': sale['payments'],
            'Due_Date': sale['due_date']
        })
        if status == "Paid":
            total_impact += total_per_invoice
            total_amount += sale['amount']

    # Calculate quarters and quarterly weighted averages
    df_rows = pd.DataFrame(rows)
    df_rows['Sale_Quarter'] = df_rows['Sale_Date'].apply(get_quarter)
    paid = df_rows[df_rows['Status'] == 'Paid']

    # Compute quarterly weighted averages (by Sale_Quarter)
    qtr_to_avg = {}
    for q, g in paid.groupby('Sale_Quarter'):
        qtr_avg = np.average(g['Weighted_Days_Late'], weights=g['Sale_Amount'])
        qtr_to_avg[q] = round(qtr_avg, 2)

    # Add column to each row with its quarter's average
    df_rows['Quarterly Weighted Avg Days Late'] = df_rows['Sale_Quarter'].apply(lambda q: qtr_to_avg.get(q, np.nan))

    # Prepare summary for display at the top
    qtr_summary = pd.DataFrame([
        {"Quarter": q, "Quarterly Weighted Avg Days Late": avg}
        for q, avg in qtr_to_avg.items()
    ]).sort_values("Quarter")

    # Format for display
    df_rows['Sale_Date'] = df_rows['Sale_Date'].dt.strftime('%d-%b-%y')
    df_rows['Due_Date'] = df_rows['Due_Date'].dt.strftime('%d-%b-%y')

    grand_weighted = round(total_impact / total_amount, 2) if total_amount else 0.0

    return df_rows, grand_weighted, qtr_summary, problematic_rows

def generate_pdf_report(df_rows, grand_weighted, qtr_summary, filename, chart_path=None):
    doc = SimpleDocTemplate(
        filename,
        pagesize=landscape(LETTER),
        rightMargin=30, leftMargin=30,
        topMargin=30, bottomMargin=30,
    )
    styles = getSampleStyleSheet()
    styleN = styles["BodyText"]
    styleH = styles["Heading1"]
    elements = []
    elements.append(Paragraph("Weighted Average Days to Pay Report", styleH))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Grand Weighted Average Days Late: <b>{grand_weighted}</b>", styleN))
    elements.append(Spacer(1, 12))
    if not qtr_summary.empty:
        for idx, row in qtr_summary.iterrows():
            elements.append(Paragraph(f"{row['Quarter']}: <b>{row['Quarterly Weighted Avg Days Late']}</b>", styleN))
    elements.append(Spacer(1, 24))
    if chart_path:
        elements.append(Image(chart_path, width=500, height=250))
        elements.append(Spacer(1, 24))
    data = [["Sale Date", "Invoice No", "Sale Amount", "Weighted Days Late", "Quarterly Weighted Avg Days Late", "Amount Remaining", "Status"]]
    for _, row in df_rows.iterrows():
        data.append([
            row["Sale_Date"],
            row["Invoice_No"],
            f"{row['Sale_Amount']:,.2f}",
            f"{row['Weighted_Days_Late']:.2f}",
            f"{row['Quarterly Weighted Avg Days Late']:.2f}" if pd.notnull(row['Quarterly Weighted Avg Days Late']) else "",
            f"{row['Amount_Remaining']:,.2f}",
            row["Status"]
        ])
    t = Table(data, colWidths=[90, 140, 100, 120, 160, 120, 100])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003366")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (2, 1), (5, -1), "RIGHT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 11),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f6f6f6")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#e6e6e6")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.gray),
    ]))
    elements.append(t)
    doc.build(elements)

st.title("Ledger Weighted Average Days to Pay Report (with Quarterly Average Column)")
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
        df_rows, grand_weighted, qtr_summary, problematic_rows = analyze_ledger(df)
        if not df_rows.empty:
            st.markdown(f"### Grand Weighted Average Days Late: **{grand_weighted}**")
            if not qtr_summary.empty:
                st.markdown("#### Quarterly Weighted Average Days Late (by Invoice Date):")
                for idx, row in qtr_summary.iterrows():
                    st.markdown(f"- **{row['Quarter']}**: {row['Quarterly Weighted Avg Days Late']}")
            st.dataframe(df_rows[
                ["Sale_Date", "Invoice_No", "Sale_Amount", "Weighted_Days_Late",
                 "Quarterly Weighted Avg Days Late", "Amount_Remaining", "Status"]
            ])

            # Visualization: Line chart of Weighted Days Late over Sale Date
            df_rows["Sale_Date_dt"] = pd.to_datetime(df_rows["Sale_Date"], format="%d-%b-%y", errors="coerce")
            df_rows = df_rows.dropna(subset=["Sale_Date_dt"])
            df_rows = df_rows.sort_values("Sale_Date_dt")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df_rows["Sale_Date_dt"], df_rows["Weighted_Days_Late"], marker="o", label="Weighted Days Late")
            ax.plot(df_rows["Sale_Date_dt"], df_rows["Quarterly Weighted Avg Days Late"], linestyle="--", color="orange", label="Quarterly Avg")
            ax.set_xlabel("Sale Date")
            ax.set_ylabel("Days Late")
            ax.set_title("Weighted Days Late (with Quarterly Average) Over Time")
            ax.legend()
            plt.grid(True)
            st.pyplot(fig, use_container_width=True)

            # Save the chart as image for PDF
            chart_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig.savefig(chart_temp.name, bbox_inches='tight')
            plt.close(fig)

            # PDF generation with chart
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                generate_pdf_report(df_rows, grand_weighted, qtr_summary, tmpfile.name, chart_path=chart_temp.name)
                tmpfile.seek(0)
                st.download_button(
                    label="Download PDF Report",
                    data=tmpfile.read(),
                    file_name="WADP_Report.pdf",
                    mime="application/pdf"
                )
        else:
            st.warning("No sales found in this data.")

        # Show problematic date rows, if any
        if len(problematic_rows) > 0:
            st.markdown("#### ⚠️ The following rows have problematic or missing dates and were skipped:")
            st.dataframe(pd.DataFrame(problematic_rows))

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Awaiting CSV file upload.")

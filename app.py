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

        date = row.get('Date')
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

    table = []
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
        table.append({
            'Sale_Date': sale['date'].strftime('%d-%b-%y'),
            'Invoice_No': sale['vch_no'],
            'Sale_Amount': sale['amount'],
            'Weighted_Days_Late': round(weighted_days, 2),
            'Amount_Remaining': round(remaining, 2),
            'Status': status,
            'Payments': sale['payments'],
            'Due_Date': sale['due_date'].strftime('%d-%b-%y')
        })
        if status == "Paid":
            total_impact += total_per_invoice
            total_amount += sale['amount']
    grand_weighted = round(total_impact / total_amount, 2) if total_amount else 0.0
    return table, grand_weighted, problematic_rows

def get_quarter(dt):
    return f"{dt.year}-Q{((dt.month-1)//3)+1}"

def compute_quarterly_averages(table):
    df = pd.DataFrame(table)
    df['Sale_Date_dt'] = pd.to_datetime(df['Sale_Date'], format='%d-%b-%y', errors='coerce')
    df['Sale_Quarter'] = df['Sale_Date_dt'].apply(get_quarter)
    paid = df[df['Status'] == 'Paid']
    # Invoice-based quarterly weighted avg (weighted by invoice amount)
    quarterly_invoice = (
        paid.groupby('Sale_Quarter')
            .apply(lambda g: np.average(g['Weighted_Days_Late'], weights=g['Sale_Amount']))
            .reset_index(name='Quarterly_Weighted_Avg_Invoice')
    )

    # Payment-based quarterly calculation
    payment_rows = []
    for row in table:
        if row['Payments']:
            sale_due = pd.to_datetime(row['Due_Date'], format='%d-%b-%y', errors='coerce')
            for p in row['Payments']:
                days_late = (p['pay_date'] - sale_due).days
                payment_rows.append({
                    'Payment_Date': p['pay_date'],
                    'Amount': p['pay_amt'],
                    'Days_Late': days_late
                })
    if payment_rows:
        payment_df = pd.DataFrame(payment_rows)
        payment_df['Payment_Quarter'] = payment_df['Payment_Date'].apply(get_quarter)
        quarterly_payment = (
            payment_df.groupby('Payment_Quarter')
            .apply(lambda g: np.average(g['Days_Late'], weights=g['Amount']))
            .reset_index(name='Quarterly_Weighted_Avg_Payment')
        )
    else:
        quarterly_payment = pd.DataFrame(columns=['Payment_Quarter', 'Quarterly_Weighted_Avg_Payment'])

    return quarterly_invoice, quarterly_payment

def generate_pdf_report(table, grand_weighted, filename, chart_path=None):
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
    elements.append(Spacer(1, 24))
    if chart_path:
        elements.append(Image(chart_path, width=500, height=250))
        elements.append(Spacer(1, 24))
    data = [["Sale Date", "Invoice No", "Sale Amount", "Weighted Days Late", "Amount Remaining", "Status"]]
    for row in table:
        data.append([
            row["Sale_Date"],
            row["Invoice_No"],
            f"{row['Sale_Amount']:,.2f}",
            f"{row['Weighted_Days_Late']:.2f}",
            f"{row['Amount_Remaining']:,.2f}",
            row["Status"]
        ])
    t = Table(data, colWidths=[90, 140, 100, 120, 120, 100])
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
    doc.build(elements)

st.title("Ledger Weighted Average Days to Pay Report (with Quarterly Analysis)")
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
        table, grand_weighted, problematic_rows = analyze_ledger(df)
        if table:
            st.markdown(f"### Grand Weighted Average Days Late: **{grand_weighted}**")
            df_results = pd.DataFrame(table)
            st.dataframe(df_results)

            # Quarterly analysis
            quarterly_invoice, quarterly_payment = compute_quarterly_averages(table)
            st.markdown("### Quarterly Weighted Average Days Late (Invoices)")
            st.dataframe(quarterly_invoice)
            st.markdown("### Quarterly Weighted Average Days Late (Payments)")
            st.dataframe(quarterly_payment)

            # Visualization: Line chart of Weighted Days Late over Sale Date
            df_results["Sale_Date_dt"] = pd.to_datetime(df_results["Sale_Date"], format="%d-%b-%y", errors="coerce")
            df_results = df_results.dropna(subset=["Sale_Date_dt"])
            df_results = df_results.sort_values("Sale_Date_dt")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df_results["Sale_Date_dt"], df_results["Weighted_Days_Late"], marker="o")
            ax.set_xlabel("Sale Date")
            ax.set_ylabel("Weighted Days Late")
            ax.set_title("Weighted Days Late Over Time")
            plt.grid(True)
            st.pyplot(fig, use_container_width=True)

            # Save the chart as image for PDF
            chart_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig.savefig(chart_temp.name, bbox_inches='tight')
            plt.close(fig)

            # PDF generation with chart
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                generate_pdf_report(table, grand_weighted, tmpfile.name, chart_path=chart_temp.name)
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

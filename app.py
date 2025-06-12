import streamlit as st
import pandas as pd
import tempfile
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer

DATE_FMT = '%d-%b-%y'
CREDIT_PERIOD_DAYS = 30
EXCLUDE_TYPES = {'CREDIT NOTE - C25', 'JOURNAL - C25'}

def parse_float(s):
    try:
        return float(str(s).replace(',', '').strip()) if s and str(s).strip() else 0.0
    except Exception:
        return 0.0

def parse_date(s):
    return datetime.strptime(str(s).strip(), DATE_FMT)

@st.cache_data(show_spinner=False)
def analyze_ledger(df):
    # Prepare sales and payments lists
    sales = []
    payments = []
    for _, row in df.iterrows():
        vch_type = str(row.get('Vch Type', '')).strip()
        if vch_type in EXCLUDE_TYPES:
            continue
        date = row.get('Date')
        if not isinstance(date, str) or not any(c.isalpha() for c in date):
            continue
        debit = parse_float(row.get('Debit', ''))
        credit = parse_float(row.get('Credit', ''))
        vch_no = str(row.get('Vch No.', '')).strip()
        particulars = str(row.get('Particulars', '')).strip()
        if particulars.lower() == "opening balance":
            sale_date = parse_date(date)
            sales.append({
                'date': sale_date,
                'vch_no': 'Opening Balance',
                'amount': debit,
                'due_date': sale_date + timedelta(days=CREDIT_PERIOD_DAYS),
                'remaining': debit,
                'payments': []
            })
        elif debit > 0:
            sale_date = parse_date(date)
            sales.append({
                'date': sale_date,
                'vch_no': vch_no,
                'amount': debit,
                'due_date': sale_date + timedelta(days=CREDIT_PERIOD_DAYS),
                'remaining': debit,
                'payments': []
            })
        elif credit > 0:
            payments.append({
                'date': parse_date(date),
                'amount': credit
            })

    # Apply FIFO payments
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

    # Prepare results table and grand weighted
    table = []
    total_impact = 0.0
    total_amount = 0.0
    for sale in sales:
        if abs(sale['remaining']) < 1e-2 and sale['payments']:
            total_per_invoice = 0.0
            for pay in sale['payments']:
                days_late = (pay['pay_date'] - sale['due_date']).days
                total_per_invoice += pay['pay_amt'] * days_late
            weighted_days = total_per_invoice / sale['amount'] if sale['amount'] else 0
            table.append({
                'Sale_Date': sale['date'].strftime('%d-%b-%y'),
                'Invoice_No': sale['vch_no'],
                'Sale_Amount': sale['amount'],
                'Weighted_Days_Late': round(weighted_days, 2)
            })
            total_impact += total_per_invoice
            total_amount += sale['amount']
    grand_weighted = round(total_impact / total_amount, 2) if total_amount else 0.0
    return table, grand_weighted

def generate_pdf_report(table, grand_weighted, filename):
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
    data = [["Sale Date", "Invoice No", "Sale Amount", "Weighted Days Late"]]
    for row in table:
        data.append([
            row["Sale_Date"],
            row["Invoice_No"],
            f"{row['Sale_Amount']:,.2f}",
            f"{row['Weighted_Days_Late']:.2f}",
        ])
    t = Table(data, colWidths=[90, 140, 100, 120])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003366")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (2, 1), (3, -1), "RIGHT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 11),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f6f6f6")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#e6e6e6")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.gray),
    ]))
    elements.append(t)
    doc.build(elements)

st.title("Ledger Weighted Average Days to Pay Report")
st.markdown("""
Upload your Tally ledger CSV file (must have columns: Date, Particulars, Vch Type, Vch No., Debit, Credit).
""")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded and read successfully.")
        st.write("Preview:", df.head())
        table, grand_weighted = analyze_ledger(df)
        if table:
            st.markdown(f"### Grand Weighted Average Days Late: **{grand_weighted}**")
            st.dataframe(pd.DataFrame(table))
            # PDF generation
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                generate_pdf_report(table, grand_weighted, tmpfile.name)
                tmpfile.seek(0)
                st.download_button(
                    label="Download PDF Report",
                    data=tmpfile.read(),
                    file_name="WADP_Report.pdf",
                    mime="application/pdf"
                )
        else:
            st.warning("No fully cleared invoices found in this data.")
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Awaiting CSV file upload.")

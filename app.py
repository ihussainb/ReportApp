import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors as rl_colors

DATE_FMT = "%d-%b-%y"
QUARTER_MONTHS = {1: "Apr–Jun", 2: "Jul–Sep", 3: "Oct–Dec", 4: "Jan–Mar"}

# --- HELPERS FOR PARSING THE MESSY TALLY EXPORT ---

def parse_tally_ledgers(file):
    """
    Parse a full Tally multi-ledger CSV export.

    Returns:
        ledgers: dict {ledger_name: DataFrame}
        ledger_addresses: dict {ledger_name: address string}
    """
    # Read as text with fallback encoding for Tally/Excel files
    if hasattr(file, "read"):
        # Read entire file as text, detect encoding
        file.seek(0)
        try:
            content = file.read().decode("utf-8")
        except Exception:
            file.seek(0)
            content = file.read().decode("latin1")
        lines = content.splitlines()
    else:
        lines = file.splitlines()
    ledgers = {}
    ledger_addresses = {}
    current_ledger = None
    current_address = None
    headers = None
    rows = []
    for i, line in enumerate(lines):
        cells = [c.strip() for c in line.strip('\n').split(",")]
        if cells[0].startswith("Ledger:"):
            # Save previous
            if current_ledger and headers and rows:
                df = pd.DataFrame(rows, columns=headers)
                ledgers[current_ledger] = df
                ledger_addresses[current_ledger] = current_address
            # New ledger
            current_ledger = cells[1]
            current_address = None
            headers = None
            rows = []
            continue
        # Address is always next line after Ledger:... (often only in 2nd col)
        if current_ledger and current_address is None and any(cells):
            current_address = cells[0]
            continue
        # Header row detection
        if (cells[:7] == ['Date', 'Particulars', '', 'Vch Type', 'Vch No.', 'Debit', 'Credit'] or
            cells[:6] == ['Date', 'Particulars', 'Vch Type', 'Vch No.', 'Debit', 'Credit']):
            # Sometimes there is no blank 3rd col
            headers = [c for c in cells if c]
            rows = []
            continue
        # Transaction rows: must have headers, not empty, not summary/closing
        if headers and (len([x for x in cells if x]) >= 5) and not (
            cells[1] if len(cells) > 1 else "").startswith("Closing Balance"):
            # Only append rows with at least a date and some value
            if cells[0] and cells[0][0].isdigit():
                # Pad short rows
                while len(cells) < len(headers):
                    cells.append("")
                rows.append(cells[:len(headers)])
    # Save last ledger
    if current_ledger and headers and rows:
        df = pd.DataFrame(rows, columns=headers)
        ledgers[current_ledger] = df
        ledger_addresses[current_ledger] = current_address
    return ledgers, ledger_addresses

def parse_float(x):
    try:
        return float(str(x).replace(",", "").strip()) if x and str(x).strip() else 0.0
    except Exception:
        return 0.0

def robust_parse_dates(df, date_col="Date"):
    dt = pd.to_datetime(df[date_col], format=DATE_FMT, errors='coerce')
    mask = dt.isna()
    if mask.any():
        dt_alt = pd.to_datetime(df.loc[mask, date_col], errors='coerce')
        dt.loc[mask] = dt_alt
    failed = df[date_col][dt.isna()]
    return dt, failed

def get_fiscal_quarter_label(dt):
    month = dt.month
    year = dt.year
    if 4 <= month <= 6:
        quarter, months = 1, QUARTER_MONTHS[1]
    elif 7 <= month <= 9:
        quarter, months = 2, QUARTER_MONTHS[2]
    elif 10 <= month <= 12:
        quarter, months = 3, QUARTER_MONTHS[3]
    else:
        quarter, months = 4, QUARTER_MONTHS[4]
        year -= 1
    return f"{year} Q{quarter} {months}", year, quarter

def format_amount_short(n):
    n = float(n)
    if abs(n) >= 100000:
        return f"{n/100000:.2f} L"
    elif abs(n) >= 1000:
        return f"{n/1000:.2f} T"
    else:
        return f"{n:.0f}"

# --- LEDGER ANALYSIS ---

def analyze_ledger(df, credit_days=0):
    """
    Accepts a single ledger's DataFrame (must have required columns).
    Returns (summary DataFrame, weighted_days_late).
    """
    sales, payments, problematic_rows = [], [], []
    df["Parsed_Date"], _ = robust_parse_dates(df, "Date")
    for _, row in df.iterrows():
        parsed_date = row.get("Parsed_Date")
        if pd.isna(parsed_date):
            problematic_rows.append(row)
            continue
        vch_type = str(row.get("Vch Type", "")).upper()
        particulars = str(row.get("Particulars", "")).strip()
        vch_no = str(row.get("Vch No.", "")).strip()
        debit = parse_float(row.get("Debit", ""))
        credit = parse_float(row.get("Credit", ""))
        if "OPENING BALANCE" in particulars.upper():
            continue
        # Sales: credit not zero, payment: debit not zero
        if credit > 0:
            sales.append({
                'date': parsed_date,
                'vch_no': vch_no,
                'amount': credit,
                'due_date': parsed_date + timedelta(days=credit_days),
                'remaining': credit,
                'payments': []
            })
        elif debit > 0:
            payments.append({
                'date': parsed_date,
                'amount': debit,
                'vch_no': vch_no,
                'particulars': particulars
            })
    # FIFO payment allocation
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
    # Calculate weighted days late
    total_impact, total_amount = 0.0, 0.0
    for sale in sales:
        total_per_invoice = 0.0
        for pay in sale['payments']:
            days_late = (pay['pay_date'] - sale['due_date']).days
            total_per_invoice += pay['pay_amt'] * days_late
        if sale['amount']:
            weighted_days = total_per_invoice / sale['amount']
            total_impact += total_per_invoice
            total_amount += sale['amount']
    weighted_days_late = round(total_impact / total_amount, 2) if total_amount else 0.0
    return weighted_days_late

def analyze_ledger_detailed(df, credit_days=0):
    """
    Returns a DataFrame for the detailed quarter-wise analysis.
    """
    sales, payments, problematic_rows = [], [], []
    df["Parsed_Date"], _ = robust_parse_dates(df, "Date")
    for _, row in df.iterrows():
        parsed_date = row.get("Parsed_Date")
        if pd.isna(parsed_date):
            problematic_rows.append(row)
            continue
        vch_type = str(row.get("Vch Type", "")).upper()
        particulars = str(row.get("Particulars", "")).strip()
        vch_no = str(row.get("Vch No.", "")).strip()
        debit = parse_float(row.get("Debit", ""))
        credit = parse_float(row.get("Credit", ""))
        if "OPENING BALANCE" in particulars.upper():
            continue
        if credit > 0:
            sales.append({
                'date': parsed_date,
                'vch_no': vch_no,
                'amount': credit,
                'due_date': parsed_date + timedelta(days=credit_days),
                'remaining': credit,
                'payments': []
            })
        elif debit > 0:
            payments.append({
                'date': parsed_date,
                'amount': debit,
                'vch_no': vch_no,
                'particulars': particulars
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
    for sale in sales:
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
        })
    df_rows = pd.DataFrame(rows)
    if not df_rows.empty:
        q_labels = df_rows['Sale_Date'].apply(lambda d: get_fiscal_quarter_label(pd.to_datetime(d)))
        df_rows[['Quarter_Label', 'Fiscal_Year', 'Fiscal_Quarter']] = pd.DataFrame(q_labels.tolist(), index=df_rows.index)
    return df_rows

# --- PDF GENERATION ---

def make_summary_pdf(summary_table, credit_days):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(LETTER), rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    elements = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('FileTitle', parent=styles['Title'], alignment=1, fontSize=22, spaceAfter=8, leading=26)
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Title'], alignment=1, fontSize=14, spaceAfter=3, leading=18)
    elements.append(Paragraph("Weighted Days Late Summary Report", title_style))
    elements.append(Paragraph(f"Credit Period: {credit_days} days", subtitle_style))
    elements.append(Spacer(1, 18))
    data = [["Company / Ledger", "Weighted Days Late"]]
    data += [[row['Ledger'], f"{row['Weighted Days Late']:.2f}"] for _, row in summary_table.iterrows()]
    table = Table(data, colWidths=[350, 200], hAlign='CENTER')
    table.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 14),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("BACKGROUND", (0,0), (-1,0), rl_colors.HexColor("#003366")),
        ("TEXTCOLOR", (0,0), (-1,0), rl_colors.white),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [rl_colors.white, rl_colors.HexColor("#f0f0f0")]),
        ("GRID", (0,0), (-1,-1), 0.3, rl_colors.gray),
        ("FONTSIZE", (0,1), (-1,-1), 13),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 6),
    ]))
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer

def make_detailed_pdf(df, ledger_name, credit_days):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(LETTER), rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    elements = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('FileTitle', parent=styles['Title'], alignment=1, fontSize=22, spaceAfter=8, leading=26)
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Title'], alignment=1, fontSize=14, spaceAfter=3, leading=18)
    elements.append(Paragraph(f"Detailed Report: {ledger_name}", title_style))
    elements.append(Paragraph(f"Credit Period: {credit_days} days", subtitle_style))
    elements.append(Spacer(1, 18))
    if df.empty:
        elements.append(Paragraph("No data for this company/ledger.", styles["Normal"]))
    else:
        data = [["Sale Date", "Invoice No", "Sale Amount", "Weighted Days Late", "Quarter"]]
        for _, row in df.iterrows():
            data.append([
                pd.to_datetime(row["Sale_Date"]).strftime("%d-%b-%y") if "Sale_Date" in row else "",
                row.get("Invoice_No", ""),
                format_amount_short(row.get("Sale_Amount", "")),
                f"{row.get('Weighted_Days_Late', 0):.2f}",
                row.get("Quarter_Label", ""),
            ])
        table = Table(data, colWidths=[90, 120, 120, 140, 150], hAlign='CENTER')
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#003366")),
            ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
            ("ALIGN", (2, 1), (4, -1), "RIGHT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 12),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("BACKGROUND", (0, 1), (-1, -1), rl_colors.HexColor("#f6f6f6")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor("#e6e6e6")]),
            ("GRID", (0, 0), (-1, -1), 0.25, rl_colors.gray),
        ]))
        elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer

# --- STREAMLIT APP ---

st.set_page_config(page_title="Tally Ledger Weighted Days Late Analyzer", layout="wide")
st.title("Tally Ledger Weighted Days Late Analyzer")

st.markdown("""
Upload a **full Tally ledger group CSV export**.
This app will:
- Parse every company/ledger,
- Let you set a credit period (0 = invoice date as due date, N = N days credit),
- Show weighted days late for each ledger (summary table),
- Let you select any company/ledger for a detailed drilldown (quarter-wise).
""")

uploaded = st.file_uploader("Upload Tally CSV (multi-ledger export)", type=["csv"])
credit_days = st.number_input("Credit Period (days)", min_value=0, value=0, step=1, help="0 means due date is invoice date.")

if uploaded:
    ledgers, ledger_addresses = parse_tally_ledgers(uploaded)
    if not ledgers:
        st.error("No ledgers found in file. Make sure this is a Group Summary CSV export from Tally.")
        st.stop()
    st.success(f"Parsed {len(ledgers)} ledgers from file.")
    summary_rows = []
    for ledger_name, df in ledgers.items():
        try:
            wd = analyze_ledger(df, credit_days)
            summary_rows.append({"Ledger": ledger_name, "Weighted Days Late": wd})
        except Exception as e:
            summary_rows.append({"Ledger": ledger_name, "Weighted Days Late": float('nan')})
    summary_table = pd.DataFrame(summary_rows)
    st.subheader("Summary: Weighted Days Late by Company/Ledger")
    st.dataframe(summary_table, use_container_width=True)
    if st.button("Download Summary PDF"):
        buffer = make_summary_pdf(summary_table, credit_days)
        st.download_button("Download Summary PDF", buffer, file_name="Weighted_Days_Late_Summary.pdf")
    st.divider()
    st.subheader("Company/Ledger Drilldown")
    ledger_list = list(ledgers.keys())
    picked_ledger = st.selectbox("Choose a company/ledger for detailed report", ledger_list)
    if picked_ledger:
        df = ledgers[picked_ledger]
        detailed = analyze_ledger_detailed(df, credit_days)
        st.write(f"Detailed data for **{picked_ledger}**:")
        st.dataframe(detailed)
        if st.button("Download Detailed PDF Report"):
            buffer = make_detailed_pdf(detailed, picked_ledger, credit_days)
            st.download_button("Download Detailed PDF", buffer, file_name=f"{picked_ledger}_Detailed_WDL_Report.pdf")
else:
    st.info("Awaiting CSV file upload.")

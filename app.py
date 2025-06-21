import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from io import BytesIO
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors as rl_colors

DATE_FMT = "%d-%b-%y"

# ----------- ROBUST PARSE LEDGERS -----------
def parse_tally_ledgers(file):
    """Robustly parse Tally group export CSV into ledgers dict."""
    if hasattr(file, "read"):
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
        # Remove BOM and trailing newlines
        line = line.replace("\ufeff", "").rstrip('\n\r')
        cells = [c.strip() for c in line.split(",")]
        # Ledger header
        if cells and cells[0].startswith("Ledger:"):
            if current_ledger and headers and rows:
                df = pd.DataFrame(rows, columns=headers)
                df.columns = [c.strip() for c in df.columns]
                ledgers[current_ledger] = df
                ledger_addresses[current_ledger] = current_address
            current_ledger = cells[1] if len(cells) > 1 else "Unknown"
            current_address = None
            headers = None
            rows = []
            continue
        # Address is right after Ledger header
        if current_ledger and current_address is None and any(cells):
            current_address = cells[0]
            continue
        # Robust header detection (relaxed on blank cols/extra spaces)
        if any("Date" in c for c in cells) and "Debit" in cells and "Credit" in cells:
            headers = [c.strip() for c in cells if c.strip()]
            rows = []
            continue
        # Transaction row (must have header)
        if headers and (len([x for x in cells if x]) >= 5) and not (
            (cells[1] if len(cells) > 1 else "").startswith("Closing Balance")):
            if cells[0] and (cells[0][0].isdigit() or cells[0][0] == '0'):
                while len(cells) < len(headers):
                    cells.append("")
                rows.append([c.strip() for c in cells[:len(headers)]])
    # Last ledger
    if current_ledger and headers and rows:
        df = pd.DataFrame(rows, columns=headers)
        df.columns = [c.strip() for c in df.columns]
        ledgers[current_ledger] = df
        ledger_addresses[current_ledger] = current_address
    return ledgers, ledger_addresses

# ----------- ANALYSIS -----------
def classify_sales_and_payments(df, credit_days=0):
    sales = []
    payments = []
    df.columns = [c.strip() for c in df.columns]  # ensure normalized columns
    df["Parsed_Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    for _, row in df.iterrows():
        vtype = str(row.get("Vch Type", "")).upper()
        particulars = str(row.get("Particulars", "")).upper()
        debit = float(str(row.get("Debit", "")).replace(",", "") or 0)
        credit = float(str(row.get("Credit", "")).replace(",", "") or 0)
        date = row.get("Parsed_Date")
        vch_no = str(row.get("Vch No.", ""))
        if pd.isna(date): continue
        if "OPENING" in particulars or "CLOSING" in particulars or "JOURNAL" in vtype:
            continue
        # SALES: look for "SALES" in vtype or particulars, and credit > 0
        if credit > 0 and ("SALES" in vtype or "SALES" in particulars):
            sales.append({
                "date": date,
                "vch_no": vch_no,
                "amount": credit,
                "due_date": date + timedelta(days=credit_days),
                "remaining": credit,
                "payments": []
            })
        # RECEIPT: look for "RECEIPT" in vtype or particulars, and debit > 0
        elif debit > 0 and ("RECEIPT" in vtype or "RECEIPT" in particulars):
            payments.append({
                "date": date,
                "amount": debit,
                "vch_no": vch_no,
                "particulars": particulars
            })
    return sales, payments

def allocate_fifo(sales, payments):
    sale_idx = 0
    for payment in payments:
        amt_left = payment['amount']
        while amt_left > 0 and sale_idx < len(sales):
            sale = sales[sale_idx]
            to_apply = min(amt_left, sale['remaining'])
            if to_apply > 0:
                sale['payments'].append({'pay_date': payment['date'], 'pay_amt': to_apply})
                sale['remaining'] -= to_apply
                amt_left -= to_apply
            if sale['remaining'] <= 0.01:
                sale['remaining'] = 0
                sale_idx += 1
            elif amt_left == 0:
                break

def weighted_days_late_calc(sales):
    total_impact, total_amount = 0, 0
    per_invoice = []
    for sale in sales:
        total_per_invoice = 0
        for pay in sale['payments']:
            days_late = (pay['pay_date'] - sale['due_date']).days
            total_per_invoice += pay['pay_amt'] * days_late
        weighted_days = total_per_invoice / sale['amount'] if sale['amount'] else 0
        per_invoice.append({
            "Sale_Date": sale['date'],
            "Invoice_No": sale['vch_no'],
            "Sale_Amount": sale['amount'],
            "Weighted_Days_Late": round(weighted_days, 2)
        })
        total_impact += total_per_invoice
        total_amount += sale['amount']
    wdl = round(total_impact / total_amount, 2) if total_amount else np.nan
    return wdl, pd.DataFrame(per_invoice)

# ----------- PDF REPORTS -----------

def make_quarter_table(df_per_invoice):
    if df_per_invoice.empty:
        return pd.DataFrame(columns=["Quarter", "Weighted Days Late"])
    df_per_invoice = df_per_invoice.copy()
    df_per_invoice["Quarter"] = df_per_invoice["Sale_Date"].dt.to_period("Q")
    quarter_results = (
        df_per_invoice.groupby("Quarter")
        .apply(lambda x: (x["Weighted_Days_Late"] * x["Sale_Amount"]).sum() / x["Sale_Amount"].sum() if x["Sale_Amount"].sum() else 0)
        .reset_index(name="Weighted Days Late")
    )
    quarter_results["Quarter"] = quarter_results["Quarter"].astype(str)
    return quarter_results

def plot_wdl_line_chart(quarter_table):
    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(quarter_table["Quarter"], quarter_table["Weighted Days Late"], marker="o", color="#003366")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Weighted Days Late")
    ax.set_title("Weighted Days Late by Quarter")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    img_buf = BytesIO()
    plt.savefig(img_buf, format="png")
    plt.close(fig)
    img_buf.seek(0)
    return img_buf

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

def make_detailed_pdf_full(df_per_invoice, ledger_name, credit_days, company_address=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(LETTER), rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    elements = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('FileTitle', parent=styles['Title'], alignment=1, fontSize=22, spaceAfter=8, leading=26)
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Title'], alignment=1, fontSize=14, spaceAfter=3, leading=18)
    elements.append(Paragraph(f"Detailed Weighted Days Late Report", title_style))
    elements.append(Paragraph(f"Ledger: {ledger_name}", subtitle_style))
    if company_address:
        elements.append(Paragraph(f"Address: {company_address}", styles['Normal']))
    elements.append(Paragraph(f"Credit Period Used: <b>{credit_days} days</b>", styles['Normal']))
    elements.append(Spacer(1, 15))
    quarter_table = make_quarter_table(df_per_invoice)
    if not quarter_table.empty:
        data = [["Quarter", "Weighted Days Late"]]
        data += [[q, f"{wdl:.2f}"] for q, wdl in zip(quarter_table["Quarter"], quarter_table["Weighted Days Late"])]
        tbl = Table(data, colWidths=[120, 180], hAlign='LEFT')
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#003366")),
            ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 13),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor("#f0f0f0")]),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("GRID", (0, 0), (-1, -1), 0.3, rl_colors.gray),
        ]))
        elements.append(Paragraph("Quarter-wise Weighted Days Late:", styles["Heading3"]))
        elements.append(tbl)
        elements.append(Spacer(1, 15))
        img_buf = plot_wdl_line_chart(quarter_table)
        elements.append(Paragraph("Weighted Days Late by Quarter (Chart):", styles["Heading3"]))
        elements.append(Image(img_buf, width=380, height=180))
        elements.append(Spacer(1, 20))
    else:
        elements.append(Paragraph("No sales/payment data for quarter table or chart.", styles["Normal"]))
    elements.append(Paragraph("Per Invoice Weighted Days Late Table:", styles["Heading3"]))
    if not df_per_invoice.empty:
        data = [["Sale Date", "Invoice No", "Sale Amount", "Weighted Days Late"]]
        for _, row in df_per_invoice.iterrows():
            data.append([
                pd.to_datetime(row["Sale_Date"]).strftime("%d-%b-%y") if "Sale_Date" in row else "",
                row.get("Invoice_No", ""),
                f"{row.get('Sale_Amount', 0):,.0f}",
                f"{row.get('Weighted_Days_Late', 0):.2f}",
            ])
        tbl = Table(data, colWidths=[90, 110, 120, 160], hAlign='LEFT')
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#003366")),
            ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 11),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor("#e6e6e6")]),
            ("GRID", (0, 0), (-1, -1), 0.25, rl_colors.gray),
        ]))
        elements.append(tbl)
    else:
        elements.append(Paragraph("No sales/payment data for per-invoice table.", styles["Normal"]))
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ----------- STREAMLIT UI -----------

st.set_page_config(page_title="Tally Ledger Weighted Days Late Analyzer", layout="wide")
st.title("Tally Ledger Weighted Days Late Analyzer")

st.markdown("""
Upload a **full Tally ledger group CSV export**.<br>
- Parses every company/ledger<br>
- Lets you set a credit period (0 = invoice date as due date, N = N days credit)<br>
- Shows weighted days late for each ledger (summary table)<br>
- Lets you select any company/ledger for a detailed drilldown (quarter-wise) with PDF export<br>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload Tally CSV (multi-ledger export)", type=["csv"])
credit_days = st.number_input("Credit Period (days)", min_value=0, value=0, step=1, help="0 means due date is invoice date.")

if uploaded:
    ledgers, ledger_addresses = parse_tally_ledgers(uploaded)
    # DEBUG: show columns for first ledger
    for name, df in ledgers.items():
        st.write(f"Ledger: {name}, Columns: {list(df.columns)}")
        st.write(df.head())
        break
    if not ledgers:
        st.error("No ledgers found in file. Make sure this is a Group Summary CSV export from Tally.")
        st.stop()
    st.success(f"Parsed {len(ledgers)} ledgers from file.")
    summary_rows = []
    ledgerwise_per_invoice = {}
    for ledger_name, df in ledgers.items():
        df.columns = [c.strip() for c in df.columns]
        required_cols = {"Date", "Particulars", "Vch Type", "Vch No.", "Debit", "Credit"}
        if df.empty or not required_cols.issubset(set(df.columns)):
            summary_rows.append({"Ledger": ledger_name, "Weighted Days Late": np.nan})
            ledgerwise_per_invoice[ledger_name] = pd.DataFrame()
            continue
        try:
            sales, payments = classify_sales_and_payments(df, credit_days)
            allocate_fifo(sales, payments)
            wdl, per_inv = weighted_days_late_calc(sales)
            summary_rows.append({"Ledger": ledger_name, "Weighted Days Late": wdl})
            ledgerwise_per_invoice[ledger_name] = per_inv
        except Exception as e:
            summary_rows.append({"Ledger": ledger_name, "Weighted Days Late": np.nan})
            ledgerwise_per_invoice[ledger_name] = pd.DataFrame()
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
        df_per_invoice = ledgerwise_per_invoice[picked_ledger]
        st.write(f"Detailed data for **{picked_ledger}**:")
        st.dataframe(df_per_invoice)
        if st.button("Download Full Detailed PDF Report"):
            pdf_buf = make_detailed_pdf_full(
                df_per_invoice, 
                picked_ledger, 
                credit_days, 
                company_address=ledger_addresses.get(picked_ledger, None)
            )
            st.download_button("Download Detailed PDF (full)", pdf_buf, file_name=f"{picked_ledger}_Full_Detailed_WDL_Report.pdf")
else:
    st.info("Awaiting CSV file upload.")

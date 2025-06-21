import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from io import BytesIO
import tempfile
import matplotlib.pyplot as plt

# --- PDF & Charting Libraries ---
from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# --- CONFIGURATION ---
DATE_FMT = "%d-%b-%y"
QUARTER_MONTHS = {1: "Apr–Jun", 2: "Jul–Sep", 3: "Oct–Dec", 4: "Jan–Mar"}

# --- HELPER & PARSING FUNCTIONS ---
def get_fiscal_quarter_label(dt):
    if pd.isna(dt): return "Invalid Date", None, None
    year, month = dt.year, dt.month
    if 4 <= month <= 6: quarter, fiscal_year = 1, year
    elif 7 <= month <= 9: quarter, fiscal_year = 2, year
    elif 10 <= month <= 12: quarter, fiscal_year = 3, year
    else: quarter, fiscal_year = 4, year - 1
    return f"{fiscal_year} Q{quarter} {QUARTER_MONTHS[quarter]}", fiscal_year, quarter

def format_amount_lakhs(n):
    try:
        n = float(n)
        if abs(n) >= 100000: return f"{n/100000:.2f} L"
        return f"{n:,.0f}"
    except (ValueError, TypeError): return "N/A"

@st.cache_data
def parse_tally_ledgers(file_content):
    ledgers, ledger_addresses = {}, {}
    current_ledger_rows, current_ledger_name, current_address, headers = [], None, None, None
    lines = file_content.splitlines()
    for line in lines:
        line = line.replace("\ufeff", "").strip()
        cells = [cell.strip() for cell in line.split(',')]
        if line.startswith("Ledger:"):
            if current_ledger_name and headers and current_ledger_rows:
                df = pd.DataFrame(current_ledger_rows, columns=headers)
                ledgers[current_ledger_name] = df
                ledger_addresses[current_ledger_name] = current_address
            current_ledger_name = cells[1].strip() if len(cells) > 1 else "Unknown"
            current_address, headers, current_ledger_rows = None, None, []
            continue
        if current_ledger_name and current_address is None and headers is None and any(cells):
            if not any(c in line for c in ["Date", "Particulars", "Debit", "Credit"]):
                 current_address = cells[0]
                 continue
        if "Date" in cells and "Particulars" in cells and "Debit" in cells and "Credit" in cells:
            headers = [h.strip() if h.strip() else f"Unnamed_{i}" for i, h in enumerate(cells)]
            continue
        if headers and len(cells) >= 4 and not (cells[1] if len(cells) > 1 else "").strip().startswith("Closing Balance"):
            while len(cells) < len(headers): cells.append("")
            current_ledger_rows.append(cells)
    if current_ledger_name and headers and current_ledger_rows:
        df = pd.DataFrame(current_ledger_rows, columns=headers)
        ledgers[current_ledger_name] = df
        ledger_addresses[current_ledger_name] = current_address
    return ledgers, ledger_addresses

# --- CORE ANALYSIS LOGIC ---
def classify_sales_and_payments_robust(df, credit_days=0):
    sales, payments = [], []
    df["Parsed_Date"] = pd.to_datetime(df["Date"], format=DATE_FMT, errors="coerce")
    for _, row in df.iterrows():
        if pd.isna(row["Parsed_Date"]): continue
        particulars = str(row.get("Particulars", "")).upper()
        vch_type = str(row.get("Vch Type", "")).upper()
        try: debit_amt = float(str(row.get("Debit", "0")).replace(",", ""))
        except (ValueError, TypeError): debit_amt = 0.0
        try: credit_amt = float(str(row.get("Credit", "0")).replace(",", ""))
        except (ValueError, TypeError): credit_amt = 0.0
        if "OPENING BALANCE" in particulars or "CLOSING BALANCE" in particulars: continue
        if "CREDIT NOTE" in vch_type and credit_amt > 0:
            payments.append({"date": row["Parsed_Date"], "amount": credit_amt, "vch_no": row["Vch No."]})
            continue
        if debit_amt > 0:
            sales.append({"date": row["Parsed_Date"], "vch_no": row["Vch No."], "amount": debit_amt, "due_date": row["Parsed_Date"] + timedelta(days=credit_days), "remaining": debit_amt, "payments": []})
        elif credit_amt > 0:
            payments.append({"date": row["Parsed_Date"], "amount": credit_amt, "vch_no": row["Vch No."]})
    return sales, payments

def allocate_payments_fifo(sales, payments):
    sales.sort(key=lambda x: x['date'])
    payments.sort(key=lambda x: x['date'])
    sale_idx = 0
    for payment in payments:
        payment_remaining = payment['amount']
        while payment_remaining > 0 and sale_idx < len(sales):
            sale = sales[sale_idx]
            amount_to_apply = min(payment_remaining, sale['remaining'])
            if amount_to_apply > 0:
                sale['payments'].append({'pay_date': payment['date'], 'pay_amt': amount_to_apply})
                sale['remaining'] -= amount_to_apply
                payment_remaining -= amount_to_apply
            if sale['remaining'] < 0.01: sale_idx += 1
            if payment_remaining < 0.01: break

@st.cache_data
def analyze_ledger_performance(_df, credit_days):
    sales, payments = classify_sales_and_payments_robust(_df, credit_days)
    if not sales: return 0, pd.DataFrame(), pd.DataFrame()
    allocate_payments_fifo(sales, payments)
    
    total_weighted_impact, total_sale_amount_paid = 0, 0
    invoice_details = []
    for sale in sales:
        if sale['amount'] == 0: continue
        invoice_weighted_impact = sum(payment['pay_amt'] * (payment['pay_date'] - sale['due_date']).days for payment in sale['payments'])
        weighted_days_for_invoice = invoice_weighted_impact / sale['amount'] if sale['amount'] > 0 else 0
        q_label, f_year, f_q = get_fiscal_quarter_label(sale['date'])
        invoice_details.append({
            "Sale_Date_DT": sale['date'], "Sale Date": sale['date'].strftime(DATE_FMT), "Invoice No": sale['vch_no'],
            "Sale Amount": sale['amount'], "Due Date": sale['due_date'].strftime(DATE_FMT),
            "Weighted Days Late": round(weighted_days_for_invoice, 2),
            "Amount Remaining": round(sale['remaining'], 2),
            "Quarter Label": q_label, "Fiscal Year": f_year, "Fiscal Quarter": f_q
        })
        if sale['remaining'] < 0.01:
            total_weighted_impact += invoice_weighted_impact
            total_sale_amount_paid += sale['amount']

    overall_wdl_paid_only = round(total_weighted_impact / total_sale_amount_paid, 2) if total_sale_amount_paid > 0 else 0
    if not invoice_details: return overall_wdl_paid_only, pd.DataFrame(), pd.DataFrame()

    details_df = pd.DataFrame(invoice_details)
    quarterly_summary = details_df.groupby('Quarter Label').apply(
        lambda g: pd.Series({
            'Wtd Avg Days Late': np.average(g['Weighted Days Late'], weights=g['Sale Amount']),
            'Total Sales': g['Sale Amount'].sum(), 'Invoices': len(g)
        })
    ).reset_index()
    
    q_labels_df = pd.DataFrame([get_fiscal_quarter_label(pd.to_datetime(s.split(' ')[0] + '-' + s.split(' ')[2].split('–')[0] + '-01', errors='coerce')) for s in quarterly_summary['Quarter Label']], columns=['label', 'Fiscal Year', 'Fiscal Quarter'])
    quarterly_summary = pd.concat([quarterly_summary, q_labels_df], axis=1)
    quarterly_summary = quarterly_summary.sort_values(['Fiscal Year', 'Fiscal Quarter']).drop(columns=['label', 'Fiscal Year', 'Fiscal Quarter'])

    return overall_wdl_paid_only, details_df, quarterly_summary

# --- PDF GENERATION FUNCTIONS ---
def generate_summary_pdf(summary_data, credit_days):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=LETTER, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    elements = []
    
    elements.append(Paragraph("Overall Summary of Weighted Average Days Late", styles['h1']))
    elements.append(Paragraph(f"Based on a Credit Period of {credit_days} Days", styles['h3']))
    elements.append(Spacer(1, 24))
    
    table_data = [["Company / Ledger", "WADL (Paid Invoices Only)"]]
    for item in summary_data:
        table_data.append([item["Company / Ledger"], f"{item['WADL']:.2f}" if isinstance(item['WADL'], (int, float)) else item['WADL']])
    
    summary_table = Table(table_data, colWidths=[350, 150])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    elements.append(summary_table)
    doc.build(elements)
    buffer.seek(0)
    return buffer

def generate_detailed_pdf(ledger_name, grand_wdl, qtr_df, details_df, credit_days, chart_path):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(LETTER), rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    elements = []

    # --- First Page Header ---
    elements.append(Paragraph(f"Ledger - {ledger_name}", styles['Title']))
    elements.append(Paragraph("Weighted Avg Days Late & Quarterly Sales Report", styles['h2']))
    elements.append(Paragraph(f"By Fiscal Year — {credit_days} Days Credit Period", styles['h3']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Grand Weighted Avg Days Late (Paid Invoices Only): <b>{grand_wdl:.2f}</b>", styles['h2']))
    elements.append(Spacer(1, 12))

    # --- Quarterly Summary Table ---
    qtr_table_data = [["Quarter", "Wtd Avg Days Late", "Total Sales", "Invoices"]]
    for _, row in qtr_df.iterrows():
        qtr_table_data.append([row["Quarter Label"], f"{row['Wtd Avg Days Late']:.2f}", format_amount_lakhs(row['Total Sales']), int(row['Invoices'])])
    
    qtr_summary_table = Table(qtr_table_data, colWidths=[170, 150, 120, 80])
    qtr_summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.darkblue), ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12), ('BACKGROUND', (0,1), (-1,-1), colors.lightblue),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    elements.append(qtr_summary_table)
    elements.append(Spacer(1, 12))

    # --- Chart ---
    if chart_path:
        elements.append(Image(chart_path, width=480, height=240))
        elements.append(Spacer(1, 12))

    # --- Per-Quarter Invoice Details ---
    details_df_sorted = details_df.sort_values(by="Sale_Date_DT")
    for q_label in qtr_df['Quarter Label']:
        q_wdl = qtr_df[qtr_df['Quarter Label'] == q_label]['Wtd Avg Days Late'].iloc[0]
        elements.append(Paragraph(f"{q_label}: Weighted Avg Days Late = {q_wdl:.2f}", styles['h3']))
        
        q_invoices = details_df_sorted[details_df_sorted['Quarter Label'] == q_label]
        invoice_data = [["Sale Date", "Invoice No", "Sale Amount", "Weighted Days Late", "Amount Remaining"]]
        for _, row in q_invoices.iterrows():
            invoice_data.append([row["Sale Date"], row["Invoice No"], f"{row['Sale Amount']:,.2f}", f"{row['Weighted Days Late']:.2f}", f"{row['Amount Remaining']:,.2f}"])
        
        invoice_table = Table(invoice_data, colWidths=[90, 140, 100, 120, 120])
        invoice_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkgreen), ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('ALIGN', (2,1), (-1,-1), 'RIGHT'), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 1, colors.black), ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.lightgrey, colors.white])
        ]))
        elements.append(invoice_table)
        elements.append(Spacer(1, 18))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# --- STREAMLIT UI ---
st.set_page_config(layout="wide")
st.title("📈 Tally Ledger WADL Analyzer")

uploaded_file = st.file_uploader("Upload Tally Multi-Ledger CSV File", type=["csv"])

if uploaded_file:
    file_content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
    ledgers, ledger_addresses = parse_tally_ledgers(file_content)
    
    if not ledgers:
        st.error("No ledgers could be parsed. Please check the file format.")
        st.stop()

    credit_days = st.number_input("Enter Credit Period (days)", min_value=0, value=30, step=1, help="0 means due date is the invoice date.")

    # --- Run Analysis for all ledgers ---
    summary_data, detailed_reports, quarterly_reports = [], {}, {}
    with st.spinner("Analyzing all ledgers..."):
        for name, df in ledgers.items():
            wdl, details_df, qtr_df = analyze_ledger_performance(df, credit_days)
            summary_data.append({"Company / Ledger": name, "WADL": wdl})
            detailed_reports[name] = details_df
            quarterly_reports[name] = qtr_df

    st.divider()
    
    # --- SECTION 1: OVERALL SUMMARY ---
    st.header("Overall Summary of Weighted Average Days Late of All Ledgers")
    st.markdown(f"**Credit Period Set To:** {credit_days} Days")
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    summary_pdf_buffer = generate_summary_pdf(summary_data, credit_days)
    st.download_button(
        label="Download Summary as PDF",
        data=summary_pdf_buffer,
        file_name=f"WADL_Summary_{credit_days}days.pdf",
        mime="application/pdf"
    )

    st.divider()

    # --- SECTION 2: DETAILED ANALYSIS ---
    st.header("In-depth Analysis per Company")
    selected_ledger = st.selectbox("Choose a company for a detailed report:", options=list(ledgers.keys()))

    if selected_ledger:
        grand_wdl = next((item['WADL'] for item in summary_data if item['Company / Ledger'] == selected_ledger), 0)
        details_df = detailed_reports[selected_ledger]
        qtr_df = quarterly_reports[selected_ledger]

        if not details_df.empty:
            # --- Display detailed report on screen ---
            st.subheader(f"Ledger - {selected_ledger}")
            st.markdown(f"**Weighted Avg Days Late & Quarterly Sales Report**")
            st.markdown(f"By Fiscal Year — {credit_days} Days Credit Period")
            st.markdown(f"#### Grand Weighted Avg Days Late (Paid Invoices Only): {grand_wdl:.2f}")

            st.markdown("##### Quarterly Performance Summary")
            qtr_display_df = qtr_df.copy()
            qtr_display_df['Wtd Avg Days Late'] = qtr_display_df['Wtd Avg Days Late'].map('{:,.2f}'.format)
            qtr_display_df['Total Sales'] = qtr_display_df['Total Sales'].apply(format_amount_lakhs)
            st.table(qtr_display_df)

            # --- Generate and display chart ---
            fig, ax = plt.subplots(figsize=(10, 4))
            chart_df = details_df.sort_values(by="Sale_Date_DT")
            ax.plot(chart_df["Sale_Date_DT"], chart_df["Weighted Days Late"], marker='o', linestyle='-', markersize=4)
            ax.set_title(f"WADL Over Time for {selected_ledger}")
            ax.set_xlabel("Sale Date")
            ax.set_ylabel("Weighted Days Late")
            plt.grid(True)
            st.pyplot(fig)
            
            # --- Display per-quarter invoice tables ---
            st.markdown("##### Invoice-by-Invoice Details (Grouped by Quarter)")
            details_df_sorted = details_df.sort_values(by="Sale_Date_DT")
            for q_label in qtr_df['Quarter Label']:
                q_wdl_val = qtr_df[qtr_df['Quarter Label'] == q_label]['Wtd Avg Days Late'].iloc[0]
                with st.expander(f"**{q_label}** (WDL: {q_wdl_val:.2f})"):
                    q_invoices = details_df_sorted[details_df_sorted['Quarter Label'] == q_label]
                    st.dataframe(q_invoices.drop(columns=['Sale_Date_DT', 'Quarter Label', 'Fiscal Year', 'Fiscal Quarter']), use_container_width=True)

            # --- Generate and provide download for detailed PDF ---
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                fig.savefig(tmpfile.name, bbox_inches='tight')
                detailed_pdf_buffer = generate_detailed_pdf(selected_ledger, grand_wdl, qtr_df, details_df, credit_days, tmpfile.name)
            
            st.download_button(
                label=f"Download Detailed Report for {selected_ledger} as PDF",
                data=detailed_pdf_buffer,
                file_name=f"Detailed_Report_{selected_ledger}_{credit_days}days.pdf",
                mime="application/pdf"
            )
        else:
            st.warning("No sales data available to generate a detailed report for this ledger.")
else:
    st.info("Awaiting your Tally CSV file upload to begin analysis.")

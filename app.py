# app.py (FINAL CORRECTED VERSION)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
import tempfile
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CONSTANTS AND HELPER CLASSES ---
DATE_FMT = "%d-%b-%y"
QUARTER_MONTHS = {1: "Apr–Jun", 2: "Jul–Sep", 3: "Oct–Dec", 4: "Jan–Mar"}
MODERN_BLUE_HEX = '#2a3f5f'
LIGHT_GRAY_HEX = '#f0f4f7'

def parse_tally_ledgers(file_content: str) -> (dict, dict):
    # This function is robust and does not need changes.
    ledgers, ledger_addresses = {}, {}
    current_ledger_rows, current_ledger_name, current_address, headers = [], None, None, None
    lines = file_content.splitlines()
    # Find the start of the actual data, skipping any initial summary lines
    data_start_index = -1
    for i, line in enumerate(lines):
        if "Date" in line and "Particulars" in line:
            data_start_index = i
            # Assume the line before the header is the ledger name if it's not found elsewhere
            if i > 0 and not lines[i-1].strip().startswith("Ledger:"):
                 current_ledger_name = lines[i-1].strip().replace("Ledger - ", "")
            break
    
    if data_start_index == -1: return {}, {} # No data header found

    headers = [h.strip() if h.strip() else f"Unnamed_{i}" for i, h in enumerate(lines[data_start_index].split(','))]
    
    for line in lines[data_start_index + 1:]:
        line = line.replace("\ufeff", "").strip()
        if not line or line.startswith(',') or "Closing Balance" in line:
            continue
        cells = [cell.strip() for cell in line.split(',')]
        if len(cells) < len(headers): continue # Skip malformed rows
        current_ledger_rows.append(cells)

    if not current_ledger_name: current_ledger_name = "Premier Cans" # Fallback
    
    if current_ledger_name and headers and current_ledger_rows:
        df = pd.DataFrame(current_ledger_rows, columns=headers)
        ledgers[current_ledger_name] = df
        ledger_addresses[current_ledger_name] = "Address not parsed in this format"

    return ledgers, ledger_addresses

class AnalysisEngine:
    def get_fiscal_quarter_label(self, dt):
        if pd.isna(dt): return "Invalid Date", None, None, None
        year, month = dt.year, dt.month
        fiscal_year_start_month = 4
        if month >= fiscal_year_start_month: fiscal_year = year
        else: fiscal_year = year - 1
        if 4 <= month <= 6: quarter, sort_date = 1, pd.Timestamp(fiscal_year, 4, 1)
        elif 7 <= month <= 9: quarter, sort_date = 2, pd.Timestamp(fiscal_year, 7, 1)
        elif 10 <= month <= 12: quarter, sort_date = 3, pd.Timestamp(fiscal_year, 10, 1)
        else: quarter, sort_date = 4, pd.Timestamp(fiscal_year + 1, 1, 1)
        q_label = f"{fiscal_year} Q{quarter} {QUARTER_MONTHS[quarter]}"
        return q_label, fiscal_year, quarter, sort_date

    def classify_sales_and_payments_robust(self, df, credit_days=0):
        sales, payments = [], []
        # Handle potential empty column names from parsing
        df.columns = [f'col_{i}' if c == '' else c for i, c in enumerate(df.columns)]
        
        df["Parsed_Date"] = pd.to_datetime(df["Date"], format=DATE_FMT, errors="coerce")
        
        for _, row in df.iterrows():
            if pd.isna(row["Parsed_Date"]): continue
            
            # Use .get() to avoid KeyError if a column is missing
            particulars = str(row.get("Particulars", "")).upper()
            # The "Opening Balance" text might be in the third column
            unnamed_col_val = str(row.get(df.columns[2], "")).upper()
            vch_type = str(row.get("Vch Type", "")).upper()
            
            try: debit_amt = float(str(row.get("Debit", "0")).replace(",", ""))
            except (ValueError, TypeError): debit_amt = 0.0
            try: credit_amt = float(str(row.get("Credit", "0")).replace(",", ""))
            except (ValueError, TypeError): credit_amt = 0.0

            # --- FINAL FIX: REMOVED THE LINE THAT IGNORED JOURNAL ENTRIES ---
            # This was the bug. Now Journal entries with credits will be treated as payments.
            
            # More robust check for Opening Balance across multiple columns
            if ("OPENING BALANCE" in particulars or "OPENING BALANCE" in unnamed_col_val) and debit_amt > 0:
                sales.append({
                    "date": row["Parsed_Date"], "vch_no": "Opening Balance", "amount": debit_amt,
                    "due_date": row["Parsed_Date"] + timedelta(days=credit_days),
                    "remaining": debit_amt, "payments": []
                })
                continue
            
            if "CREDIT NOTE" in vch_type and credit_amt > 0:
                payments.append({"date": row["Parsed_Date"], "amount": credit_amt, "vch_no": row["Vch No."]})
                continue
            
            if debit_amt > 0:
                sales.append({
                    "date": row["Parsed_Date"], "vch_no": row["Vch No."], "amount": debit_amt,
                    "due_date": row["Parsed_Date"] + timedelta(days=credit_days),
                    "remaining": debit_amt, "payments": []
                })
            elif credit_amt > 0:
                payments.append({"date": row["Parsed_Date"], "amount": credit_amt, "vch_no": row["Vch No."]})
        return sales, payments

    def allocate_payments_fifo(self, sales, payments):
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

    def run_full_analysis(self, df, credit_days):
        sales, payments = self.classify_sales_and_payments_robust(df, credit_days)
        if not sales: return 0, pd.DataFrame(), pd.DataFrame()
        self.allocate_payments_fifo(sales, payments)
        invoice_details = []
        for sale in sales:
            if sale['amount'] == 0: continue
            invoice_weighted_impact = 0
            for payment in sale['payments']:
                effective_payment_date = max(payment['pay_date'], sale['date'])
                days_late = (effective_payment_date - sale['due_date']).days
                invoice_weighted_impact += payment['pay_amt'] * days_late
            weighted_days_for_invoice = invoice_weighted_impact / sale['amount'] if sale['amount'] > 0 else 0
            q_label, f_year, f_q, q_sort_date = self.get_fiscal_quarter_label(sale['date'])
            invoice_details.append({
                "Sale_Date_DT": sale['date'], "Sale Date": sale['date'].strftime(DATE_FMT), "Invoice No": sale['vch_no'],
                "Sale Amount": sale['amount'], "Due Date": sale['due_date'].strftime(DATE_FMT),
                "Weighted Days Late": round(weighted_days_for_invoice, 1),
                "Amount Remaining": round(sale['remaining'], 2),
                "Quarter Label": q_label, "Fiscal Year": f_year, "Fiscal Quarter": f_q, "Quarter Sort Date": q_sort_date
            })
        if not invoice_details: return 0, pd.DataFrame(), pd.DataFrame()
        details_df = pd.DataFrame(invoice_details)
        total_sale_amount = details_df['Sale Amount'].sum()
        total_weighted_impact = (details_df['Weighted Days Late'] * details_df['Sale Amount']).sum()
        grand_wdl = round(total_weighted_impact / total_sale_amount, 1) if total_sale_amount > 0 else 0
        
        quarterly_summary = details_df.groupby('Quarter Label').apply(
            lambda g: pd.Series({
                'Wtd Avg Days Late': np.average(g['Weighted Days Late'], weights=g['Sale Amount']),
                'Total Sales': g['Sale Amount'].sum(), 'Invoices': len(g),
                'Sort_Date': g['Quarter Sort Date'].iloc[0]
            })
        ).reset_index()
        quarterly_summary = quarterly_summary.sort_values('Sort_Date').drop(columns=['Sort_Date'])
        quarterly_summary.rename(columns={'Quarter Label': 'Quarter'}, inplace=True)
        
        return grand_wdl, details_df, quarterly_summary

class PdfGenerator:
    # This class is correct and does not need changes.
    def __init__(self):
        self.primary_color = HexColor(MODERN_BLUE_HEX)
        self.secondary_color = HexColor(LIGHT_GRAY_HEX)
        self.font_light = colors.whitesmoke
        self.font_dark = colors.darkslategray
        self.grid_color = colors.lightgrey
    def _get_wadl_color(self, wadl_val):
        if not isinstance(wadl_val, (int, float)): return self.font_dark
        if wadl_val <= 30: return colors.green
        elif 30 < wadl_val <= 60: return colors.orange
        else: return colors.red
    def format_amount_lakhs(self, n):
        try:
            n = float(n)
            if abs(n) >= 100000: return f"{n/100000:.2f} L"
            return f"{n:,.0f}"
        except (ValueError, TypeError): return "N/A"
    def generate_detailed_pdf(self, ledger_name, grand_wdl, qtr_df, details_df, credit_days, chart_path):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=landscape(LETTER), rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='CenterTitle', parent=styles['Title'], alignment=TA_CENTER))
        styles.add(ParagraphStyle(name='CenterH2', parent=styles['h2'], alignment=TA_CENTER))
        styles.add(ParagraphStyle(name='CenterH3', parent=styles['h3'], alignment=TA_CENTER, textColor=self.font_dark))
        styles.add(ParagraphStyle(name='LeftH3', parent=styles['h3'], alignment=TA_LEFT, fontName='Helvetica-Bold', textColor=self.font_dark))
        elements = []
        elements.append(Paragraph(f"Ledger - {ledger_name}", styles['CenterTitle']))
        elements.append(Paragraph("Weighted Avg Days Late & Quarterly Sales Report", styles['CenterH2']))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(f"By Fiscal Year — {credit_days} Days Credit Period", styles['CenterH3']))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Grand Weighted Avg Days Late: <b>{grand_wdl:.1f}</b>", styles['CenterH2']))
        elements.append(Spacer(1, 24))
        qtr_table_data = [["Quarter", "Wtd Avg Days Late", "Total Sales", "Invoices"]]
        qtr_table_styles = [
            ('BACKGROUND', (0,0), (-1,0), self.primary_color), ('TEXTCOLOR',(0,0),(-1,0), self.font_light),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0), (-1,0), 11),
            ('BOTTOMPADDING', (0,0), (-1,0), 10), ('TOPPADDING', (0,0), (-1,0), 10),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, self.secondary_color]), ('GRID', (0,0), (-1,-1), 1, self.grid_color)
        ]
        for i, row in qtr_df.iterrows():
            row_index = i + 1
            wadl_val = row['Wtd Avg Days Late']
            qtr_table_data.append([row["Quarter"], f"{wadl_val:.1f}", self.format_amount_lakhs(row['Total Sales']), int(row['Invoices'])])
            wadl_color = self._get_wadl_color(wadl_val)
            qtr_table_styles.append(('TEXTCOLOR', (1, row_index), (1, row_index), wadl_color))
        qtr_summary_table = Table(qtr_table_data, colWidths=[170, 150, 120, 80], hAlign='CENTER')
        qtr_summary_table.setStyle(TableStyle(qtr_table_styles))
        elements.append(qtr_summary_table)
        elements.append(Spacer(1, 24))
        if chart_path:
            elements.append(Image(chart_path, width=540, height=270, hAlign='CENTER'))
            elements.append(Spacer(1, 24))
        details_df_sorted = details_df.sort_values(by="Sale_Date_DT")
        for q_label in qtr_df['Quarter']:
            q_wdl = qtr_df[qtr_df['Quarter'] == q_label]['Wtd Avg Days Late'].iloc[0]
            elements.append(Paragraph(f"{q_label}: Weighted Avg Days Late = {q_wdl:.1f}", styles['LeftH3']))
            q_invoices = details_df_sorted[details_df_sorted['Quarter Label'] == q_label]
            invoice_data = [["Sale Date", "Invoice No", "Sale Amount", "Wtd Days Late", "Amount Remaining"]]
            for _, row in q_invoices.iterrows():
                invoice_data.append([row["Sale Date"], row["Invoice No"], f"{row['Sale Amount']:,.2f}", f"{row['Weighted Days Late']:.1f}", f"{row['Amount Remaining']:,.2f}"])
            invoice_table = Table(invoice_data, colWidths=[90, 140, 110, 110, 120])
            invoice_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), self.primary_color), ('TEXTCOLOR',(0,0),(-1,0), self.font_light),
                ('ALIGN', (0,0), (1, -1), 'LEFT'), ('ALIGN', (2,0), (-1,-1), 'RIGHT'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('GRID', (0,0), (-1,-1), 1, self.grid_color), ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, self.secondary_color])
            ]))
            elements.append(invoice_table)
            elements.append(Spacer(1, 18))
        doc.build(elements)
        buffer.seek(0)
        return buffer

# --- STREAMLIT CACHED FUNCTIONS ---
@st.cache_data
def run_analysis_for_all(_file_content, credit_days):
    ledgers, _ = parse_tally_ledgers(_file_content)
    if not ledgers:
        return pd.DataFrame(), {}, {}
    analyzer = AnalysisEngine()
    summary_data, detailed_reports, quarterly_reports = [], {}, {}
    for name, df in ledgers.items():
        wdl, details_df, qtr_df = analyzer.run_full_analysis(df, credit_days)
        summary_data.append({"Company / Ledger": name, "WADL": wdl})
        detailed_reports[name] = details_df
        quarterly_reports[name] = qtr_df
    summary_df = pd.DataFrame(summary_data)
    return summary_df, detailed_reports, quarterly_reports

@st.cache_data
def generate_pdf_base64(_file_content, credit_days, ledger_name):
    summary_df, detailed_reports, quarterly_reports = run_analysis_for_all(_file_content, credit_days)
    details_df = detailed_reports.get(ledger_name)
    qtr_df = quarterly_reports.get(ledger_name)
    if details_df is None or qtr_df is None or summary_df[summary_df['Company / Ledger'] == ledger_name].empty:
        return ""
    grand_wdl = summary_df[summary_df['Company / Ledger'] == ledger_name]['WADL'].iloc[0]
    pdf_creator = PdfGenerator()
    chart_path = ""
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        chart_df = details_df.sort_values(by="Sale_Date_DT")
        ax.plot(chart_df["Sale_Date_DT"], chart_df["Weighted Days Late"], marker='o', linestyle='-', markersize=4)
        plt.grid(True, linestyle='--', alpha=0.6)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            fig.savefig(tmpfile.name, bbox_inches='tight', dpi=300)
            chart_path = tmpfile.name
        plt.close(fig)
        pdf_buffer = pdf_creator.generate_detailed_pdf(ledger_name, grand_wdl, qtr_df, details_df, credit_days, chart_path)
        pdf_base64 = base64.b64encode(pdf_buffer.read()).decode('utf-8')
        return pdf_base64
    finally:
        if chart_path and os.path.exists(chart_path):
            os.remove(chart_path)

# --- STREAMLIT UI ---
st.set_page_config(layout="wide")
st.title("📊 Tally Ledger Analysis Engine")
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("Upload Tally Ledger CSV", type="csv")
credit_days = st.sidebar.number_input("Credit Days", min_value=0, value=0, step=1)

if uploaded_file is not None:
    file_content = uploaded_file.getvalue().decode("utf-8")
    summary_df, detailed_reports, quarterly_reports = run_analysis_for_all(file_content, credit_days)
    if summary_df.empty:
        st.warning("No ledgers found or data could not be parsed from the uploaded file.")
    else:
        st.header("Overall Ledger Summary")
        st.dataframe(summary_df.style.format({"WADL": "{:.1f}"}))
        st.divider()
        st.header("Detailed Ledger View")
        selected_ledger = st.selectbox("Select a Ledger to View Details", options=summary_df["Company / Ledger"].tolist())
        if selected_ledger:
            details_df = detailed_reports[selected_ledger]
            qtr_df = quarterly_reports[selected_ledger]
            grand_wdl = summary_df[summary_df['Company / Ledger'] == selected_ledger]['WADL'].iloc[0]
            st.metric(label=f"Grand Weighted Avg Days Late for {selected_ledger}", value=f"{grand_wdl:.1f} days")
            st.subheader("Quarterly Summary")
            st.dataframe(qtr_df.style.format({'Wtd Avg Days Late': '{:.1f}', 'Total Sales': '{:,.2f}', 'Invoices': '{:,.0f}'}))
            st.subheader("Invoice Details")
            st.dataframe(details_df[["Sale Date", "Invoice No", "Sale Amount", "Weighted Days Late", "Amount Remaining"]].style.format({'Sale Amount': '{:,.2f}', 'Weighted Days Late': '{:.1f}', 'Amount Remaining': '{:,.2f}'}))
            st.subheader("Download Report")
            pdf_base64 = generate_pdf_base64(file_content, credit_days, selected_ledger)
            if pdf_base64:
                st.download_button(label="📥 Download Detailed PDF Report", data=base64.b64decode(pdf_base64), file_name=f"{selected_ledger}_Report_{credit_days}days.pdf", mime="application/octet-stream")
            else:
                st.error("Could not generate PDF.")
else:
    st.info("Please upload a Tally ledger CSV file and set the credit days to begin analysis.")

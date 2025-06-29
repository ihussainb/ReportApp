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
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
from reportlab.lib import colors
from reportlab.lib.colors import HexColor

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
DATE_FMT = "%d-%b-%y"
QUARTER_MONTHS = {1: "Apr–Jun", 2: "Jul–Sep", 3: "Oct–Dec", 4: "Jan–Mar"}
MODERN_BLUE_HEX = '#2a3f5f'
LIGHT_GRAY_HEX = '#f0f4f7'

# ==============================================================================
# --- DATA PARSING ---
# ==============================================================================
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

# ==============================================================================
# --- ANALYSIS ENGINE ---
# ==============================================================================
class AnalysisEngine:
    def get_fiscal_quarter_label(self, dt):
        if pd.isna(dt): return "Invalid Date", None, None, None
        year, month = dt.year, dt.month
        if 4 <= month <= 6: quarter, fiscal_year, sort_date = 1, year, pd.Timestamp(year, 4, 1)
        elif 7 <= month <= 9: quarter, fiscal_year, sort_date = 2, year, pd.Timestamp(year, 7, 1)
        elif 10 <= month <= 12: quarter, fiscal_year, sort_date = 3, year, pd.Timestamp(year, 10, 1)
        else: quarter, fiscal_year, sort_date = 4, year - 1, pd.Timestamp(year, 1, 1)
        return f"{fiscal_year} Q{quarter} {QUARTER_MONTHS[quarter]}", fiscal_year, quarter, sort_date

    def classify_sales_and_payments_robust(self, df, credit_days=0):
        sales, payments = [], []
        df["Parsed_Date"] = pd.to_datetime(df["Date"], format=DATE_FMT, errors="coerce")
        for _, row in df.iterrows():
            if pd.isna(row["Parsed_Date"]): continue
            particulars = str(row.get("Particulars", "")).upper()
            vch_type = str(row.get("Vch Type", "")).upper()
            if "JOURNAL" in vch_type: continue
            try: debit_amt = float(str(row.get("Debit", "0")).replace(",", ""))
            except (ValueError, TypeError): debit_amt = 0.0
            try: credit_amt = float(str(row.get("Credit", "0")).replace(",", ""))
            except (ValueError, TypeError): credit_amt = 0.0
            if "OPENING BALANCE" in particulars and debit_amt > 0:
                sales.append({"date": row["Parsed_Date"], "vch_no": "Opening Balance", "amount": debit_amt, "due_date": row["Parsed_Date"], "remaining": debit_amt, "payments": []})
                continue
            if "CLOSING BALANCE" in particulars: continue
            if "CREDIT NOTE" in vch_type and credit_amt > 0:
                payments.append({"date": row["Parsed_Date"], "amount": credit_amt, "vch_no": row["Vch No."]})
                continue
            if debit_amt > 0:
                sales.append({"date": row["Parsed_Date"], "vch_no": row["Vch No."], "amount": debit_amt, "due_date": row["Parsed_Date"] + timedelta(days=credit_days), "remaining": debit_amt, "payments": []})
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
                "Weighted Days Late": round(weighted_days_for_invoice, 1), # FORMATTING FIX
                "Amount Remaining": round(sale['remaining'], 2),
                "Quarter Label": q_label, "Fiscal Year": f_year, "Fiscal Quarter": f_q, "Quarter Sort Date": q_sort_date
            })
        if not invoice_details: return 0, pd.DataFrame(), pd.DataFrame()
        details_df = pd.DataFrame(invoice_details)
        total_sale_amount = details_df['Sale Amount'].sum()
        total_weighted_impact = (details_df['Weighted Days Late'] * details_df['Sale Amount']).sum()
        grand_wdl = round(total_weighted_impact / total_sale_amount, 1) if total_sale_amount > 0 else 0 # FORMATTING FIX
        quarterly_summary = details_df.groupby('Quarter Label').apply(
            lambda g: pd.Series({
                'Wtd Avg Days Late': np.average(g['Weighted Days Late'], weights=g['Sale Amount']),
                'Total Sales': g['Sale Amount'].sum(), 'Invoices': len(g),
                'Sort_Date': g['Quarter Sort Date'].iloc[0]
            })
        ).reset_index()
        quarterly_summary = quarterly_summary.sort_values('Sort_Date').drop(columns=['Sort_Date'])
        return grand_wdl, details_df, quarterly_summary

# ==============================================================================
# --- PDF GENERATION ---
# ==============================================================================
class PdfGenerator:
    def __init__(self):
        self.primary_color = HexColor(MODERN_BLUE_HEX)
        self.secondary_color = HexColor(LIGHT_GRAY_HEX)
        self.font_light = colors.whitesmoke
        self.font_dark = colors.darkslategray
        self.grid_color = colors.lightgrey

    def _get_wadl_color(self, wadl_val):
        if not isinstance(wadl_val, (int, float)):
            return self.font_dark
        if wadl_val <= 30: return colors.green
        elif 30 < wadl_val <= 60: return colors.orange
        else: return colors.red

    def format_amount_lakhs(self, n):
        try:
            n = float(n)
            if abs(n) >= 100000: return f"{n/100000:.2f} L"
            return f"{n:,.0f}"
        except (ValueError, TypeError): return "N/A"

    def generate_summary_pdf(self, summary_data, credit_days):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=LETTER, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='CenterH1', parent=styles['h1'], alignment=TA_CENTER))
        styles.add(ParagraphStyle(name='CenterH3', parent=styles['h3'], alignment=TA_CENTER, textColor=self.font_dark))
        elements = []
        elements.append(Paragraph("Overall Summary of Weighted Average Days Late", styles['CenterH1']))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(f"Based on a Credit Period of {credit_days} Days", styles['CenterH3']))
        elements.append(Spacer(1, 24))
        table_data = [["Company / Ledger", "Grand WADL (All Invoices)"]]
        table_styles = [
            ('BACKGROUND', (0,0), (-1,0), self.primary_color), ('TEXTCOLOR',(0,0),(-1,0), self.font_light),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0), (-1,0), 12),
            ('BOTTOMPADDING', (0,0), (-1,0), 12), ('TOPPADDING', (0,0), (-1,0), 12),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, self.secondary_color]), ('GRID', (0,0), (-1,-1), 1, self.grid_color)
        ]
        for i, item in enumerate(summary_data):
            row_index = i + 1
            wadl_val = item['WADL']
            wadl_text = f"{wadl_val:.1f}" if isinstance(wadl_val, (int, float)) else wadl_val # FORMATTING FIX
            table_data.append([item["Company / Ledger"], wadl_text])
            wadl_color = self._get_wadl_color(wadl_val)
            table_styles.append(('TEXTCOLOR', (1, row_index), (1, row_index), wadl_color))
        summary_table = Table(table_data, colWidths=[350, 150], hAlign='CENTER')
        summary_table.setStyle(TableStyle(table_styles))
        elements.append(summary_table)
        doc.build(elements)
        buffer.seek(0)
        return buffer

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
        elements.append(Paragraph(f"Grand Weighted Avg Days Late: <b>{grand_wdl:.1f}</b>", styles['CenterH2'])) # FORMATTING FIX
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
            qtr_table_data.append([row["Quarter Label"], f"{wadl_val:.1f}", self.format_amount_lakhs(row['Total Sales']), int(row['Invoices'])]) # FORMATTING FIX
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
        for q_label in qtr_df['Quarter Label']:
            q_wdl = qtr_df[qtr_df['Quarter Label'] == q_label]['Wtd Avg Days Late'].iloc[0]
            elements.append(Paragraph(f"{q_label}: Weighted Avg Days Late = {q_wdl:.1f}", styles['LeftH3'])) # FORMATTING FIX
            q_invoices = details_df_sorted[details_df_sorted['Quarter Label'] == q_label]
            invoice_data = [["Sale Date", "Invoice No", "Sale Amount", "Wtd Days Late", "Amount Remaining"]]
            for _, row in q_invoices.iterrows():
                invoice_data.append([row["Sale Date"], row["Invoice No"], f"{row['Sale Amount']:,.2f}", f"{row['Weighted Days Late']:.1f}", f"{row['Amount Remaining']:,.2f}"]) # FORMATTING FIX
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

# ==============================================================================
# --- STREAMLIT UI (MAIN APP) ---
# ==============================================================================
def style_wadl(val):
    if not isinstance(val, (int, float)): return ''
    if val <= 30: color = 'green'
    elif 30 < val <= 60: color = 'orange'
    else: color = 'red'
    return f'color: {color}'

def main():
    st.set_page_config(layout="wide")
    st.title("📈 Tally Ledger WADL Analyzer")

    uploaded_file = st.file_uploader("Upload Tally Multi-Ledger CSV File", type=["csv"])

    if uploaded_file:
        if st.button("Clear Cache & Rerun Analysis"):
            st.cache_data.clear()
            st.success("Cache cleared. Re-running analysis with the latest code.")
        
        file_content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
        ledgers, ledger_addresses = parse_tally_ledgers(file_content)
        
        if not ledgers:
            st.error("No ledgers could be parsed. Please check the file format.")
            st.stop()

        credit_days = st.number_input("Enter Credit Period (days)", min_value=0, value=30, step=1, help="0 means due date is the invoice date.")

        analyzer = AnalysisEngine()
        pdf_creator = PdfGenerator()

        summary_data, detailed_reports, quarterly_reports = [], {}, {}
        with st.spinner("Analyzing all ledgers..."):
            for name, df in ledgers.items():
                wdl, details_df, qtr_df = analyzer.run_full_analysis(df, credit_days)
                summary_data.append({"Company / Ledger": name, "WADL": wdl})
                detailed_reports[name] = details_df
                quarterly_reports[name] = qtr_df
        
        summary_df = pd.DataFrame(summary_data)
        numeric_summary_df = summary_df[pd.to_numeric(summary_df['WADL'], errors='coerce').notnull()].copy()
        numeric_summary_df['WADL'] = numeric_summary_df['WADL'].astype(float)

        st.divider()

        st.header("Customer Ranking Dashboard")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ✅ Top 5 Best Paying Customers")
            best_df = numeric_summary_df.sort_values(by="WADL", ascending=True).head(5)
            # FORMATTING FIX
            st.dataframe(best_df.style.applymap(style_wadl, subset=['WADL']).format({'WADL': '{:.1f}'}), use_container_width=True)
        with col2:
            st.markdown("#### 🚨 Top 5 Worst Paying Customers")
            worst_df = numeric_summary_df.sort_values(by="WADL", ascending=False).head(5)
            # FORMATTING FIX
            st.dataframe(worst_df.style.applymap(style_wadl, subset=['WADL']).format({'WADL': '{:.1f}'}), use_container_width=True)

        st.divider()
        
        st.header("Overall Summary of All Ledgers")
        st.markdown(f"**Credit Period Set To:** {credit_days} Days")

        if not numeric_summary_df.empty:
            min_wadl = float(numeric_summary_df['WADL'].min())
            max_wadl = float(numeric_summary_df['WADL'].max())
            wadl_range = st.slider(
                'Filter by WADL Range:',
                min_value=min_wadl,
                max_value=max_wadl,
                value=(min_wadl, max_wadl)
            )
            filtered_df = summary_df[
                (pd.to_numeric(summary_df['WADL'], errors='coerce') >= wadl_range[0]) &
                (pd.to_numeric(summary_df['WADL'], errors='coerce') <= wadl_range[1])
            ]
            # FORMATTING FIX
            st.dataframe(filtered_df.style.applymap(style_wadl, subset=['WADL']).format({'WADL': '{:.1f}'}), use_container_width=True)
        else:
            st.dataframe(summary_df)

        summary_pdf_buffer = pdf_creator.generate_summary_pdf(summary_df.to_dict('records'), credit_days)
        st.download_button(
            label="Download Summary as PDF", data=summary_pdf_buffer,
            file_name=f"WADL_Summary_{credit_days}days.pdf", mime="application/pdf"
        )

        st.divider()

        st.header("In-depth Analysis per Company")
        all_ledgers = list(ledgers.keys())
        selected_ledgers = st.multiselect("Search and select one or more companies for a detailed report:", options=all_ledgers)

        for selected_ledger in selected_ledgers:
            grand_wdl = next((item['WADL'] for item in summary_data if item['Company / Ledger'] == selected_ledger), 0)
            details_df = detailed_reports[selected_ledger]
            qtr_df = quarterly_reports[selected_ledger]

            if not details_df.empty:
                with st.container():
                    st.subheader(f"Detailed Report for: {selected_ledger}")
                    # FORMATTING FIX
                    st.markdown(f"**Grand Weighted Avg Days Late: {grand_wdl:.1f}**")
                    
                    st.markdown("##### Quarterly Performance Summary")
                    qtr_display_df = qtr_df.copy()
                    # FORMATTING FIX
                    qtr_display_df['Wtd Avg Days Late'] = qtr_display_df['Wtd Avg Days Late'].map('{:,.1f}'.format)
                    qtr_display_df['Total Sales'] = qtr_display_df['Total Sales'].apply(pdf_creator.format_amount_lakhs)
                    st.table(qtr_display_df)
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    chart_df = details_df.sort_values(by="Sale_Date_DT")
                    ax.plot(chart_df["Sale_Date_DT"], chart_df["Weighted Days Late"], marker='o', linestyle='-', markersize=4, color=MODERN_BLUE_HEX)
                    ax.set_title(f"WADL Over Time for {selected_ledger}")
                    ax.set_xlabel("Sale Date")
                    ax.set_ylabel("Weighted Days Late")
                    plt.grid(True, linestyle='--', alpha=0.6)
                    st.pyplot(fig)
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                        fig.savefig(tmpfile.name, bbox_inches='tight', dpi=300)
                        detailed_pdf_buffer = pdf_creator.generate_detailed_pdf(selected_ledger, grand_wdl, qtr_df, details_df, credit_days, tmpfile.name)
                    st.download_button(
                        label=f"Download Detailed Report for {selected_ledger} as PDF",
                        data=detailed_pdf_buffer,
                        file_name=f"Detailed_Report_{selected_ledger}_{credit_days}days.pdf",
                        mime="application/pdf",
                        key=f"pdf_{selected_ledger}"
                    )
                    st.markdown("---")
            else:
                st.warning(f"No sales data available to generate a detailed report for {selected_ledger}.")
    else:
        st.info("Awaiting your Tally CSV file upload to begin analysis.")

if __name__ == "__main__":
    main()

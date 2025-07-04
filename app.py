# engine/main.py (DEFINITIVE FINAL VERSION - ALL BUGS FIXED)
import sys
import json
import pandas as pd
import numpy as np
from datetime import timedelta
from io import BytesIO
import base64
import traceback
from typing import List, Dict, Any
import matplotlib
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

try:
    if hasattr(sys, '_MEIPASS'):
        os.environ['MPLCONFIGDIR'] = sys._MEIPASS
except Exception:
    pass

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
import tempfile

# --- YOUR ANALYSIS AND PDF CLASSES ---
DATE_FMT = "%d-%b-%y"
QUARTER_MONTHS = {1: "Apr–Jun", 2: "Jul–Sep", 3: "Oct–Dec", 4: "Jan–Mar"}
MODERN_BLUE_HEX = '#2a3f5f'
LIGHT_GRAY_HEX = '#f0f4f7'

def parse_tally_ledgers(file_content: str) -> (Dict[str, pd.DataFrame], Dict[str, str]):
    # --- NEW, ROBUST, HANG-PROOF MULTI-LEDGER PARSER ---
    ledgers, ledger_addresses = {}, {}
    # Split the entire file content by the "Ledger:" delimiter. This is more robust than a line-by-line state machine.
    # The first split part will be anything before the first ledger, so we skip it.
    ledger_texts = file_content.split("Ledger:")[1:]

    if not ledger_texts:
        # Handle case where "Ledger:" is not present (e.g., premier cans.csv)
        lines = file_content.splitlines()
        ledger_name = "Unknown Ledger"
        # Try to find the name at the end
        for line in reversed(lines):
            if "Ledger -" in line:
                ledger_name = line.split("Ledger -")[1].strip()
                break
        
        headers = None
        data_rows = []
        for line in lines:
            line = line.replace("\ufeff", "").strip()
            if not line: continue
            cells = [cell.strip() for cell in line.split(',')]
            if "Date" in line and "Particulars" in line:
                headers = [h.strip() if h.strip() else f"Unnamed_{i}" for i, h in enumerate(cells)]
                continue
            if headers and len(cells) == len(headers) and cells[0] and (cells[0][0].isdigit() or "Opening Balance" in line):
                data_rows.append(cells)
        if data_rows and headers:
            df = pd.DataFrame(data_rows, columns=headers)
            ledgers[ledger_name] = df
            ledger_addresses[ledger_name] = "Not Parsed"
        return ledgers, ledger_addresses

    # Process each ledger block
    for text_block in ledger_texts:
        lines = text_block.strip().splitlines()
        if not lines: continue

        ledger_name = lines[0].split(',')[0].strip()
        address = None
        headers = None
        data_rows = []

        for i, line in enumerate(lines[1:]): # Start from the line after the name
            cells = [cell.strip() for cell in line.split(',')]
            if "Date" in line and "Particulars" in line:
                headers = [h.strip() if h.strip() else f"Unnamed_{i}" for i, h in enumerate(cells)]
                continue
            
            # Capture address if it's before the header
            if headers is None and len(cells) > 0:
                address = cells[0]
                continue

            if headers:
                if len(cells) == len(headers) and cells[0] and (cells[0][0].isdigit() or "Opening Balance" in line):
                    data_rows.append(cells)
        
        if data_rows and headers:
            df = pd.DataFrame(data_rows, columns=headers)
            ledgers[ledger_name] = df
            ledger_addresses[ledger_name] = address

    return ledgers, ledger_addresses

class AnalysisEngine:
    def get_fiscal_quarter_label(self, dt):
        # YOUR ORIGINAL QUARTER LOGIC - UNCHANGED
        if pd.isna(dt): return "Invalid Date", None, None, None
        year, month = dt.year, dt.month
        if 4 <= month <= 6: quarter, fiscal_year, sort_date = 1, year, pd.Timestamp(year, 4, 1)
        elif 7 <= month <= 9: quarter, fiscal_year, sort_date = 2, year, pd.Timestamp(year, 7, 1)
        elif 10 <= month <= 12: quarter, fiscal_year, sort_date = 3, year, pd.Timestamp(year, 10, 1)
        else: quarter, fiscal_year, sort_date = 4, year - 1, pd.Timestamp(year, 1, 1)
        return f"{fiscal_year} Q{quarter} {QUARTER_MONTHS[quarter]}", fiscal_year, quarter, sort_date

    def classify_sales_and_payments_robust(self, df, credit_days=0):
        # --- BOTH BUGS FIXED HERE ---
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
            
            unnamed_col_val = str(row.get(df.columns[2], "")).upper()
            if ("OPENING BALANCE" in particulars or "OPENING BALANCE" in unnamed_col_val) and debit_amt > 0:
                # BUG FIX #1: Applying credit_days to Opening Balance
                sales.append({
                    "date": row["Parsed_Date"], "vch_no": "Opening Balance", "amount": debit_amt,
                    "due_date": row["Parsed_Date"] + timedelta(days=credit_days),
                    "remaining": debit_amt, "payments": []
                })
                continue
            
            if "CLOSING BALANCE" in particulars: continue
            
            # BUG FIX #2: Journal Credits are now processed as payments
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
    # UNCHANGED
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
            wadl_text = f"{wadl_val:.1f}" if isinstance(wadl_val, (int, float)) else wadl_val
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

def _run_analysis_for_all(file_content: str, credit_days: int) -> (pd.DataFrame, Dict, Dict):
    ledgers, _ = parse_tally_ledgers(file_content)
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

def analyze_all_ledgers(file_content: str, credit_days: int) -> Dict[str, Any]:
    summary_df, _, _ = _run_analysis_for_all(file_content, credit_days)
    if summary_df.empty:
        return {"summary": [], "best_5": [], "worst_5": [], "all_ledgers": []}
    all_ledgers = summary_df["Company / Ledger"].tolist()
    numeric_summary_df = summary_df.copy()
    numeric_summary_df['WADL'] = pd.to_numeric(summary_df['WADL'], errors='coerce')
    numeric_summary_df.dropna(subset=['WADL'], inplace=True)
    best_df = numeric_summary_df.sort_values(by="WADL", ascending=True).head(5)
    worst_df = numeric_summary_df.sort_values(by="WADL", ascending=False).head(5)
    return {
        "summary": summary_df.to_dict('records'),
        "best_5": best_df.to_dict('records'),
        "worst_5": worst_df.to_dict('records'),
        "all_ledgers": all_ledgers
    }

def generate_detailed_pdf_base64(file_content: str, credit_days: int, ledger_name: str) -> str:
    summary_df, detailed_reports, quarterly_reports = _run_analysis_for_all(file_content, credit_days)
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

def main():
    for line in sys.stdin:
        try:
            request = json.loads(line)
            command = request.get("command")
            payload = request.get("payload", {})
            
            if command == "analyze_all":
                result = analyze_all_ledgers(**payload)
            elif command == "generate_pdf":
                pdf_base64 = generate_detailed_pdf_base64(**payload)
                result = {"pdf_base64": pdf_base64, "filename": f"{payload.get('ledger_name', 'Report')}_Report.pdf"}
            else:
                result = {"error": f"Unknown command: {command}"}

            response = {"result": result}
            print(json.dumps(response), flush=True)

        except Exception as e:
            error_details = traceback.format_exc()
            error_response = {"error": error_details}
            print(json.dumps(error_response), flush=True)

if __name__ == "__main__":
    main()

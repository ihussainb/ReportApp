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
    # Convert to datetime, coercing errors to NaT (Not a Time)
    dt = pd.to_datetime(df[date_col], format=DATE_FMT, errors='coerce')
    
    # Find where the initial parsing failed
    mask = dt.isna()
    if mask.any():
        # For the failed rows, try a more general parsing method
        dt_alt = pd.to_datetime(df.loc[mask, date_col], errors='coerce')
        dt.loc[mask] = dt_alt
        
    # Identify any rows that still couldn't be parsed
    failed = df[date_col][dt.isna()]
    return dt, failed

def parse_float(s):
    try:
        # Clean and convert a string to a float
        return float(str(s).replace(',', '').strip()) if s and str(s).strip() else 0.0
    except (ValueError, TypeError):
        return 0.0

def get_fiscal_quarter_label(dt):
    month = dt.month
    year = dt.year
    if 4 <= month <= 6:
        quarter = 1
        months = QUARTER_MONTHS[quarter]
    elif 7 <= month <= 9:
        quarter = 2
        months = QUARTER_MONTHS[quarter]
    elif 10 <= month <= 12:
        quarter = 3
        months = QUARTER_MONTHS[quarter]
    else:  # Jan-Mar
        quarter = 4
        year -= 1  # Assign to the previous fiscal year
        months = QUARTER_MONTHS[quarter]
    return f"{year} Q{quarter} {months}", year, quarter

@st.cache_data(show_spinner=False)
def analyze_ledger(df):
    sales = []
    payments = []
    problematic_rows = []

    # --- IMPROVEMENT: Remove blank/summary rows typical in Tally exports ---
    df.dropna(subset=['Date'], inplace=True)
    df = df[df['Date'].astype(str).str.strip() != ''].copy()


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

    if not rows:
        return pd.DataFrame(), 0.0, {}, pd.Series(dtype='float64'), problematic_rows

    df_rows = pd.DataFrame(rows)
    q_labels = df_rows['Sale_Date'].apply(lambda d: get_fiscal_quarter_label(pd.to_datetime(d)))
    df_rows[['Quarter_Label', 'Fiscal_Year', 'Fiscal_Quarter']] = pd.DataFrame(q_labels.tolist(), index=df_rows.index)
    paid = df_rows[df_rows['Amount_Remaining'] < 0.01]

    paid = paid.sort_values(['Fiscal_Year', 'Fiscal_Quarter'])
    qtr_to_avg = {}
    if not paid.empty:
        for _, group in paid.groupby(['Fiscal_Year', 'Fiscal_Quarter', 'Quarter_Label']):
            label = group['Quarter_Label'].iloc[0]
            qtr_avg = np.average(group['Weighted_Days_Late'], weights=group['Sale_Amount'])
            qtr_to_avg[label] = round(qtr_avg, 2)

    quarter_amounts = paid.groupby('Quarter_Label')['Sale_Amount'].sum()
    total_paid = quarter_amounts.sum()
    quarter_weightage = (quarter_amounts / total_paid * 100).round(1) if total_paid > 0 else pd.Series(dtype='float64')

    df_rows['Sale_Date'] = df_rows['Sale_Date'].dt.strftime('%d-%b-%y')
    df_rows['Due_Date'] = df_rows['Due_Date'].dt.strftime('%d-%b-%y')

    grand_weighted = round(total_impact / total_amount, 2) if total_amount else 0.0

    return df_rows, grand_weighted, qtr_to_avg, quarter_weightage, problematic_rows

def add_first_page_elements(elements, report_title, grand_weighted, qtr_to_avg, quarter_weightage):
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('FileTitle', parent=styles['Title'], alignment=1, fontSize=22, spaceAfter=8, leading=26)
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Title'], alignment=1, fontSize=14, spaceAfter=3, leading=18)
    subsubtitle_style = ParagraphStyle('SubSubtitle', parent=styles['Title'], alignment=1, fontSize=13, spaceAfter=10, leading=16)
    grand_style = ParagraphStyle('Grand', parent=styles['Heading2'], alignment=1, fontSize=14, textColor=colors.HexColor("#003366"), leading=18)

    clean_filename = os.path.splitext(os.path.basename(report_title))[0]
    elements.append(Paragraph(f"{clean_filename}", title_style))
    elements.append(Paragraph("Weighted Average Days & Quarterly to Pay Report", subtitle_style))
    elements.append(Paragraph("By Fiscal Year — 30 Days Credit Period", subsubtitle_style))
    elements.append(Paragraph(f"Grand Weighted Avg Days Late: <b>{grand_weighted}</b>", grand_style))
    elements.append(Spacer(1, 12))

    data = [["Quarter", "Weighted Avg Days Late", "% Paid Amount"]]
    for q in qtr_to_avg.keys():
        weight = f"{quarter_weightage.get(q, 0.0):.1f}%" if q in quarter_weightage else ""
        data.append([q, f"{qtr_to_avg[q]:.1f}", weight])
    table = Table(data, colWidths=[220, 170, 110], hAlign='CENTER')
    table.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,0), 13),
        ("ALIGN", (0,0), (-1,-1), "CENTER"), ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#003366")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f0f0f0")]),
        ("GRID", (0,0), (-1,-1), 0.3, colors.gray), ("FONTSIZE", (0,1), (-1,-1), 12),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6), ("TOPPADDING", (0,0), (-1,-1), 6),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 18))

def generate_pdf_report_grouped(df_rows, grand_weighted, qtr_to_avg, quarter_weightage, buffer, report_title, chart_path=None):
    doc = SimpleDocTemplate(buffer, pagesize=landscape(LETTER), rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    elements = []
    add_first_page_elements(elements, report_title, grand_weighted, qtr_to_avg, quarter_weightage)
    if chart_path:
        elements.append(Image(chart_path, width=500, height=250))
        elements.append(Spacer(1, 18))
    styles = getSampleStyleSheet()
    styleQ = ParagraphStyle('QuarterStyle', parent=styles['Heading2'], spaceAfter=4, spaceBefore=12, textColor=colors.HexColor("#003366"))
    unique_quarters = df_rows.drop_duplicates(subset=["Quarter_Label"])[["Fiscal_Year", "Fiscal_Quarter", "Quarter_Label"]].sort_values(["Fiscal_Year", "Fiscal_Quarter"])
    table_header = ["Sale Date", "Invoice No", "Sale Amount", "Weighted Days Late", "Amount Remaining"]
    for _, quarter_row in unique_quarters.iterrows():
        qlabel = quarter_row["Quarter_Label"]
        q_rows = df_rows[df_rows["Quarter_Label"] == qlabel]
        elements.append(Paragraph(f"{qlabel}: Weighted Avg Days Late = {qtr_to_avg.get(qlabel, '')}", styleQ))
        data = [table_header]
        for _, row in q_rows.iterrows():
            data.append([row["Sale_Date"], row["Invoice_No"], f"{row['Sale_Amount']:,.2f}", f"{row['Weighted_Days_Late']:.1f}", f"{row['Amount_Remaining']:,.2f}"])
        t = Table(data, colWidths=[90, 140, 100, 120, 120])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003366")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (2, 1), (4, -1), "RIGHT"), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 11), ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f6f6f6")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#e6e6e6")]),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.gray),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 18))
    doc.build(elements)

st.title("Ledger Weighted Average Days to Pay Report (Fiscal Year Quarterly Grouped)")
st.markdown("""
Upload your Tally ledger file (CSV or Excel). It must have columns: Date, Particulars, Vch Type, Vch No., Debit, Credit.
The app assumes the data headers are on **row 13**, typical for Tally exports.
""")

uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])

if uploaded_file:
    try:
        try:
            df = pd.read_csv(uploaded_file, header=12)
        except Exception:
            df = pd.read_excel(uploaded_file, header=12)

        # --- FIX: Robustly remove 'Unnamed' columns ---
        # The na=False argument prevents errors if a column name is not a string (e.g., a float)
        mask = df.columns.str.contains('^Unnamed', na=False)
        df = df.loc[:, ~mask]

        if "Date" not in df.columns:
            st.error("Date column not found! Please ensure the file is a standard Tally export with headers on row 13.")
            st.stop()
            
        st.success("File uploaded and read successfully.")
        st.write("Preview:", df.head())
        
        df_rows, grand_weighted, qtr_to_avg, quarter_weightage, problematic_rows = analyze_ledger(df)
        
        if not df_rows.empty:
            st.markdown(f"### Grand Weighted Avg Days Late: **{grand_weighted}**")
            st.markdown("#### By Fiscal Year — 30 Days Credit Period")
            summary_data = []
            for q in qtr_to_avg.keys():
                summary_data.append({
                    "Quarter": q,
                    "Weighted Avg Days Late": f"{qtr_to_avg[q]:.1f}",
                    "% Paid Amount": f"{quarter_weightage.get(q, 0.0):.1f}%"
                })
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df)

            unique_quarters = df_rows.drop_duplicates(subset=["Quarter_Label"])[["Fiscal_Year", "Fiscal_Quarter", "Quarter_Label"]].sort_values(["Fiscal_Year", "Fiscal_Quarter"])
            for _, quarter_row in unique_quarters.iterrows():
                qlabel = quarter_row["Quarter_Label"]
                st.markdown(f"### {qlabel}: Weighted Avg Days Late = {qtr_to_avg.get(qlabel, '')}")
                q_rows = df_rows[df_rows['Quarter_Label'] == qlabel]
                st.dataframe(q_rows[["Sale_Date", "Invoice_No", "Sale_Amount", "Weighted_Days_Late", "Amount_Remaining"]])

            df_rows["Sale_Date_dt"] = pd.to_datetime(df_rows["Sale_Date"], format="%d-%b-%y", errors="coerce")
            df_rows = df_rows.dropna(subset=["Sale_Date_dt"])
            df_rows = df_rows.sort_values("Sale_Date_dt")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df_rows["Sale_Date_dt"], df_rows["Weighted_Days_Late"], marker="o", linestyle="-", label="Weighted Days Late")
            ax.set_xlabel("Sale Date")
            ax.set_ylabel("Days Late")
            ax.set_title("Weighted Days Late Over Time")
            ax.legend()
            plt.grid(True)
            st.pyplot(fig, use_container_width=True)
            
            chart_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig.savefig(chart_temp.name, bbox_inches='tight')
            plt.close(fig)
            
            pdf_buffer = BytesIO()
            generate_pdf_report_grouped(df_rows, grand_weighted, qtr_to_avg, quarter_weightage, pdf_buffer, uploaded_file.name, chart_path=chart_temp.name)
            pdf_buffer.seek(0)
            
            st.download_button(label="Download PDF Report", data=pdf_buffer, file_name="WADP_Report.pdf", mime="application/pdf")
        else:
            st.warning("No valid sales data found to analyze.")
            
        if len(problematic_rows) > 0:
            st.markdown("#### ⚠️ The following rows have problematic or missing dates and were skipped:")
            st.dataframe(pd.DataFrame(problematic_rows))
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
        
else:
    st.info("Awaiting CSV or Excel file upload.")

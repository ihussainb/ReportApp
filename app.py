import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from io import StringIO

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

        # FIX #1: Exclude "Opening Balance" from all calculations, per user's target report.
        if "OPENING BALANCE" in particulars or "CLOSING BALANCE" in particulars:
            continue

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

def analyze_ledger_performance(sales):
    total_weighted_impact, total_sale_amount = 0, 0
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
            total_sale_amount += sale['amount']

    overall_wdl = round(total_weighted_impact / total_sale_amount, 2) if total_sale_amount > 0 else 0
    if not invoice_details: return overall_wdl, pd.DataFrame(), pd.DataFrame()

    details_df = pd.DataFrame(invoice_details)
    
    # FIX #2: Calculate quarterly summary based on ALL invoices, not just paid ones.
    # This aligns the logic with the user's target report.
    quarterly_summary = details_df.groupby('Quarter Label').apply(
        lambda g: pd.Series({
            'Wtd Avg Days Late': np.average(g['Weighted Days Late'], weights=g['Sale Amount']),
            'Total Sales': g['Sale Amount'].sum(), 'Invoices': len(g)
        })
    ).reset_index()
    
    q_labels_df = pd.DataFrame([get_fiscal_quarter_label(pd.to_datetime(s.split(' ')[0] + '-' + s.split(' ')[2].split('–')[0] + '-01', errors='coerce')) for s in quarterly_summary['Quarter Label']], columns=['label', 'Fiscal Year', 'Fiscal Quarter'])
    quarterly_summary = pd.concat([quarterly_summary, q_labels_df], axis=1)
    quarterly_summary = quarterly_summary.sort_values(['Fiscal Year', 'Fiscal Quarter']).drop(columns=['label', 'Fiscal Year', 'Fiscal Quarter'])

    return overall_wdl, details_df, quarterly_summary

# --- STREAMLIT UI ---
st.set_page_config(page_title="Tally Ledger Analyzer", layout="wide")
st.title("📈 Tally Ledger Weighted Days Late Analyzer")
st.markdown("Upload a Tally **Group of Accounts** CSV export to analyze payment delays.")

uploaded_file = st.file_uploader("Upload Tally Multi-Ledger CSV File", type=["csv"])
credit_days = st.number_input("Credit Period (in days)", min_value=0, value=30, step=1)

if uploaded_file:
    try: file_content = uploaded_file.getvalue().decode('utf-8')
    except UnicodeDecodeError: file_content = uploaded_file.getvalue().decode('latin-1')

    ledgers, ledger_addresses = parse_tally_ledgers(file_content)
    if not ledgers: st.error("No ledgers found. Please check the file format."); st.stop()

    st.success(f"Successfully parsed **{len(ledgers)}** ledgers.")

    summary_data, detailed_reports, quarterly_reports = [], {}, {}
    with st.spinner("Analyzing all ledgers..."):
        for name, df in ledgers.items():
            if df.empty: continue
            sales, payments = classify_sales_and_payments_robust(df, credit_days)
            if not sales:
                summary_data.append({"Company / Ledger": name, "Overall Weighted Days Late (Paid Only)": "No Sales", "Address": ledger_addresses.get(name, "N/A")})
                continue
            allocate_payments_fifo(sales, payments)
            wdl, details_df, qtr_df = analyze_ledger_performance(sales)
            summary_data.append({"Company / Ledger": name, "Overall Weighted Days Late (Paid Only)": wdl, "Address": ledger_addresses.get(name, "N/A")})
            detailed_reports[name] = details_df
            quarterly_reports[name] = qtr_df

    st.divider()
    st.header("🏢 Overall Company Summary")
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

    st.divider()
    st.header("📊 Detailed Drill-Down Report")
    selected_ledger = st.selectbox("Choose a company for a detailed quarterly report:", options=list(ledgers.keys()))

    if selected_ledger and selected_ledger in detailed_reports:
        st.subheader(f"Report for: {selected_ledger}")
        st.markdown(f"**Address:** {ledger_addresses.get(selected_ledger, 'N/A')}")
        
        qtr_df = quarterly_reports.get(selected_ledger)
        details_df = detailed_reports.get(selected_ledger)

        if qtr_df is not None and not qtr_df.empty:
            st.markdown("#### Quarterly Performance Summary")
            qtr_display_df = qtr_df.copy()
            qtr_display_df['Wtd Avg Days Late'] = qtr_display_df['Wtd Avg Days Late'].map('{:,.2f}'.format)
            qtr_display_df['Total Sales'] = qtr_display_df['Total Sales'].apply(format_amount_lakhs)
            st.table(qtr_display_df)

            st.markdown("#### Invoice-by-Invoice Details (Grouped by Quarter)")
            details_df_sorted = details_df.sort_values(by="Sale_Date_DT")
            
            for q_label in qtr_display_df['Quarter Label']:
                wdl_val = qtr_display_df[qtr_display_df['Quarter Label'] == q_label]['Wtd Avg Days Late'].iloc[0]
                with st.expander(f"**{q_label}** (WDL: {wdl_val})"):
                    q_invoices = details_df_sorted[details_df_sorted['Quarter Label'] == q_label]
                    st.dataframe(q_invoices.drop(columns=['Sale_Date_DT', 'Quarter Label', 'Fiscal Year', 'Fiscal Quarter']), use_container_width=True)
        elif details_df is not None and not details_df.empty:
             st.markdown("#### All Invoice Details")
             st.dataframe(details_df.drop(columns=['Sale_Date_DT']), use_container_width=True)
        else:
            st.info("No sales data available to generate a report for this ledger.")
else:
    st.info("Awaiting your Tally CSV file upload.")

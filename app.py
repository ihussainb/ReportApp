import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta

DATE_FMT = '%d-%b-%y'
CREDIT_PERIOD_DAYS = 30  # Change this if your default credit period is not 30

def robust_parse_dates(df, date_col="Date"):
    dt = pd.to_datetime(df[date_col], format=DATE_FMT, errors='coerce')
    mask = dt.isna()
    if mask.any():
        dt_alt = pd.to_datetime(df.loc[mask, date_col], errors='coerce')
        dt.loc[mask] = dt_alt
    return dt

def parse_float(s):
    try:
        return float(str(s).replace(',', '').strip()) if s and str(s).strip() else 0.0
    except Exception:
        return 0.0

def get_fiscal_quarter_label(dt):
    month = dt.month
    year = dt.year
    if 4 <= month <= 6:
        quarter = 1
        months = "Apr–Jun"
    elif 7 <= month <= 9:
        quarter = 2
        months = "Jul–Sep"
    elif 10 <= month <= 12:
        quarter = 3
        months = "Oct–Dec"
    else:
        quarter = 4
        year -= 1
        months = "Jan–Mar"
    return f"{year} Q{quarter} {months}"

def find_column(cols, default):
    # For messy files, find "Vch Type" even if unnamed columns present
    cols = [c.strip() for c in cols]
    if default in cols:
        return default
    # Tally: after all unnamed columns
    unnamed = [i for i, c in enumerate(cols) if c.startswith("Unnamed")]
    if unnamed:
        idx = unnamed[-1] + 1
        if idx < len(cols):
            return cols[idx]
    # fallback
    return default

@st.cache_data(show_spinner=False)
def analyze_ledger(df, credit_period_days=30):
    # Robustly handle unnamed columns, find key columns
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    colnames = list(df.columns)
    date_col = "Date"
    particulars_col = "Particulars"
    debit_col = "Debit"
    credit_col = "Credit"
    vch_type_col = find_column(colnames, "Vch Type")
    vch_no_col = "Vch No."
    df["Parsed_Date"] = robust_parse_dates(df, date_col)

    sales = []
    payments = []
    problematic_rows = []
    tax_keywords = ['TDS', 'GST', 'TAX CREDIT', 'INCOME TAX', 'ADVANCE TAX', 'IT.', 'TAX']

    for idx, row in df.iterrows():
        vch_type = str(row.get(vch_type_col, '')).strip().upper()
        particulars = str(row.get(particulars_col, '')).strip()
        particulars_upper = particulars.upper()
        vch_no = str(row.get(vch_no_col, '')).strip()
        debit = parse_float(row.get(debit_col, ''))
        credit = parse_float(row.get(credit_col, ''))
        parsed_date = row.get("Parsed_Date")
        if pd.isna(parsed_date):
            problematic_rows.append(row)
            continue
        if particulars.lower() == "opening balance":
            continue  # skip opening balance row in WDL
        elif 'CREDIT NOTE' in vch_type:
            cn_amt = credit if credit > 0 else debit
            if cn_amt > 0:
                payments.append({
                    'date': parsed_date,
                    'amount': cn_amt,
                    'vch_no': vch_no,
                    'particulars': particulars
                })
        elif (any(keyword in particulars_upper for keyword in tax_keywords) or vch_type == "JOURNAL - C25") and credit > 0:
            payments.append({
                'date': parsed_date,
                'amount': credit,
                'vch_no': vch_no,
                'particulars': particulars
            })
        elif debit > 0:
            sales.append({
                'date': parsed_date,
                'vch_no': vch_no,
                'amount': debit,
                'due_date': parsed_date + timedelta(days=credit_period_days),
                'remaining': debit,
                'payments': []
            })
        elif credit > 0 and ("RECEIPT" in vch_type or "RECEIPT" in particulars_upper):
            payments.append({
                'date': parsed_date,
                'amount': credit,
                'vch_no': vch_no,
                'particulars': particulars
            })

    # FIFO allocation
    sale_idx = 0
    for payment in payments:
        amt_left = payment['amount']
        while amt_left > 0 and sale_idx < len(sales):
            sale = sales[sale_idx]
            to_apply = min(amt_left, sale['remaining'])
            if to_apply > 0:
                sale['payments'].append({
                    'pay_date': payment['date'],
                    'pay_amt': to_apply,
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
            'Due_Date': sale['due_date'],
            'Quarter_Label': get_fiscal_quarter_label(sale['date'])
        })
        if abs(remaining) < 1e-2:
            total_impact += total_per_invoice
            total_amount += sale['amount']
    df_rows = pd.DataFrame(rows)
    paid = df_rows[df_rows['Amount_Remaining'] < 0.01]
    summary = paid.groupby("Quarter_Label").apply(
        lambda g: np.average(g["Weighted_Days_Late"], weights=g["Sale_Amount"])
    ).round(2)
    grand_weighted = round(total_impact / total_amount, 2) if total_amount else 0.0
    return df_rows, grand_weighted, summary, problematic_rows

st.title("Ledger Weighted Avg Days Late Report")
uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])
credit_period_days = st.number_input("Credit period (days)", min_value=0, value=30, step=1)

if uploaded_file:
    try:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            df = pd.read_excel(uploaded_file)
        if "Date" not in df.columns:
            st.error("Date column not found in uploaded file!")
            st.stop()
        st.success("File uploaded and read successfully.")
        st.write("Preview:", df.head())
        df_rows, grand_weighted, summary, problematic_rows = analyze_ledger(df, credit_period_days)
        st.markdown(f"### Grand Weighted Avg Days Late: **{grand_weighted}**")
        st.dataframe(summary.reset_index().rename(
            columns={0: "Wtd Avg Days Late", "Quarter_Label": "Quarter"}))
        quarters = df_rows["Quarter_Label"].unique()
        for q in quarters:
            st.markdown(f"### {q}:")
            st.dataframe(df_rows[df_rows["Quarter_Label"] == q][
                ["Sale_Date", "Invoice_No", "Sale_Amount", "Weighted_Days_Late", "Amount_Remaining"]
            ])
        if len(problematic_rows) > 0:
            st.markdown("#### ⚠️ The following rows have problematic or missing dates and were skipped:")
            st.dataframe(pd.DataFrame(problematic_rows))
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Awaiting CSV or Excel file upload.")

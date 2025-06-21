import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta

DATE_FMT = "%d-%b-%y"

# ----------- ROBUST PARSE LEDGERS -----------
def parse_tally_ledgers(file):
    """Robustly parse Tally group export CSV into ledgers dict, correctly handling unnamed columns."""
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
        line = line.replace("\ufeff", "").rstrip('\n\r')
        cells = [c.strip() for c in line.split(",")]
        # Ledger start
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
        # Address line
        if current_ledger and current_address is None and any(cells):
            current_address = cells[0]
            continue
        # Header row - keep ALL columns, name blanks as Unnamed_{i}
        if any("Date" in c for c in cells) and "Debit" in cells and "Credit" in cells:
            headers = [c.strip() if c.strip() else f"Unnamed_{i}" for i, c in enumerate(cells)]
            rows = []
            continue
        # Transaction row
        if headers and (len([x for x in cells if x]) >= 5) and not (
            (cells[1] if len(cells) > 1 else "").startswith("Closing Balance")):
            if cells[0] and (cells[0][0].isdigit() or cells[0][0] == '0'):
                while len(cells) < len(headers):
                    cells.append("")
                rows.append([c.strip() for c in cells[:len(headers)]])
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
    df.columns = [c.strip() for c in df.columns]
    colnames = list(df.columns)
    date_col = "Date"
    particulars_col = "Particulars"
    debit_col = "Debit"
    credit_col = "Credit"
    unnamed_cols = [c for c in colnames if c.startswith("Unnamed")]
    if unnamed_cols:
        last_unnamed_idx = colnames.index(unnamed_cols[-1])
        vch_type_col = colnames[last_unnamed_idx + 1]
    else:
        vch_type_col = "Vch Type"
    vch_no_col = "Vch No."
    if vch_type_col not in colnames:
        vch_type_col = "Vch Type"
    if vch_no_col not in colnames:
        vch_no_col = "Vch No."
    df["Parsed_Date"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    for _, row in df.iterrows():
        vtype = str(row.get(vch_type_col, "")).upper()
        particulars = str(row.get(particulars_col, "")).upper()
        sale_amt = float(str(row.get(debit_col, "")).replace(",", "") or 0)
        pay_amt = float(str(row.get(credit_col, "")).replace(",", "") or 0)
        date = row.get("Parsed_Date")
        vch_no = str(row.get(vch_no_col, ""))
        if pd.isna(date): continue
        if "OPENING" in particulars or "CLOSING" in particulars or "JOURNAL" in vtype:
            continue
        if ("SALES" in vtype or "SALES" in particulars) and sale_amt > 0:
            sales.append({
                "date": date,
                "vch_no": vch_no,
                "amount": sale_amt,
                "due_date": date + timedelta(days=credit_days),
                "remaining": sale_amt,
                "payments": []
            })
        elif ("RECEIPT" in vtype or "RECEIPT" in particulars) and pay_amt > 0:
            payments.append({
                "date": date,
                "amount": pay_amt,
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

# ----------- STREAMLIT UI -----------

st.set_page_config(page_title="Tally Ledger Weighted Days Late Analyzer", layout="wide")
st.title("Tally Ledger Weighted Days Late Analyzer")

st.markdown("""
Upload a **full Tally ledger group CSV export**.<br>
- Handles all unnamed (blank) columns in Tally export.<br>
- Lets you set a credit period (0 = invoice date as due date, N = N days credit)<br>
- Shows weighted days late for each ledger (summary table)<br>
- Lets you select any company/ledger for a detailed drilldown (per-invoice).<br>
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
        required_cols = {"Date", "Particulars", "Debit", "Credit"}
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
    st.divider()
    st.subheader("Company/Ledger Drilldown")
    ledger_list = list(ledgers.keys())
    picked_ledger = st.selectbox("Choose a company/ledger for detailed report", ledger_list)
    if picked_ledger:
        df_per_invoice = ledgerwise_per_invoice[picked_ledger]
        st.write(f"Detailed data for **{picked_ledger}**:")
        st.dataframe(df_per_invoice)
else:
    st.info("Awaiting CSV file upload.")

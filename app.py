import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from io import StringIO

# Set a consistent date format for parsing and display
DATE_FMT = "%d-%b-%y"

# --- ROBUST PARSING LOGIC ---
def parse_tally_ledgers(file_content):
    """
    Parses a Tally multi-ledger export text into a dictionary of DataFrames.

    Args:
        file_content (str): The string content of the uploaded Tally CSV.

    Returns:
        tuple: A tuple containing:
            - dict: A dictionary where keys are ledger names and values are their transaction DataFrames.
            - dict: A dictionary mapping ledger names to their addresses.
    """
    ledgers = {}
    ledger_addresses = {}
    current_ledger_rows = []
    current_ledger_name = None
    current_address = None
    headers = None

    lines = file_content.splitlines()

    for line in lines:
        # Clean up the line by removing extra characters and splitting by comma
        line = line.replace("\ufeff", "").strip()
        cells = [cell.strip() for cell in line.split(',')]

        # Check if the line indicates the start of a new ledger
        if line.startswith("Ledger:"):
            # If we have data for a previous ledger, save it to a DataFrame
            if current_ledger_name and headers and current_ledger_rows:
                df = pd.DataFrame(current_ledger_rows, columns=headers)
                ledgers[current_ledger_name] = df
                ledger_addresses[current_ledger_name] = current_address

            # Reset for the new ledger
            current_ledger_name = cells[1].strip() if len(cells) > 1 else "Unknown Ledger"
            current_address = None
            headers = None
            current_ledger_rows = []
            continue

        # Capture the address line which typically follows the "Ledger:" line
        if current_ledger_name and current_address is None and headers is None and any(cells):
            # Heuristic: The address is the first non-empty line after the ledger name
            if not any(c in line for c in ["Date", "Particulars", "Debit", "Credit"]):
                 current_address = cells[0]
                 continue

        # Identify the header row
        if "Date" in cells and "Particulars" in cells and "Debit" in cells and "Credit" in cells:
            # Clean and store the headers
            headers = [h.strip() if h.strip() else f"Unnamed_{i}" for i, h in enumerate(cells)]
            continue

        # Process transaction rows
        # A valid transaction row must come after headers are found and have enough data
        if headers and len(cells) >= 4 and not cells[1].strip().startswith("Closing Balance"):
            # Ensure the row has the same number of columns as the header
            while len(cells) < len(headers):
                cells.append("")
            current_ledger_rows.append(cells)

    # Save the very last ledger in the file
    if current_ledger_name and headers and current_ledger_rows:
        df = pd.DataFrame(current_ledger_rows, columns=headers)
        ledgers[current_ledger_name] = df
        ledger_addresses[current_ledger_name] = current_address

    return ledgers, ledger_addresses


# --- FINANCIAL ANALYSIS LOGIC ---
def classify_sales_and_payments(df, credit_days=0):
    """Separates transactions into sales and payments."""
    sales = []
    payments = []

    # Ensure required columns exist
    required_cols = {"Date", "Particulars", "Debit", "Credit", "Vch Type", "Vch No."}
    if not required_cols.issubset(df.columns):
        st.warning(f"One of the required columns {required_cols} is missing. Skipping analysis for this ledger.")
        return [], []

    # Coerce date parsing, handling potential errors gracefully
    df["Parsed_Date"] = pd.to_datetime(df["Date"], format=DATE_FMT, errors="coerce")

    for _, row in df.iterrows():
        # Skip rows where date could not be parsed
        if pd.isna(row["Parsed_Date"]):
            continue

        vch_type = str(row.get("Vch Type", "")).upper()
        particulars = str(row.get("Particulars", "")).upper()
        
        # Parse debit and credit, handling commas and empty values
        try:
            debit_amt = float(str(row.get("Debit", "0")).replace(",", ""))
        except (ValueError, TypeError):
            debit_amt = 0.0
            
        try:
            credit_amt = float(str(row.get("Credit", "0")).replace(",", ""))
        except (ValueError, TypeError):
            credit_amt = 0.0

        # Skip opening/closing balances and journals from this analysis
        if "OPENING BALANCE" in particulars or "CLOSING BALANCE" in particulars or "JOURNAL" in vch_type:
            continue

        # Classify as Sale (Debit entry, typically a Sales voucher)
        if ("SALES" in vch_type or "SALES" in particulars) and debit_amt > 0:
            sales.append({
                "date": row["Parsed_Date"],
                "vch_no": row["Vch No."],
                "amount": debit_amt,
                "due_date": row["Parsed_Date"] + timedelta(days=credit_days),
                "remaining": debit_amt,
                "payments": []
            })
        # Classify as Payment (Credit entry, typically a Receipt voucher)
        elif ("RECEIPT" in vch_type or "RECEIPT" in particulars or "CREDIT NOTE" in vch_type) and credit_amt > 0:
            payments.append({
                "date": row["Parsed_Date"],
                "amount": credit_amt,
                "vch_no": row["Vch No."],
            })

    return sales, payments

def allocate_payments_fifo(sales, payments):
    """Allocates payments to sales on a First-In, First-Out basis."""
    # Sort sales and payments by date to ensure correct FIFO allocation
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

            # If the sale is fully paid, move to the next one
            if sale['remaining'] < 0.01: # Use a small epsilon for float comparison
                sale_idx += 1
            
            if payment_remaining < 0.01:
                break

def calculate_weighted_days_late(sales):
    """Calculates the overall weighted days late and provides per-invoice details."""
    total_weighted_impact = 0
    total_sale_amount = 0
    invoice_details = []

    for sale in sales:
        if sale['amount'] == 0:
            continue

        invoice_weighted_impact = 0
        for payment in sale['payments']:
            days_late = (payment['pay_date'] - sale['due_date']).days
            invoice_weighted_impact += payment['pay_amt'] * days_late
        
        # Calculate weighted days late for this specific invoice
        weighted_days_for_invoice = invoice_weighted_impact / sale['amount'] if sale['amount'] > 0 else 0

        invoice_details.append({
            "Sale Date": sale['date'].strftime(DATE_FMT),
            "Invoice No": sale['vch_no'],
            "Sale Amount": sale['amount'],
            "Due Date": sale['due_date'].strftime(DATE_FMT),
            "Weighted Days Late": round(weighted_days_for_invoice, 2),
            "Amount Remaining": round(sale['remaining'], 2)
        })
        
        # Add to the grand total only if the invoice is fully paid
        if sale['remaining'] < 0.01:
            total_weighted_impact += invoice_weighted_impact
            total_sale_amount += sale['amount']

    # Calculate the grand weighted average for all fully paid invoices
    overall_wdl = round(total_weighted_impact / total_sale_amount, 2) if total_sale_amount > 0 else 0
    
    return overall_wdl, pd.DataFrame(invoice_details)


# --- STREAMLIT USER INTERFACE ---

st.set_page_config(page_title="Tally Ledger Analyzer", layout="wide")
st.title("📈 Tally Ledger Weighted Days Late Analyzer")

st.markdown("""
**Instructions:**
1.  Export a **Group of Accounts** from Tally to CSV format.
2.  Upload the CSV file below.
3.  Set the **Credit Period** (e.g., 0 for immediate due date, 30 for 30 days credit).
4.  The app will display a summary for all companies and allow you to drill down for details.
""")

# --- UI Components ---
uploaded_file = st.file_uploader("Upload Tally Multi-Ledger CSV File", type=["csv"])
credit_days = st.number_input("Credit Period (in days)", min_value=0, value=30, step=1, help="Enter the number of credit days. 0 means the invoice date is the due date.")

if uploaded_file:
    try:
        # To handle different file encodings, read as bytes and decode
        file_content = uploaded_file.getvalue().decode('utf-8')
    except UnicodeDecodeError:
        file_content = uploaded_file.getvalue().decode('latin-1')

    ledgers, ledger_addresses = parse_tally_ledgers(file_content)

    if not ledgers:
        st.error("No ledgers could be parsed from the file. Please ensure it is a valid Tally Group Export CSV.")
        st.stop()

    st.success(f"Successfully parsed **{len(ledgers)}** ledgers from the file.")

    summary_data = []
    detailed_reports = {}

    with st.spinner("Analyzing all ledgers... This may take a moment."):
        for name, df in ledgers.items():
            if df.empty:
                continue
            
            sales, payments = classify_sales_and_payments(df, credit_days)
            
            if not sales:
                # Add to summary even if no sales, to show it was processed
                summary_data.append({"Company / Ledger": name, "Weighted Days Late": "No Sales Data", "Address": ledger_addresses.get(name, "N/A")})
                continue

            allocate_payments_fifo(sales, payments)
            wdl, per_invoice_df = calculate_weighted_days_late(sales)
            
            summary_data.append({"Company / Ledger": name, "Weighted Days Late": wdl, "Address": ledger_addresses.get(name, "N/A")})
            detailed_reports[name] = per_invoice_df

    # --- Display Results ---
    st.divider()
    st.header("Summary Report")
    st.markdown("Weighted Average Days Late for all fully paid invoices, per company.")
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)

    st.divider()
    st.header("Detailed Drill-Down")
    
    # Let user select a company from the parsed list
    selected_ledger = st.selectbox("Choose a company to see a detailed invoice-by-invoice report:", options=list(ledgers.keys()))

    if selected_ledger and selected_ledger in detailed_reports:
        st.subheader(f"Detailed Report for: {selected_ledger}")
        st.markdown(f"**Address:** {ledger_addresses.get(selected_ledger, 'N/A')}")
        
        report_df = detailed_reports[selected_ledger]
        if not report_df.empty:
            st.dataframe(report_df, use_container_width=True)
        else:
            st.info("No sales data was found for this ledger to generate a detailed report.")
else:
    st.info("Awaiting your Tally CSV file upload to begin analysis.")

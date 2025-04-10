import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
import time
import datetime

# --- 1. Data Loading and Processing ---

# API Configuration for local deployment
# api_url_base = "YOUR_API_URL_HERE"  # REMOVE THIS LINE!
page_size = 500
all_batches = []

# Account Category Mapping (Critical)
account_category_mapping = {
    '11000001001': 'Assets',
    '61000003001': 'Operating Expenses',
    '61000006011': 'Operating Expenses',
    '12000002007': 'Assets',
    '21000001001': 'Liabilities',
    '61000008001': 'Operating Expenses',
    '13000001001': 'Assets',
    '61000001001': 'Operating Expenses',
    '61000017001': 'Operating Expenses',
    '61000018001': 'Operating Expenses',
    '61000019001': 'Operating Expenses',
    '61000022001': 'Operating Expenses',
    '61000025001': 'Operating Expenses',
    '61000026001': 'Operating Expenses',
    '61000027001': 'Operating Expenses',
    # Add more mappings here!
}

@st.cache_data
def fetch_data_from_api():
    """Fetches all journal batches from the API with pagination and error handling."""
    global all_batches
    skip = 0

    # Get API URL from Streamlit Secrets
    api_url_base = st.secrets["SAGE_API_URL"]  # Access the secret

    try:
        # First request to get the total count
        count_url = f"{api_url_base}?$top=1&$skip=0&$count=true"
        try:
            count_response = requests.get(count_url)
            count_response.raise_for_status()
            total_count = count_response.json().get('@odata.count', 0)
            st.write(f"Total journal batches reported by API: {total_count}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching count: {e}")
            st.stop()  # Stop execution if count fails
        except json.JSONDecodeError as e:
            st.error("Error decoding count JSON")
            st.stop()

        # --- Custom Loading Bar with IE Networks Colors ---
        progress_bar = st.progress(0)
        progress_text = st.empty()
        fetch_complete = False

        while skip < total_count:
            paginated_url = f"{api_url_base}?$top={page_size}&$skip={skip}"
            # st.write(f"Fetching data from: {paginated_url}")  # Removed URL display
            try:
                response = requests.get(paginated_url)
                response.raise_for_status()
                data = response.json().get('value', [])
                if not data:
                    st.warning("No more data returned by the API. Stopping.")
                    break
            except requests.exceptions.RequestException as e:
                st.error(f"Error fetching data: {e}")
                st.stop()
            except json.JSONDecodeError as e:
                st.error("Error decoding data JSON")
                st.stop()
            all_batches.extend(data)
            skip += page_size
            time.sleep(0.1)  # Rate limiting

            # Update progress bar
            progress_percent = min(100, int((skip / total_count) * 100))
            progress_bar.progress(progress_percent)
            progress_text.text(f"Fetching data: {progress_percent}%")

        fetch_complete = True  # Set flag when fetch is complete
        progress_text.text("Data fetch complete!")
        time.sleep(1)  # Give it a moment to show completion
        progress_bar.empty()  # Clear the progress bar

        st.success(f"Successfully fetched {len(all_batches)} journal batches.")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

    return all_batches

@st.cache_data
def process_data(all_batches):
    """Processes the fetched data into a Pandas DataFrame with feature engineering."""
    processed_transactions_ceo = []

    if all_batches is None:
        return None

    with st.spinner("Processing data..."):
        for batch in all_batches:
            batch_number = batch.get('BatchNumber')
            batch_date = batch.get('DateCreated')
            source_ledger = batch.get('SourceLedger')
            batch_description = batch.get('Description')
            status = batch.get('Status')
            posting_sequence = batch.get('PostingSequence')
            number_of_entries = batch.get('NumberOfEntries')

            for header in batch.get('JournalHeaders', []):
                entry_number = header.get('EntryNumber')
                posting_date = header.get('PostingDate')
                fiscal_year = header.get('FiscalYear')
                fiscal_period = header.get('FiscalPeriod')
                header_description = header.get('Description')
                document_date = header.get('DocumentDate')
                source_type = header.get('SourceType')

                for detail in header.get('JournalDetails', []):
                    transaction = {
                        'BatchNumber': batch_number,
                        'BatchDate': batch_date,
                        'SourceLedger': source_ledger,
                        'BatchDescription': batch_description,
                        'Status': status,
                        'PostingSequence': posting_sequence,
                        'NumberOfEntries': number_of_entries,
                        'EntryNumber': entry_number,
                        'PostingDate': posting_date,
                        'FiscalYear': fiscal_year,
                        'FiscalPeriod': fiscal_period,
                        'EntryDescription': header_description,
                        'DocumentDate': document_date,
                        'SourceType': source_type,
                        'TransactionNumber': detail.get('TransactionNumber'),
                        'AccountNumber': detail.get('AccountNumber'),
                        'AccountDescription': detail.get('AcctDescription'),
                        'Amount': detail.get('Amount'),
                        'JournalDate': detail.get('JournalDate'),
                        'TransactionDescription': detail.get('Description'),
                        'Reference': detail.get('Reference'),
                        'HomeCurrency': detail.get('HomeCurrency')
                    }
                    processed_transactions_ceo.append(transaction)

        df_ceo = pd.DataFrame(processed_transactions_ceo)

        # Convert date and numeric fields
        date_cols = ['BatchDate', 'PostingDate', 'JournalDate', 'DocumentDate']
        for col in date_cols:
            if col in df_ceo.columns:
                df_ceo[col] = pd.to_datetime(df_ceo[col], errors='coerce')

        numeric_cols = ['Amount', 'QuantityTotal', 'NumberOfEntries', 'Debits', 'Credits']
        for col in numeric_cols:
            if col in df_ceo.columns:
                df_ceo[col] = pd.to_numeric(df_ceo[col], errors='coerce')

        # --- Feature Engineering for CEO Dashboard ---

        # 1. Transaction Type
        df_ceo['TransactionType'] = df_ceo['Amount'].apply(lambda x: 'Debit' if x > 0 else 'Credit')

        # 2. Time-Based Features (Crucial for trends)
        if 'JournalDate' in df_ceo.columns:
            df_ceo['Year'] = df_ceo['JournalDate'].dt.year
            df_ceo['Month'] = df_ceo['JournalDate'].dt.month
            df_ceo['Quarter'] = df_ceo['JournalDate'].dt.quarter

            # Handle timezone and convert to Period
            if df_ceo['JournalDate'].dtype != 'object':
                df_ceo['JournalDate'] = df_ceo['JournalDate'].dt.tz_localize(None).dt.to_period('M')

            df_ceo['MonthYear'] = df_ceo['JournalDate'].astype(str)

        # 3. Account Category (Requires a mapping - this is critical)
        df_ceo['AccountCategory'] = df_ceo['AccountNumber'].map(account_category_mapping).fillna('Uncategorized')

        st.success("Data processing complete.")
        return df_ceo

@st.cache_data
def calculate_kpis(df):
    """Calculates the KPIs from the processed DataFrame."""
    if df is None:
        return None

    with st.spinner("Calculating KPIs..."):
        # --- 1. Revenue Trends ---
        revenue_accounts = df[df['AccountCategory'] == 'Revenue']
        monthly_revenue = revenue_accounts.groupby('MonthYear')['Amount'].sum().reset_index()
        monthly_revenue = monthly_revenue.rename(columns={'Amount': 'Revenue'})

        # Calculate Revenue Growth Rate
        monthly_revenue['Revenue_Lag'] = monthly_revenue['Revenue'].shift(1)
        monthly_revenue['Revenue_Growth_Rate'] = (
                                                     (monthly_revenue['Revenue'] - monthly_revenue['Revenue_Lag']) /
                                                     monthly_revenue['Revenue_Lag']
                                                 ) * 100
        monthly_revenue['Revenue_Growth_Rate'] = monthly_revenue['Revenue_Growth_Rate'].replace(
            [float('inf'), float('-inf')], 0)

        # --- 2. COGS Trends ---
        cogs_accounts = df[df['AccountCategory'] == 'Cost of Goods Sold']
        monthly_cogs = cogs_accounts.groupby('MonthYear')['Amount'].sum().reset_index()
        monthly_cogs = monthly_cogs.rename(columns={'Amount': 'COGS'})

        # Calculate COGS as a Percentage of Revenue
        monthly_trends = pd.merge(monthly_revenue, monthly_cogs, on='MonthYear', how='outer').fillna(0)
        monthly_trends['COGS_Pct_of_Revenue'] = (
                                                     (monthly_trends['COGS'].abs() / monthly_trends['Revenue'].abs()) *
                                                     100
                                                 ).fillna(0)
        monthly_trends['COGS_Pct_of_Revenue'] = monthly_trends['COGS_Pct_of_Revenue'].replace([float('inf'), float('-inf')], 0)

        # --- 3. Operating Expenses Trends ---
        operating_expense_accounts = df[df['AccountCategory'] == 'Operating Expenses']
        monthly_operating_expenses = operating_expense_accounts.groupby('MonthYear')['Amount'].sum().reset_index()
        monthly_operating_expenses = monthly_operating_expenses.rename(columns={'Amount': 'Operating_Expenses'})

        # Calculate Operating Expenses as a Percentage of Revenue
        monthly_trends = pd.merge(monthly_trends, monthly_operating_expenses, on='MonthYear', how='outer').fillna(0)
        monthly_trends['Operating_Expenses_Pct_of_Revenue'] = (
                                                                 (monthly_trends['Operating_Expenses'].abs() /
                                                                  monthly_trends['Revenue'].abs()) * 100
                                                             ).fillna(0)
        monthly_trends['Operating_Expenses_Pct_of_Revenue'] = monthly_trends['Operating_Expenses_Pct_of_Revenue'].replace(
            [float('inf'), float('-inf')], 0)

        # --- 4. Gross Profit and Gross Profit Margin Trends ---
        # Calculate Gross Profit
        monthly_trends['Gross_Profit'] = monthly_trends['Revenue'] - monthly_trends['COGS']

        # Calculate Gross Profit Margin
        monthly_trends['Gross_Profit_Margin'] = (
                                                     (monthly_trends['Gross_Profit'].abs() /
                                                      monthly_trends['Revenue'].abs()) * 100
                                                 ).fillna(0)
        monthly_trends['Gross_Profit_Margin'] = monthly_trends['Gross_Profit_Margin'].replace([float('inf'), float('-inf')], 0)

        # --- 5. Key Account Balances (Example: Cash) ---
        cash_accounts = df[df['AccountCategory'] == 'Assets']  # Adjust based on your mapping
        monthly_cash = cash_accounts.groupby('MonthYear')['Amount'].sum().reset_index()
        monthly_cash = monthly_cash.rename(columns={'Amount': 'Cash_Balance'})
        monthly_trends = pd.merge(monthly_trends, monthly_cash, on='MonthYear', how='outer').fillna(0)

        # --- 7. Transaction Volume Over Time ---
        monthly_transaction_volume = df.groupby('MonthYear').size().reset_index(name='Transaction_Volume')
        monthly_trends = pd.merge(monthly_trends, monthly_transaction_volume, on='MonthYear', how='outer').fillna(0)

        st.success("KPIs calculated.")
        return monthly_trends

# --- 2. Streamlit App ---

def main():
    # --- IE Networks Logo ---
    st.image(
        "https://drive.google.com/file/d/1pXOacEUJBLBZ9H-vRlGcj4DWtgzGoqwi/view?usp=sharing",
        width=200,
    )  # Replace with your logo URL

    st.title("IE Networks CEO Dashboard")

    # --- Fetch and Process Data ---
    all_batches = fetch_data_from_api()
    if all_batches is None:
        st.error("Failed to fetch data. App cannot continue.")
        return

    df_ceo = process_data(all_batches)
    if df_ceo is None:
        st.error("Failed to process data. App cannot continue.")
        return

    # --- Calculate KPIs ---
    monthly_trends = calculate_kpis(df_ceo)
    if monthly_trends is None:
        st.error("Failed to calculate KPIs. App cannot continue.")
        return

    # --- Display Data and Visualizations ---
    st.dataframe(df_ceo)  # Display the entire DataFrame with scrolling
    # st.data_editor(df_ceo) # If you want to enable editing
    # st.write(df_ceo.shape)  # Display the shape of the DataFrame

    # --- KPI Visualizations ---
    # 1. Revenue Trends
    st.header("Revenue Trends")
    fig_revenue = px.line(monthly_trends, x='MonthYear', y='Revenue', title='Monthly Revenue')
    st.plotly_chart(fig_revenue, use_container_width=True)

    # 2. Revenue Growth Rate
    st.header("Revenue Growth Rate")
    fig_revenue_growth = px.line(
        monthly_trends, x='MonthYear', y='Revenue_Growth_Rate', title='Monthly Revenue Growth Rate'
    )
    st.plotly_chart(fig_revenue_growth, use_container_width=True)

    # 3. Gross Profit Margin
    st.header("Gross Profit Margin")
    fig_gross_profit_margin = px.line(
        monthly_trends, x='MonthYear', y='Gross_Profit_Margin', title='Monthly Gross Profit Margin'
    )
    st.plotly_chart(fig_gross_profit_margin, use_container_width=True)

    # 4. Operating Expenses
    st.header("Operating Expenses")
    fig_operating_expenses = px.line(
        monthly_trends, x='MonthYear', y='Operating_Expenses', title='Monthly Operating Expenses'
    )
    st.plotly_chart(fig_operating_expenses, use_container_width=True)

    # 5. Cash Balance
    st.header("Cash Balance")
    fig_cash_balance = px.line(monthly_trends, x='MonthYear', y='Cash_Balance', title='Monthly Cash Balance')
    st.plotly_chart(fig_cash_balance, use_container_width=True)

    # 6. Transaction Volume
    st.header("Transaction Volume")
    fig_transaction_volume = px.line(
        monthly_trends, x='MonthYear', y='Transaction_Volume', title='Monthly Transaction Volume'
    )
    st.plotly_chart(fig_transaction_volume, use_container_width=True)

    # --- Drill-Down (Example - Revenue by Account Category) ---
    st.header("Revenue by Account Category")
    revenue_category = df_ceo[df_ceo['AccountCategory'] != 'Uncategorized']
    revenue_category_monthly = (
        revenue_category.groupby(['MonthYear', 'AccountCategory'])['Amount'].sum().reset_index()
    )

    # Create a bar chart for revenue by account category over time
    fig_revenue_category = px.bar(
        revenue_category_monthly,
        x='MonthYear',
        y='Amount',
        color='AccountCategory',
        title='Monthly Revenue by Account Category',
        barmode='group',
    )
    st.plotly_chart(fig_revenue_category, use_container_width=True)

    # --- Additional KPIs and Visualizations can be added here ---

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
import time
import datetime
import logging
from typing import Optional, List, Dict
import asyncio
import aiohttp

# --- Configuration ---

# Set up logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
PAGE_SIZE = 500
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2  # seconds
TIMEOUT = 30  # seconds

# Account Category Mapping
ACCOUNT_CATEGORY_MAPPING = {
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
    # Add Revenue and COGS mappings here after obtaining from Sage
    # Example: '51000001001': 'Revenue',
    #          '52000002001': 'Cost of Goods Sold',
}

# --- Utility Functions ---

def setup_secrets() -> Optional[str]:
    """Retrieve API URL from Streamlit secrets or environment variables."""
    try:
        return st.secrets.get("SAGE_API_URL")
    except Exception as e:
        logger.error(f"Failed to load secrets: {e}")
        st.error("Configuration error: API URL not found. Please check secrets configuration.")
        return None

async def fetch_batch(session: aiohttp.ClientSession, url: str, retries: int = RETRY_ATTEMPTS) -> Optional[Dict]:
    """Fetch a single batch from the API with retry logic."""
    for attempt in range(retries):
        try:
            async with session.get(url, timeout=TIMEOUT) as response:
                response.raise_for_status()
                return await response.json()
        except (aiohttp.ClientError, json.JSONDecodeError) as e:
            logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(RETRY_DELAY)
            continue
    logger.error(f"Failed to fetch {url} after {retries} attempts")
    return None

async def fetch_all_batches(api_url_base: str, page_size: int) -> List[Dict]:
    """Async function to fetch all journal batches."""
    all_batches = []
    skip = 0

    async with aiohttp.ClientSession() as session:
        # Get total count
        count_url = f"{api_url_base}?$top=1&$skip=0&$count=true"
        count_data = await fetch_batch(session, count_url)
        if not count_data:
            st.error("Failed to fetch total count from API.")
            return all_batches
        total_count = count_data.get('@odata.count', 0)
        logger.info(f"Total journal batches: {total_count}")

        # Progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()

        while skip < total_count:
            paginated_url = f"{api_url_base}?$top={page_size}&$skip={skip}"
            data = await fetch_batch(session, paginated_url)
            if not data or not data.get('value'):
                logger.warning("No more data returned by API.")
                break
            all_batches.extend(data['value'])
            skip += page_size

            # Update progress
            progress_percent = min(100, int((skip / total_count) * 100))
            progress_bar.progress(progress_percent)
            progress_text.text(f"Fetching data: {progress_percent}%")

        progress_text.text("Data fetch complete!")
        time.sleep(1)
        progress_bar.empty()
        st.success(f"Fetched {len(all_batches)} journal batches.")

    return all_batches

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_data_from_api() -> Optional[List[Dict]]:
    """Fetch all journal batches from the API with async pagination, wrapped for Streamlit caching."""
    api_url_base = setup_secrets()
    if not api_url_base:
        return None

    try:
        # Run async fetching in a synchronous context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        all_batches = loop.run_until_complete(fetch_all_batches(api_url_base, PAGE_SIZE))
        loop.close()
        return all_batches
    except Exception as e:
        logger.error(f"Unexpected error in fetch_data_from_api: {e}")
        st.error("An error occurred while fetching data. Please try again later.")
        return None

@st.cache_data(show_spinner=False, ttl=3600)
def process_data(all_batches: List[Dict]) -> Optional[pd.DataFrame]:
    """Process fetched data into a Pandas DataFrame with feature engineering."""
    if not all_batches:
        logger.error("No data to process")
        return None

    processed_transactions = []
    missing_mappings = set()

    with st.spinner("Processing data..."):
        for batch in all_batches:
            batch_info = {
                'BatchNumber': batch.get('BatchNumber'),
                'BatchDate': batch.get('DateCreated'),
                'SourceLedger': batch.get('SourceLedger'),
                'BatchDescription': batch.get('Description'),
                'Status': batch.get('Status'),
                'PostingSequence': batch.get('PostingSequence'),
                'NumberOfEntries': batch.get('NumberOfEntries'),
            }

            for header in batch.get('JournalHeaders', []):
                header_info = {
                    'EntryNumber': header.get('EntryNumber'),
                    'PostingDate': header.get('PostingDate'),
                    'FiscalYear': header.get('FiscalYear'),
                    'FiscalPeriod': header.get('FiscalPeriod'),
                    'EntryDescription': header.get('Description'),
                    'DocumentDate': header.get('DocumentDate'),
                    'SourceType': header.get('SourceType'),
                }

                for detail in header.get('JournalDetails', []):
                    account_number = detail.get('AccountNumber')
                    account_category = ACCOUNT_CATEGORY_MAPPING.get(account_number, 'Uncategorized')
                    if account_category == 'Uncategorized':
                        missing_mappings.add(account_number)

                    transaction = {
                        **batch_info,
                        **header_info,
                        'TransactionNumber': detail.get('TransactionNumber'),
                        'AccountNumber': account_number,
                        'AccountDescription': detail.get('AcctDescription'),
                        'Amount': detail.get('Amount'),
                        'JournalDate': detail.get('JournalDate'),
                        'TransactionDescription': detail.get('Description'),
                        'Reference': detail.get('Reference'),
                        'HomeCurrency': detail.get('HomeCurrency'),
                        'AccountCategory': account_category,
                    }
                    processed_transactions.append(transaction)

        if missing_mappings:
            logger.warning(f"Missing account mappings for: {missing_mappings}")
            st.warning(f"Some accounts are uncategorized: {len(missing_mappings)} unique accounts. Check app.log for details.")

        df = pd.DataFrame(processed_transactions)

        # Type conversions
        date_cols = ['BatchDate', 'PostingDate', 'JournalDate', 'DocumentDate']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        numeric_cols = ['Amount', 'NumberOfEntries']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Feature engineering
        if 'JournalDate' in df.columns:
            df['Year'] = df['JournalDate'].dt.year
            df['Month'] = df['JournalDate'].dt.month
            df['Quarter'] = df['JournalDate'].dt.quarter
            df['MonthYear'] = df['JournalDate'].dt.to_period('M').astype(str)
            df['TransactionType'] = df['Amount'].apply(lambda x: 'Debit' if pd.notna(x) and x > 0 else 'Credit')

        logger.info("Data processing complete")
        st.success("Data processing complete.")
        return df

@st.cache_data(show_spinner=False, ttl=3600)
def calculate_kpis(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Calculate KPIs from the processed DataFrame."""
    if df.empty:
        logger.error("Empty DataFrame for KPI calculation")
        return None

    with st.spinner("Calculating KPIs..."):
        # Initialize monthly_trends with MonthYear from all transactions
        if 'MonthYear' in df.columns:
            monthly_trends = pd.DataFrame(df['MonthYear'].unique(), columns=['MonthYear'])
        else:
            logger.error("MonthYear column missing in DataFrame")
            st.error("Cannot calculate KPIs: No valid date data.")
            return None

        # Revenue Trends
        revenue_accounts = df[df['AccountCategory'] == 'Revenue']
        if not revenue_accounts.empty:
            monthly_revenue = revenue_accounts.groupby('MonthYear')['Amount'].sum().reset_index(name='Revenue')
            monthly_revenue['Revenue_Lag'] = monthly_revenue['Revenue'].shift(1)
            monthly_revenue['Revenue_Growth_Rate'] = (
                (monthly_revenue['Revenue'] - monthly_revenue['Revenue_Lag']) / monthly_revenue['Revenue_Lag'] * 100
            ).fillna(0).replace([float('inf'), -float('inf')], 0)
            monthly_trends = monthly_trends.merge(monthly_revenue, on='MonthYear', how='outer')
        else:
            logger.warning("No Revenue accounts found")
            monthly_trends['Revenue'] = 0
            monthly_trends['Revenue_Growth_Rate'] = 0

        # COGS Trends
        cogs_accounts = df[df['AccountCategory'] == 'Cost of Goods Sold']
        if not cogs_accounts.empty:
            monthly_cogs = cogs_accounts.groupby('MonthYear')['Amount'].sum().reset_index(name='COGS')
            monthly_trends = monthly_trends.merge(monthly_cogs, on='MonthYear', how='outer')
        else:
            logger.warning("No Cost of Goods Sold accounts found")
            monthly_trends['COGS'] = 0

        # Operating Expenses
        op_exp_accounts = df[df['AccountCategory'] == 'Operating Expenses']
        if not op_exp_accounts.empty:
            monthly_op_exp = op_exp_accounts.groupby('MonthYear')['Amount'].sum().reset_index(name='Operating_Expenses')
            monthly_trends = monthly_trends.merge(monthly_op_exp, on='MonthYear', how='outer')
        else:
            logger.warning("No Operating Expenses accounts found")
            monthly_trends['Operating_Expenses'] = 0

        # Fill NaN values
        monthly_trends = monthly_trends.fillna(0)

        # Calculate derived KPIs
        monthly_trends['COGS_Pct_of_Revenue'] = (
            (monthly_trends['COGS'].abs() / monthly_trends['Revenue'].abs()) * 100
        ).fillna(0).replace([float('inf'), -float('inf')], 0)

        monthly_trends['Operating_Expenses_Pct_of_Revenue'] = (
            (monthly_trends['Operating_Expenses'].abs() / monthly_trends['Revenue'].abs()) * 100
        ).fillna(0).replace([float('inf'), -float('inf')], 0)

        monthly_trends['Gross_Profit'] = monthly_trends['Revenue'] - monthly_trends['COGS']
        monthly_trends['Gross_Profit_Margin'] = (
            (monthly_trends['Gross_Profit'].abs() / monthly_trends['Revenue'].abs()) * 100
        ).fillna(0).replace([float('inf'), -float('inf')], 0)

        # Cash Balance
        cash_accounts = df[df['AccountCategory'] == 'Assets']
        if not cash_accounts.empty:
            monthly_cash = cash_accounts.groupby('MonthYear')['Amount'].sum().reset_index(name='Cash_Balance')
            monthly_trends = monthly_trends.merge(monthly_cash, on='MonthYear', how='outer')
        else:
            logger.warning("No Assets accounts found")
            monthly_trends['Cash_Balance'] = 0

        # Transaction Volume
        monthly_volume = df.groupby('MonthYear').size().reset_index(name='Transaction_Volume')
        monthly_trends = monthly_trends.merge(monthly_volume, on='MonthYear', how='outer').fillna(0)

        logger.info("KPIs calculated")
        st.success("KPIs calculated.")
        return monthly_trends

# --- Streamlit App ---

def render_dashboard(df: pd.DataFrame, monthly_trends: pd.DataFrame):
    """Render the Streamlit dashboard."""
    st.image(
        "https://asset.cloudinary.com/djftqvask/f835addefe28e317faa68714ed30d080",
        width=200,
    )
    st.title("IE Networks CEO Dashboard")

    # Filters
    st.sidebar.header("Filters")
    min_date = df['JournalDate'].min().date() if not pd.isna(df['JournalDate'].min()) else datetime.date.today()
    max_date = df['JournalDate'].max().date() if not pd.isna(df['JournalDate'].max()) else datetime.date.today()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date,
    )
    account_categories = st.sidebar.multiselect(
        "Account Categories", options=df['AccountCategory'].unique(), default=df['AccountCategory'].unique()
    )

    # Filter DataFrame
    filtered_df = df[
        (df['JournalDate'].dt.date >= pd.Timestamp(date_range[0]).date()) &
        (df['JournalDate'].dt.date <= pd.Timestamp(date_range[1]).date()) &
        (df['AccountCategory'].isin(account_categories))
    ]

    # Display Data
    st.subheader("Transaction Data")
    st.dataframe(filtered_df, use_container_width=True)

    # Visualizations
    st.header("Key Performance Indicators")

    # Revenue
    if 'Revenue' in monthly_trends.columns:
        fig_revenue = px.line(monthly_trends, x='MonthYear', y='Revenue', title='Monthly Revenue')
        st.plotly_chart(fig_revenue, use_container_width=True)
    else:
        st.warning("No Revenue data available for visualization.")

    # Revenue Growth
    if 'Revenue_Growth_Rate' in monthly_trends.columns:
        fig_growth = px.line(monthly_trends, x='MonthYear', y='Revenue_Growth_Rate', title='Revenue Growth Rate')
        st.plotly_chart(fig_growth, use_container_width=True)
    else:
        st.warning("No Revenue Growth Rate data available.")

    # Gross Profit Margin
    if 'Gross_Profit_Margin' in monthly_trends.columns:
        fig_gpm = px.line(monthly_trends, x='MonthYear', y='Gross_Profit_Margin', title='Gross Profit Margin')
        st.plotly_chart(fig_gpm, use_container_width=True)
    else:
        st.warning("No Gross Profit Margin data available.")

    # Operating Expenses
    if 'Operating_Expenses' in monthly_trends.columns:
        fig_opex = px.line(monthly_trends, x='MonthYear', y='Operating_Expenses', title='Operating Expenses')
        st.plotly_chart(fig_opex, use_container_width=True)
    else:
        st.warning("No Operating Expenses data available.")

    # Cash Balance
    if 'Cash_Balance' in monthly_trends.columns:
        fig_cash = px.line(monthly_trends, x='MonthYear', y='Cash_Balance', title='Cash Balance')
        st.plotly_chart(fig_cash, use_container_width=True)
    else:
        st.warning("No Cash Balance data available.")

    # Transaction Volume
    if 'Transaction_Volume' in monthly_trends.columns:
        fig_volume = px.line(monthly_trends, x='MonthYear', y='Transaction_Volume', title='Transaction Volume')
        st.plotly_chart(fig_volume, use_container_width=True)
    else:
        st.warning("No Transaction Volume data available.")

    # Revenue by Category
    st.header("Revenue by Account Category")
    revenue_category = filtered_df[filtered_df['AccountCategory'] != 'Uncategorized']
    if not revenue_category.empty:
        revenue_category_monthly = (
            revenue_category.groupby(['MonthYear', 'AccountCategory'])['Amount'].sum().reset_index()
        )
        fig_revenue_cat = px.bar(
            revenue_category_monthly,
            x='MonthYear',
            y='Amount',
            color='AccountCategory',
            title='Monthly Revenue by Account Category',
            barmode='group',
        )
        st.plotly_chart(fig_revenue_cat, use_container_width=True)
    else:
        st.warning("No categorized revenue data available for visualization.")

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="IE Networks CEO Dashboard", layout="wide")

    all_batches = fetch_data_from_api()
    if not all_batches:
        return

    df_ceo = process_data(all_batches)
    if df_ceo is None or df_ceo.empty:
        return

    monthly_trends = calculate_kpis(df_ceo)
    if monthly_trends is None or monthly_trends.empty:
        return

    render_dashboard(df_ceo, monthly_trends)

if __name__ == "__main__":
    main()
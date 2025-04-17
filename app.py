import streamlit as st
import pandas as pd
import plotly.express as px
import json
import time
import datetime
import logging
from typing import Optional, List, Dict
import asyncio
import aiohttp
import os

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

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load unique_accounts.csv
try:
    unique_accounts = pd.read_csv(os.path.join(BASE_DIR, 'unique_accounts.csv'))
    logger.info(f"Loaded unique_accounts.csv successfully with {len(unique_accounts)} accounts")
except FileNotFoundError:
    logger.error("unique_accounts.csv not found")
    st.error("Error: unique_accounts.csv not found. Please ensure the file is included in the app directory.")
    unique_accounts = pd.DataFrame(columns=['AccountNumber', 'AccountDescription'])
except Exception as e:
    logger.error(f"Failed to load unique_accounts.csv: {e}")
    st.error(f"Error loading unique_accounts.csv: {e}")
    unique_accounts = pd.DataFrame(columns=['AccountNumber', 'AccountDescription'])

# Categorization function
def categorize_account(account_number: str, description: str) -> str:
    """
    Categorize an account based on AccountNumber prefix and AccountDescription.
    
    Parameters:
    - account_number: The account number as a string.
    - description: The account description as a string.
    
    Returns:
    - The category of the account (e.g., 'Assets - Cash', 'Operating Expenses').
    """
    try:
        description = str(description).upper().strip() if description else ''
        account_number = str(account_number).strip() if account_number else ''
        
        # Handle invalid account numbers
        if not account_number or not any(c.isdigit() for c in account_number):
            logger.debug(f"Invalid account number: {account_number}, description: {description}")
            return 'Uncategorized'
        
        # Get the first one or two digits
        prefix = account_number[:2] if len(account_number) >= 2 else account_number[:1]
        
        # Prefix-based categorization
        if prefix.startswith('11'):
            return 'Assets - Cash'
        elif prefix.startswith('12'):
            return 'Assets - Accounts Receivable'
        elif prefix.startswith('14'):
            return 'Assets - Prepaid Expenses'
        elif prefix.startswith('16'):
            return 'Assets - Goods in Transit'
        elif prefix == '17':
            if 'FIXED ASSET' in description:
                return 'Fixed Assets'
            elif 'ACCU. DEP.' in description:
                return 'Accumulated Depreciation'
            return 'Fixed Assets'
        elif prefix.startswith('1'):
            return 'Assets - Other'
        elif prefix.startswith('21'):
            return 'Liabilities - Accounts Payable'
        elif prefix.startswith('22'):
            return 'Liabilities - Loans'
        elif prefix.startswith('24'):
            return 'Liabilities - Other Accrued'
        elif prefix.startswith('2'):
            return 'Liabilities - Other'
        elif prefix.startswith('3'):
            return 'Equity'
        elif prefix.startswith('4'):
            return 'Revenue'
        elif prefix.startswith('5'):
            if any(keyword in description for keyword in [
                'MATERIAL', 'LABOUR', 'SUBCONTRACT', 'FREIGHT', 'CUSTOM CLEARANCE'
            ]):
                return 'Cost of Goods Sold'
            elif any(keyword in description for keyword in [
                'BANK SERVICE', 'TRAINING', 'LICENSE', 'INSURANCE', 'PERFORMANCE GUARANTEE', 
                'SUPPORT', 'BID BOND', 'ADVANCE GUARANTEE'
            ]):
                return 'Operating Expenses'
            return 'Operating Expenses'
        elif prefix.startswith('6'):
            return 'Operating Expenses'
        else:
            logger.debug(f"Unmatched prefix for account: {account_number}, description: {description}")
            return 'Uncategorized'
    except Exception as e:
        logger.error(f"Error categorizing account {account_number}: {e}")
        return 'Uncategorized'

# Manual overrides for specific accounts
manual_overrides = {
    '51000007005': 'Operating Expenses',  # COST OF BANK SERVICE CHARGE CBE VDI
    '51000108012': 'Cost of Goods Sold',  # COST OF-EPSS Networking Infra& Modula- FOREIGN...
    '1700': 'Fixed Assets'  # Total Fixed Asset (PPE)
}

# --- Utility Functions ---

def setup_secrets() -> Optional[Dict]:
    """Retrieve API URL and credentials from Streamlit secrets."""
    try:
        secrets = {
            "api_url": st.secrets.get("SAGE_API_URL", "http://196.188.234.230/Sage300WebApi/v1.0/-/IEDATA/GL/GLJournalBatches"),
            "username": st.secrets.get("ADMIN"),
            "password": st.secrets.get("ADMIN"),
            "api_token": st.secrets.get("sage_api_token")
        }
        if not secrets["api_url"]:
            raise ValueError("SAGE_API_URL is not set")
        logger.info("Secrets loaded successfully")
        return secrets
    except Exception as e:
        logger.error(f"Failed to load secrets: {e}")
        st.error(f"Configuration error: {e}. Please check secrets configuration.")
        return None

async def fetch_batch(session: aiohttp.ClientSession, url: str, auth: Optional[tuple] = None, 
                    headers: Optional[Dict] = None, retries: int = RETRY_ATTEMPTS) -> Optional[Dict]:
    """Fetch a single batch from the API with retry logic."""
    headers = headers or {}
    for attempt in range(retries):
        try:
            async with session.get(url, auth=auth, headers=headers, timeout=TIMEOUT) as response:
                response.raise_for_status()
                data = await response.json()
                logger.debug(f"Fetched batch from {url}")
                return data
        except aiohttp.ClientResponseError as e:
            logger.warning(f"Attempt {attempt + 1} failed for {url}: HTTP {e.status} - {e.message}")
            if e.status in (401, 403):
                st.error(f"Authentication error: Invalid credentials (HTTP {e.status})")
                return None
        except (aiohttp.ClientError, json.JSONDecodeError) as e:
            logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
        if attempt < retries - 1:
            await asyncio.sleep(RETRY_DELAY)
    logger.error(f"Failed to fetch {url} after {retries} attempts")
    st.error(f"Failed to fetch data from {url} after {retries} attempts")
    return None

async def fetch_all_batches(api_url_base: str, page_size: int, auth: Optional[tuple], 
                          headers: Optional[Dict]) -> List[Dict]:
    """Async function to fetch all journal batches."""
    all_batches = []
    skip = 0

    async with aiohttp.ClientSession() as session:
        count_url = f"{api_url_base}?$top=1&$skip=0&$count=true"
        count_data = await fetch_batch(session, count_url, auth, headers)
        if not count_data:
            st.error("Failed to fetch total count from API.")
            return all_batches
        total_count = count_data.get('@odata.count', 0)
        logger.info(f"Total journal batches: {total_count}")

        progress_bar = st.progress(0)
        progress_text = st.empty()

        while skip < total_count:
            paginated_url = f"{api_url_base}?$top={page_size}&$skip={skip}"
            data = await fetch_batch(session, paginated_url, auth, headers)
            if not data or not data.get('value'):
                logger.warning("No more data returned by API.")
                break
            all_batches.extend(data['value'])
            skip += page_size

            progress_percent = min(100, int((skip / total_count) * 100))
            progress_bar.progress(progress_percent)
            progress_text.text(f"Fetching data: {progress_percent}%")

        progress_text.text("Data fetch complete!")
        time.sleep(1)
        progress_bar.empty()
        st.success(f"Fetched {len(all_batches)} journal batches.")

    return all_batches

def load_cached_data() -> Optional[List[Dict]]:
    """Load cached data if API fetch fails."""
    try:
        with open('cached_data.json', 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded cached data: {len(data)} batches")
        st.warning("Using cached data due to API failure.")
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load cached data: {e}")
        return None

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_data_from_api() -> Optional[List[Dict]]:
    """Fetch all journal batches from the API with async pagination, wrapped for Streamlit caching."""
    secrets = setup_secrets()
    if not secrets:
        return load_cached_data()

    auth = None
    headers = {}
    if secrets["username"] and secrets["password"]:
        auth = aiohttp.BasicAuth(secrets["username"], secrets["password"])
    elif secrets["api_token"]:
        headers["Authorization"] = f"Bearer {secrets['api_token']}"

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        all_batches = loop.run_until_complete(
            fetch_all_batches(secrets["api_url"], PAGE_SIZE, auth, headers)
        )
        loop.close()
        if all_batches:
            try:
                with open('cached_data.json', 'w') as f:
                    json.dump(all_batches, f)
                logger.info("Saved data to cache")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
        return all_batches
    except Exception as e:
        logger.error(f"Unexpected error in fetch_data_from_api: {e}")
        st.error(f"An error occurred while fetching data: {e}")
        return load_cached_data()

@st.cache_data(show_spinner=False, ttl=3600)
def process_data(all_batches: List[Dict]) -> Optional[pd.DataFrame]:
    """Process fetched data into a Pandas DataFrame with feature engineering."""
    if not all_batches:
        logger.error("No data to process")
        return None

    processed_transactions = []
    missing_mappings = set()
    account_prefixes = set()

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
                    account_description = detail.get('AcctDescription', '')
                    
                    # Log account prefix for debugging
                    if account_number:
                        prefix = str(account_number)[:2] if len(str(account_number)) >= 2 else str(account_number)[:1]
                        account_prefixes.add(prefix)
                    
                    # Categorize account
                    category = manual_overrides.get(account_number, categorize_account(account_number, account_description))
                    
                    if category == 'Uncategorized':
                        missing_mappings.add(f"{account_number}: {account_description}")

                    transaction = {
                        **batch_info,
                        **header_info,
                        'TransactionNumber': detail.get('TransactionNumber'),
                        'AccountNumber': account_number,
                        'AccountDescription': account_description,
                        'Amount': detail.get('Amount'),
                        'JournalDate': detail.get('JournalDate'),
                        'TransactionDescription': detail.get('Description'),
                        'Reference': detail.get('Reference'),
                        'HomeCurrency': detail.get('HomeCurrency'),
                        'AccountCategory': category,
                    }
                    processed_transactions.append(transaction)

        if missing_mappings:
            logger.warning(f"Uncategorized accounts: {list(missing_mappings)[:50]}")  # Limit to 50 for brevity
            st.warning(f"Some accounts are uncategorized: {len(missing_mappings)} unique accounts. Check app.log for details.")
        
        # Log unique prefixes for debugging
        logger.info(f"Account prefixes found: {sorted(account_prefixes)}")

        df = pd.DataFrame(processed_transactions)

        # Type conversions
        date_cols = ['BatchDate', 'PostingDate', 'JournalDate', 'DocumentDate']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)

        numeric_cols = ['Amount', 'NumberOfEntries']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Feature engineering
        if 'JournalDate' in df.columns:
            # Convert to timezone-naive to avoid PeriodArray warning
            df['JournalDate'] = df['JournalDate'].dt.tz_localize(None)
            df['Year'] = df['JournalDate'].dt.year
            df['Month'] = df['JournalDate'].dt.month
            df['Quarter'] = df['JournalDate'].dt.quarter
            df['MonthYear'] = df['JournalDate'].dt.to_period('M').astype(str)
            df['TransactionType'] = df['Amount'].apply(lambda x: 'Debit' if pd.notna(x) and x > 0 else 'Credit')
        else:
            logger.error("JournalDate column missing")
            st.error("Error: JournalDate column missing. Cannot process data.")
            return None

        # Pre-aggregate for performance
        monthly_category_totals = df.groupby(['MonthYear', 'AccountCategory'])['Amount'].sum().reset_index()
        st.session_state['monthly_category_totals'] = monthly_category_totals

        # Log category distribution
        if 'AccountCategory' in df.columns:
            category_counts = df['AccountCategory'].value_counts().to_dict()
            logger.info(f"Category distribution: {category_counts}")

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
            monthly_revenue['Revenue_Change_Pct'] = monthly_revenue['Revenue'].pct_change() * 100
            monthly_trends = monthly_trends.merge(monthly_revenue, on='MonthYear', how='outer')
        else:
            logger.warning("No Revenue accounts found in data")
            monthly_trends['Revenue'] = 0
            monthly_trends['Revenue_Growth_Rate'] = 0
            monthly_trends['Revenue_Change_Pct'] = 0
            st.warning("No Revenue accounts found. Revenue KPIs may be incomplete. Check account numbers starting with '4'.")

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

        # Cash Balance
        cash_accounts = df[df['AccountCategory'].str.startswith('Assets - Cash')]
        if not cash_accounts.empty:
            monthly_cash = cash_accounts.groupby('MonthYear')['Amount'].sum().reset_index(name='Cash_Balance')
            monthly_trends = monthly_trends.merge(monthly_cash, on='MonthYear', how='outer')
        else:
            logger.warning("No Assets - Cash accounts found")
            monthly_trends['Cash_Balance'] = 0

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

        # Net Profit
        monthly_trends['Net_Profit'] = monthly_trends['Revenue'] - monthly_trends['COGS'] - monthly_trends['Operating_Expenses']

        # Transaction Volume
        monthly_volume = df.groupby('MonthYear').size().reset_index(name='Transaction_Volume')
        monthly_trends = monthly_trends.merge(monthly_volume, on='MonthYear', how='outer').fillna(0)

        logger.info("KPIs calculated")
        st.success("KPIs calculated.")
        return monthly_trends

# --- Streamlit App ---

def render_dashboard(df: pd.DataFrame, monthly_trends: pd.DataFrame):
    """Render the Streamlit dashboard."""
    # Center the logo using columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image(
                os.path.join(BASE_DIR, 'assets', 'ie_networks_logo.png'),
                width=200,
            )
        except FileNotFoundError:
            logger.error("Logo file not found at assets/ie_networks_logo.png")
            st.warning("Logo not found. Please ensure 'ie_networks_logo.png' is in the assets folder.")
    st.title("IE Networks CEO Dashboard")

    # KPI Cards
    st.header("At-a-Glance Metrics")
    cols = st.columns(4)
    with cols[0]:
        st.metric("Total Revenue", f"${monthly_trends['Revenue'].sum():,.2f}")
    with cols[1]:
        st.metric("Cash Balance", f"${monthly_trends['Cash_Balance'].sum():,.2f}")
    with cols[2]:
        st.metric("Gross Profit Margin", f"{monthly_trends['Gross_Profit_Margin'].mean():.1f}%")
    with cols[3]:
        st.metric("Net Profit", f"${monthly_trends['Net_Profit'].sum():,.2f}")

    # Alerts
    if 'Revenue_Change_Pct' in monthly_trends.columns:
        alerts = monthly_trends[monthly_trends['Revenue_Change_Pct'].abs() > 20]
        if not alerts.empty:
            st.error("Revenue Alerts: Significant changes detected!")
            st.dataframe(alerts[['MonthYear', 'Revenue_Change_Pct']].style.format({'Revenue_Change_Pct': '{:.2f}%'}))

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
        "Account Categories", options=sorted(df['AccountCategory'].unique()), default=df['AccountCategory'].unique()
    )

    # Filter DataFrame
    filtered_df = df[
        (df['JournalDate'].dt.date >= pd.Timestamp(date_range[0]).date()) &
        (df['JournalDate'].dt.date <= pd.Timestamp(date_range[1]).date()) &
        (df['AccountCategory'].isin(account_categories))
    ]

    # Top Accounts Drill-Down
    if account_categories:
        st.subheader("Top Accounts by Category")
        for category in account_categories:
            cat_data = filtered_df[filtered_df['AccountCategory'] == category]
            if not cat_data.empty:
                top_accounts = cat_data.groupby(['AccountNumber', 'AccountDescription'])['Amount'].sum().nlargest(5).reset_index()
                st.write(f"Top 5 {category} Accounts")
                st.dataframe(top_accounts.style.format({'Amount': '${:,.2f}'}), use_container_width=True)
                # Download button
                csv = cat_data.to_csv(index=False)
                st.download_button(f"Download {category} Transactions", csv, f"{category}_transactions.csv")

    # Display Data
    st.subheader("Transaction Data")
    st.dataframe(filtered_df, use_container_width=True)

    # Visualizations
    st.header("Key Performance Indicators")

    # Revenue
    if 'Revenue' in monthly_trends.columns and monthly_trends['Revenue'].sum() != 0:
        fig_revenue = px.line(monthly_trends, x='MonthYear', y='Revenue', title='Monthly Revenue')
        st.plotly_chart(fig_revenue, use_container_width=True)
    else:
        st.warning("No Revenue data available for visualization.")

    # Revenue Growth
    if 'Revenue_Growth_Rate' in monthly_trends.columns and monthly_trends['Revenue_Growth_Rate'].sum() != 0:
        fig_growth = px.line(monthly_trends, x='MonthYear', y='Revenue_Growth_Rate', title='Revenue Growth Rate')
        st.plotly_chart(fig_growth, use_container_width=True)
    else:
        st.warning("No Revenue Growth Rate data available.")

    # Gross Profit Margin
    if 'Gross_Profit_Margin' in monthly_trends.columns and monthly_trends['Gross_Profit_Margin'].sum() != 0:
        fig_gpm = px.line(monthly_trends, x='MonthYear', y='Gross_Profit_Margin', title='Gross Profit Margin')
        st.plotly_chart(fig_gpm, use_container_width=True)
    else:
        st.warning("No Gross Profit Margin data available.")

    # Operating Expenses
    if 'Operating_Expenses' in monthly_trends.columns and monthly_trends['Operating_Expenses'].sum() != 0:
        fig_opex = px.line(monthly_trends, x='MonthYear', y='Operating_Expenses', title='Operating Expenses')
        st.plotly_chart(fig_opex, use_container_width=True)
    else:
        st.warning("No Operating Expenses data available.")

    # Cash Balance
    if 'Cash_Balance' in monthly_trends.columns and monthly_trends['Cash_Balance'].sum() != 0:
        fig_cash = px.line(monthly_trends, x='MonthYear', y='Cash_Balance', title='Cash Balance')
        st.plotly_chart(fig_cash, use_container_width=True)
    else:
        st.warning("No Cash Balance data available.")

    # Net Profit
    if 'Net_Profit' in monthly_trends.columns and monthly_trends['Net_Profit'].sum() != 0:
        fig_net_profit = px.line(monthly_trends, x='MonthYear', y='Net_Profit', title='Net Profit')
        st.plotly_chart(fig_net_profit, use_container_width=True)
    else:
        st.warning("No Net Profit data available.")

    # Transaction Volume
    if 'Transaction_Volume' in monthly_trends.columns and monthly_trends['Transaction_Volume'].sum() != 0:
        fig_volume = px.line(monthly_trends, x='MonthYear', y='Transaction_Volume', title='Transaction Volume')
        st.plotly_chart(fig_volume, use_container_width=True)
    else:
        st.warning("No Transaction Volume data available.")

    # Revenue by Category
    st.header("Revenue by Account Category")
    revenue_category = filtered_df[filtered_df['AccountCategory'] == 'Revenue']
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

    # Expenses by Category
    st.header("Expenses by Account Category")
    expense_categories = ['Cost of Goods Sold', 'Operating Expenses']
    expense_data = filtered_df[filtered_df['AccountCategory'].isin(expense_categories)]
    if not expense_data.empty:
        expense_category_monthly = (
            expense_data.groupby(['MonthYear', 'AccountCategory'])['Amount'].sum().reset_index()
        )
        fig_expense_cat = px.bar(
            expense_category_monthly,
            x='MonthYear',
            y='Amount',
            color='AccountCategory',
            title='Monthly Expenses by Account Category',
            barmode='stack',
        )
        st.plotly_chart(fig_expense_cat, use_container_width=True)
    else:
        st.warning("No categorized expense data available for visualization.")

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="IE Networks CEO Dashboard", layout="wide")

    all_batches = fetch_data_from_api()
    if not all_batches:
        logger.error("No data fetched. Stopping app.")
        st.stop()

    df_ceo = process_data(all_batches)
    if df_ceo is None or df_ceo.empty:
        logger.error("No processed data available. Stopping app.")
        st.stop()

    monthly_trends = calculate_kpis(df_ceo)
    if monthly_trends is None or monthly_trends.empty:
        logger.error("No KPIs calculated. Stopping app.")
        st.stop()

    render_dashboard(df_ceo, monthly_trends)

if __name__ == "__main__":
    main()
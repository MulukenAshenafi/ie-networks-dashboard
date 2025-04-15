import streamlit as st
import pandas as pd
import requests
import logging
import plotly.express as px
from datetime import datetime

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Account Category Mapping
ACCOUNT_CATEGORY_MAPPING = {
    '11000001001': 'Assets',
    '12000002007': 'Assets',
    '13000001001': 'Assets',
    '21000001001': 'Liabilities',
    '61000001001': 'Operating Expenses',
    '61000003001': 'Operating Expenses',
    '61000006011': 'Operating Expenses',
    '61000008001': 'Operating Expenses',
    '61000017001': 'Operating Expenses',
    '61000018001': 'Operating Expenses',
    '61000019001': 'Operating Expenses',
    '61000022001': 'Operating Expenses',
    '61000025001': 'Operating Expenses',
    '61000026001': 'Operating Expenses',
    '61000027001': 'Operating Expenses',
    '51000002002': 'Cost of Goods Sold',
    '51000002003': 'Cost of Goods Sold',
    '51000005002': 'Cost of Goods Sold',
    '51000007005': 'Operating Expenses',
    '61000002001': 'Operating Expenses',
    '61000004001': 'Operating Expenses',
    '61000006015': 'Operating Expenses',
}

def fetch_data():
    """Fetch journal batch data from Sage 300 API."""
    url = "http://196.188.234.230/Sage300WebApi/v1.0/-/IEDATA/GL/GLJournalBatches?$count=true"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        logging.info(f"Fetched {len(data.get('value', []))} journal batches.")
        return data
    except requests.RequestException as e:
        logging.error(f"Error fetching data: {e}")
        st.error("Failed to fetch data from Sage API.")
        return {}

def process_data(data):
    """Process Sage JSON data into a DataFrame."""
    records = []
    for batch in data.get('value', []):
        for header in batch.get('JournalHeaders', []):
            for detail in header.get('JournalDetails', []):
                record = {
                    'BatchNumber': batch['BatchNumber'],
                    'EntryNumber': header['EntryNumber'],
                    'AccountNumber': detail['AccountNumber'],
                    'AccountDescription': detail.get('AcctDescription', ''),
                    'JournalDate': detail.get('JournalDate'),
                    'Amount': detail['Amount'],
                    'SourceLedger': batch['SourceLedger'],
                }
                records.append(record)
    df = pd.DataFrame(records)
    if df.empty:
        logging.warning("No data processed into DataFrame.")
        return df
    logging.info(f"DataFrame columns: {df.columns.tolist()}")
    logging.info(f"Sample data: \n{df.head().to_string()}")
    df['JournalDate'] = pd.to_datetime(df['JournalDate'], utc=True)
    df['AccountCategory'] = df['AccountNumber'].map(ACCOUNT_CATEGORY_MAPPING).fillna('Uncategorized')
    # Log uncategorized accounts
    uncategorized = df[df['AccountCategory'] == 'Uncategorized'][['AccountNumber', 'AccountDescription']].drop_duplicates()
    if not uncategorized.empty:
        logging.warning(f"Uncategorized accounts: \n{uncategorized.to_string()}")
        logging.warning(f"Total unique uncategorized accounts: {len(uncategorized)}")
        st.warning(f"Some accounts are uncategorized: {len(uncategorized)} unique accounts. Check app.log for details.")
    return df

def calculate_kpis(df):
    """Calculate financial KPIs from categorized data."""
    if df.empty:
        logging.warning("Empty DataFrame passed to calculate_kpis.")
        return {}
    kpis = {}
    # Revenue
    revenue_accounts = df[df['AccountCategory'] == 'Revenue']
    if not revenue_accounts.empty:
        kpis['Revenue'] = revenue_accounts['Amount'].sum()
    else:
        kpis['Revenue'] = 0
        logging.info("No Revenue accounts found.")
        st.warning("No Revenue accounts mapped. Revenue KPIs may be incomplete.")
    
    # Cost of Goods Sold
    cogs_accounts = df[df['AccountCategory'] == 'Cost of Goods Sold']
    kpis['COGS'] = cogs_accounts['Amount'].sum() if not cogs_accounts.empty else 0
    
    # Operating Expenses
    operating_expenses_accounts = df[df['AccountCategory'] == 'Operating Expenses']
    kpis['Operating_Expenses'] = operating_expenses_accounts['Amount'].sum() if not operating_expenses_accounts.empty else 0
    
    # Gross Profit
    kpis['Gross_Profit'] = kpis['Revenue'] - kpis['COGS']
    
    # Cash Balance (simplified as net of Asset accounts)
    asset_accounts = df[df['AccountCategory'] == 'Assets']
    kpis['Cash_Balance'] = asset_accounts['Amount'].sum() if not asset_accounts.empty else 0
    
    return kpis

def main():
    """Main Streamlit app."""
    st.title("IE Networks Financial Dashboard")
    
    # Fetch and process data
    data = fetch_data()
    if not data:
        st.stop()
    
    df = process_data(data)
    if df.empty:
        st.error("No data processed. Check Sage API or data structure.")
        st.stop()
    
    # Create MonthYear column
    if 'JournalDate' not in df.columns:
        logging.error("JournalDate column missing in DataFrame")
        st.error("Error: JournalDate column missing. Please check data source.")
        st.stop()
    df['MonthYear'] = df['JournalDate'].dt.to_period('M').astype(str)
    
    # Sidebar filters
    st.sidebar.header("Filters")
    start_date = st.sidebar.date_input("Start Date", value=datetime(2017, 1, 1))
    end_date = st.sidebar.date_input("End Date", value=datetime.now())
    selected_categories = st.sidebar.multiselect(
        "Account Categories",
        options=df['AccountCategory'].unique(),
        default=['Revenue', 'Cost of Goods Sold', 'Operating Expenses', 'Assets']
    )
    
    # Filter data
    filtered_df = df[
        (df['JournalDate'].dt.date >= start_date) &
        (df['JournalDate'].dt.date <= end_date) &
        (df['AccountCategory'].isin(selected_categories))
    ]
    
    # Calculate KPIs
    kpis = calculate_kpis(filtered_df)
    
    # Display KPIs
    st.header("Key Performance Indicators")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Revenue", f"ETB {kpis.get('Revenue', 0):,.2f}")
        st.metric("COGS", f"ETB {kpis.get('COGS', 0):,.2f}")
    with col2:
        st.metric("Gross Profit", f"ETB {kpis.get('Gross_Profit', 0):,.2f}")
        st.metric("Operating Expenses", f"ETB {kpis.get('Operating_Expenses', 0):,.2f}")
    with col3:
        st.metric("Cash Balance", f"ETB {kpis.get('Cash_Balance', 0):,.2f}")
    
    # Charts
    st.header("Financial Trends")
    if not filtered_df.empty:
        # Monthly Trend
        monthly_data = filtered_df.groupby(['MonthYear', 'AccountCategory'])['Amount'].sum().reset_index()
        fig = px.line(
            monthly_data,
            x='MonthYear',
            y='Amount',
            color='AccountCategory',
            title="Monthly Financial Trends",
            labels={'Amount': 'Amount (ETB)', 'MonthYear': 'Month-Year'}
        )
        st.plotly_chart(fig)
        
        # Category Breakdown
        category_data = filtered_df.groupby('AccountCategory')['Amount'].sum().reset_index()
        fig_pie = px.pie(
            category_data,
            values='Amount',
            names='AccountCategory',
            title="Financial Breakdown by Category"
        )
        st.plotly_chart(fig_pie)
    else:
        st.warning("No data available for the selected filters.")
    
    # Raw Data
    st.header("Raw Data")
    st.dataframe(filtered_df)

if __name__ == "__main__":
    main()
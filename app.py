import streamlit as st
import yfinance as yf
import finnhub
import pandas as pd
from datetime import datetime, timedelta

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI Financial Analyst",
    page_icon="🤖",
    layout="wide"
)

# --- FINNHUB API KEY ---
# Ensure the Finnhub API key is set in Streamlit secrets
try:
    finnhub_client = finnhub.Client(api_key=st.secrets["FINNHUB_API_KEY"])
except Exception as e:
    st.error("Finnhub API key not found. Please add it to your Streamlit secrets.")
    st.stop()


# --- APP TITLE ---
st.title("The AI Financial Analyst Engine 📈")
st.write("An autonomous web application that ingests and analyzes real-time news and official financial reports.")


# --- STOCK TICKER INPUT ---
ticker = st.text_input("Enter a US Stock Ticker Symbol (e.g., AAPL, GOOGL)", "AAPL").upper()


if ticker:
    st.write("---")
    
    # --- DATA FETCHING AND DISPLAY ---
    try:
        # Create two columns for layout
        col1, col2 = st.columns([3, 2]) # 3 parts for chart, 2 for news

        # --- COLUMN 1: PRICE CHART & COMPANY INFO ---
        with col1:
            st.subheader(f"Historical Price Chart for {ticker}")
            
            # yfinance Ticker object
            stock = yf.Ticker(ticker)
            
            # Get historical data (e.g., last 5 years)
            hist_data = stock.history(period="5y")
            
            if hist_data.empty:
                st.warning(f"No historical data found for ticker: {ticker}. Please check the symbol.")
            else:
                # Display line chart
                st.line_chart(hist_data['Close'])

                # Display company information
                info = stock.info
                st.subheader("Company Information")
                st.write(f"**Name:** {info.get('longName', 'N/A')}")
                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                st.write(f"**Website:** {info.get('website', 'N/A')}")
                st.write("**Business Summary:**")
                st.write(info.get('longBusinessSummary', 'No summary available.'))


        # --- COLUMN 2: NEWS HEADLINES ---
        with col2:
            st.subheader(f"Recent News for {ticker}")
            
            # Get today's and last week's date for news query
            today = datetime.now()
            one_week_ago = today - timedelta(days=7)
            
            # Format dates for Finnhub API
            start_date = one_week_ago.strftime('%Y-%m-%d')
            end_date = today.strftime('%Y-%m-%d')
            
            # Fetch news
            news = finnhub_client.company_news(ticker, _from=start_date, to=end_date)
            
            if not news:
                st.info(f"No recent news found for {ticker}.")
            else:
                for item in news[:10]: # Display top 10 news items
                    st.markdown(f"**[{item['headline']}]({item['url']})**")
                    st.write(f"_{datetime.fromtimestamp(item['datetime']).strftime('%Y-%m-%d %H:%M')}_ - {item['source']}")
                    st.write("---")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("This could be due to an invalid ticker symbol or a network issue. Please try again.")
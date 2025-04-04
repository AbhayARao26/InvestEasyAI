import requests
import os
from dotenv import load_dotenv
from newsapi import NewsApiClient
import pandas as pd
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# API Keys
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# Initialize News API client
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

def get_stock_quote(symbol):
    """
    Get real-time stock quote from Alpha Vantage
    """
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        data = response.json()
        
        if "Global Quote" in data:
            quote = data["Global Quote"]
            return {
                "symbol": quote["01. symbol"],
                "price": float(quote["05. price"]),
                "change": float(quote["09. change"]),
                "change_percent": float(quote["10. change percent"].strip('%')),
                "volume": int(quote["06. volume"]),
                "last_updated": quote["07. latest trading day"]
            }
        return None
    except Exception as e:
        print(f"Error fetching stock quote: {str(e)}")
        return None

def get_stock_history(symbol, interval="daily", outputsize="compact"):
    """
    Get historical stock data from Alpha Vantage
    """
    try:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_{interval.upper()}&symbol={symbol}&outputsize={outputsize}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        data = response.json()
        
        if f"Time Series ({interval.capitalize()})" in data:
            time_series = data[f"Time Series ({interval.capitalize()})"]
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            return df
        return None
    except Exception as e:
        print(f"Error fetching stock history: {str(e)}")
        return None

def get_company_overview(symbol):
    """
    Get company overview from Alpha Vantage
    """
    try:
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        return response.json()
    except Exception as e:
        print(f"Error fetching company overview: {str(e)}")
        return None

def get_news(symbol, days=7):
    """
    Get news articles related to a stock symbol
    """
    try:
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        news = newsapi.get_everything(
            q=symbol,
            from_param=from_date,
            language='en',
            sort_by='publishedAt',
            page_size=10
        )
        return news['articles']
    except Exception as e:
        print(f"Error fetching news: {str(e)}")
        return []

def get_market_sentiment(symbol):
    """
    Get market sentiment indicators from Alpha Vantage
    """
    try:
        url = f"https://www.alphavantage.co/query?function=SENTIMENT&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        return response.json()
    except Exception as e:
        print(f"Error fetching market sentiment: {str(e)}")
        return None

def get_top_gainers():
    """
    Get top gaining stocks from Alpha Vantage
    """
    try:
        url = f"https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        data = response.json()
        return data.get("top_gainers", [])
    except Exception as e:
        print(f"Error fetching top gainers: {str(e)}")
        return []

def get_top_losers():
    """
    Get top losing stocks from Alpha Vantage
    """
    try:
        url = f"https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        data = response.json()
        return data.get("top_losers", [])
    except Exception as e:
        print(f"Error fetching top losers: {str(e)}")
        return [] 
import streamlit as st
import requests
import json
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
import google.generativeai as genai
from newsapi import NewsApiClient
import sqlite3
import bcrypt
import jwt
import yfinance as yf
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error
import time

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Database setup
def get_db_connection():
    """Create a database connection"""
    conn = sqlite3.connect('investeasy.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database with required tables"""
    conn = get_db_connection()
    c = conn.cursor()
    
    # Create users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create portfolio table
    c.execute('''
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            stock_symbol TEXT NOT NULL,
            quantity REAL NOT NULL,
            avg_price REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users (username)
        )
    ''')
    
    # Create financial_goals table
    c.execute('''
        CREATE TABLE IF NOT EXISTS financial_goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            investment_amount REAL NOT NULL,
            target_return REAL NOT NULL,
            time_period INTEGER NOT NULL,
            risk_tolerance REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users (username)
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

# API Keys
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Initialize News API
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# JWT Secret
JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key')

def register_user(username, password):
    """Register a new user"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Hash the password
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # Insert the new user
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)',
                 (username, hashed.decode('utf-8')))
        
        conn.commit()
        conn.close()
        return True, "User registered successfully"
    except sqlite3.IntegrityError:
        return False, "Username already exists"
    except Exception as e:
        return False, f"Error registering user: {str(e)}"

def verify_user(username, password):
    """Verify user credentials"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Get user's hashed password
        c.execute('SELECT password FROM users WHERE username = ?', (username,))
        result = c.fetchone()
        
        conn.close()
        
        if result and bcrypt.checkpw(password.encode('utf-8'), result['password'].encode('utf-8')):
            return True, "Login successful"
        return False, "Invalid username or password"
    except Exception as e:
        return False, f"Error verifying user: {str(e)}"

import yfinance as yf

def get_user_portfolio(username):
    """Get user's portfolio with current prices using Yahoo Finance"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Get portfolio stocks
        c.execute('''
            SELECT stock_symbol, quantity, avg_price 
            FROM portfolio 
            WHERE username = ?
        ''', (username,))
        portfolio = c.fetchall()
        
        conn.close()
        
        if not portfolio:
            return []

        portfolio_data = []

        for stock in portfolio:
            try:
                symbol = stock['stock_symbol']
                quantity = float(stock['quantity'])
                avg_price = float(stock['avg_price'])

                # Fetch current stock price from Yahoo Finance
                ticker = yf.Ticker(symbol)
                stock_info = ticker.info

                if 'currentPrice' in stock_info and stock_info['currentPrice']:
                    current_price = float(stock_info['currentPrice'])

                    total_investment = quantity * avg_price
                    current_value = quantity * current_price
                    profit_loss = current_value - total_investment
                    profit_loss_pct = (profit_loss / total_investment) * 100

                    portfolio_data.append({
                        'stock_symbol': symbol,
                        'quantity': quantity,
                        'avg_price': avg_price,
                        'current_price': current_price,
                        'total_investment': total_investment,
                        'current_value': current_value,
                        'profit_loss': profit_loss,
                        'profit_loss_pct': profit_loss_pct
                    })
                else:
                    print(f"[yfinance] Could not find current price for {symbol}")
            except Exception as e:
                print(f"[ERROR] While processing {symbol}: {str(e)}")
        
        return portfolio_data

    except Exception as e:
        print(f"[DB ERROR] {str(e)}")
        return []


def add_stock_to_portfolio(username, stock_symbol, quantity, avg_price):
    """Add or update a stock in user's portfolio"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Check if stock already exists
        c.execute('''
            SELECT quantity FROM portfolio 
            WHERE username = ? AND stock_symbol = ?
        ''', (username, stock_symbol))
        existing = c.fetchone()
        
        if existing:
            # Update existing stock
            new_quantity = existing['quantity'] + quantity
            new_avg_price = ((existing['quantity'] * avg_price) + (quantity * avg_price)) / new_quantity
            c.execute('''
                UPDATE portfolio 
                SET quantity = ?, avg_price = ?
                WHERE username = ? AND stock_symbol = ?
            ''', (new_quantity, new_avg_price, username, stock_symbol))
        else:
            # Add new stock
            c.execute('''
                INSERT INTO portfolio (username, stock_symbol, quantity, avg_price)
                VALUES (?, ?, ?, ?)
            ''', (username, stock_symbol, quantity, avg_price))
        
        conn.commit()
        conn.close()
        return True, "Stock added successfully"
    except Exception as e:
        return False, f"Error adding stock: {str(e)}"

def get_user_financial_goals(username):
    """Get user's financial goals"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        c.execute('''
            SELECT * FROM financial_goals 
            WHERE username = ? 
            ORDER BY created_at DESC 
            LIMIT 1
        ''', (username,))
        goals = c.fetchone()
        
        conn.close()
        return goals
    except Exception as e:
        print(f"Error fetching financial goals: {str(e)}")
        return None

def update_financial_goals(username, investment_amount, target_return, time_period, risk_tolerance):
    """Update user's financial goals"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO financial_goals 
            (username, investment_amount, target_return, time_period, risk_tolerance)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, investment_amount, target_return, time_period, risk_tolerance))
        
        conn.commit()
        conn.close()
        return True, "Financial goals updated successfully"
    except Exception as e:
        return False, f"Error updating financial goals: {str(e)}"

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=1)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm='HS256')
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.JWTError:
        return None

# Configure Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # Use the correct model name for Gemini Pro
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        print(f"Error initializing Gemini model: {str(e)}")
        model = None
else:
    print("Gemini API key not found")
    model = None

# Configure News API
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
if NEWS_API_KEY:
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    except Exception as e:
        print(f"Error initializing News API: {str(e)}")
        newsapi = None
else:
    print("News API key not found")
    newsapi = None

# Database configuration
db_config = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'investeasy')
}

# FastAPI backend URL
BACKEND_URL = "http://localhost:8000"

# Initialize session state
if 'user' not in st.session_state:
    st.session_state.user = None
if 'token' not in st.session_state:
    st.session_state.token = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'api_request_count' not in st.session_state:
    st.session_state.api_request_count = 0
if 'last_request_date' not in st.session_state:
    st.session_state.last_request_date = datetime.now().date()
if 'cached_portfolio' not in st.session_state:
    st.session_state.cached_portfolio = None
if 'portfolio_last_updated' not in st.session_state:
    st.session_state.portfolio_last_updated = None

def login(username, password):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Get user from database
        c.execute("""
            SELECT id, username, password 
            FROM users 
            WHERE username = ?
        """, (username,))
        
        user = c.fetchone()
        
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            # Create token for the user
            token = create_access_token(data={"sub": username})
            st.session_state.token = token
            st.session_state.user = username
            return True, "Login successful"
        return False, "Invalid credentials"
    except Exception as e:
        st.error(f"Error during login: {str(e)}")
        return False, "Error during login"
    finally:
        if 'c' in locals():
            c.close()
        if 'conn' in locals():
            conn.close()

def register(username, password):
    """Register a new user"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Check if username already exists
        c.execute("SELECT id FROM users WHERE username = ?", (username,))
        if c.fetchone():
            return False, "Username already exists"
        
        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # Insert new user
        c.execute(
            """
            INSERT INTO users (username, password)
            VALUES (?, ?)
            """,
            (username, hashed_password.decode('utf-8'))
        )
        
        conn.commit()
        
        # Get the new user's ID
        c.execute("SELECT id FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        
        if user:
            # Create a token for the new user
            token = create_access_token(data={"sub": username})
            st.session_state.token = token
            st.session_state.user = username
            return True, "Registration successful"
        else:
            return False, "Failed to create user"
            
    except Exception as e:
        return False, str(e)
    finally:
        if 'c' in locals():
            c.close()
        if 'conn' in locals():
            conn.close()

def check_api_limit():
    current_date = datetime.now().date()
    
    # Reset counter if it's a new day
    if current_date != st.session_state.last_request_date:
        st.session_state.api_request_count = 0
        st.session_state.last_request_date = current_date
    
    # Check if we're approaching the limit
    if st.session_state.api_request_count >= 20:  # Warning at 20 requests
        st.warning(f"⚠️ Warning: You have used {st.session_state.api_request_count}/25 AlphaVantage API requests today.")
    
    # Check if we've hit the limit
    if st.session_state.api_request_count >= 25:
        st.error("❌ Daily AlphaVantage API limit reached. Please try again tomorrow.")
        return False
    
    return True

def get_stock_data(symbol):
    try:
        # Use yfinance instead of AlphaVantage
        stock = yf.Ticker(symbol)
        
        # Get 1 month of data with 1 day interval
        hist = stock.history(period="1mo", interval="1d")
        
        if hist.empty:
            # Try getting data with a different interval if 1d fails
            hist = stock.history(period="1mo", interval="1wk")
            if hist.empty:
                return None
        
        # Calculate metrics from yfinance data
        current_price = hist['Close'].iloc[-1]
        prev_price = hist['Close'].iloc[0]
        change = current_price - prev_price
        change_percent = (change / prev_price) * 100
        volume = hist['Volume'].iloc[-1]
        
        # Get additional info from yfinance
        info = stock.info
        beta = info.get('beta', 1.0)
        market_cap = info.get('marketCap', 0)
        sector = info.get('sector', 'Unknown')
        
        return {
            "symbol": symbol,
            "price": float(current_price),
            "change": float(change),
            "change_percent": float(change_percent),
            "volume": int(volume),
            "beta": float(beta),
            "market_cap": float(market_cap),
            "sector": sector
        }
    except Exception as e:
        print(f"Error fetching stock data for {symbol}: {str(e)}")
        return None
    
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load FinBERT model and tokenizer once (cache for reuse)
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
model.eval()  # Set to evaluation mode

    
def get_finbert_sentiment(text):
    try:
        response = requests.post(
            "https://abhay250--finebert-backend.hf.space/sentiment",
            json={"text": text}
        )
        if response.status_code == 200:
            return response.json().get("sentiment_score", 0.0)
        else:
            return 0.0
    except:
        return 0.0


def get_news_sentiment(symbol):
    if not newsapi:
        st.warning("News API not configured")
        return 0.0  # Neutral sentiment

    try:
        stock = yf.Ticker(symbol)
        company_name = stock.info.get('longName', symbol)

        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        news = newsapi.get_everything(
            q=company_name,
            from_param=from_date,
            language='en',
            sort_by='publishedAt',
            page_size=5
        )

        if not news.get('articles'):
            return 0.0  # Neutral if no news

        sentiments = []
        for article in news['articles']:
            try:
                sentiment_score = get_finbert_sentiment(article['title'])
                sentiments.append(sentiment_score)
            except:
                continue

        if not sentiments:
            return 0.0

        return np.mean(sentiments)

    except Exception as e:
        print(f"Error fetching news for {symbol}: {str(e)}")
        return 0.0


def analyze_sentiment(text):
    if not model:
        return "neutral (AI model not available)"
    
    try:
        prompt = f"""
        Analyze the sentiment of this financial news text and provide a one-word response:
        positive, negative, or neutral.
        
        Text: {text}
        
        Response (one word only):
        """
        response = model.generate_content(prompt)
        return response.text.strip().lower()
    except Exception as e:
        st.error(f"Error analyzing sentiment: {str(e)}")
        return "neutral (error in analysis)"

def get_ai_response(user_input):
    if not model:
        return "I apologize, but the AI model is not available at the moment. Please check your Gemini API key configuration."
    
    try:
        prompt = f"""
        You are a helpful financial advisor assistant. Please provide a clear and concise response to the following question:

        {user_input}

        Keep your response focused on financial advice and investment-related information.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}. Please try again later."

def get_user_portfolio_with_prices(username):
    """Fetch user's portfolio with current prices from MySQL database"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        c.execute("""
            SELECT stock_symbol, quantity, avg_price 
            FROM portfolio 
            WHERE username = ?
        """, (username,))
        
        raw_portfolio = c.fetchall()
        conn.close()

        enriched_portfolio = []
        for stock in raw_portfolio:
            ticker = stock['stock_symbol']
            try:
                stock_data = yf.Ticker(ticker)
                current_price = stock_data.info.get("regularMarketPrice", None)
                if current_price is None:
                    current_price = stock_data.history(period="1d")["Close"].iloc[-1]
                
                enriched_portfolio.append({
                    "ticker": ticker,
                    "buy_price": float(stock['avg_price']),
                    "quantity": int(stock['quantity']),
                    "current_price": round(float(current_price), 2)
                })
            except Exception as e:
                print(f"Error fetching price for {ticker}: {str(e)}")
                continue

        return enriched_portfolio
    except Exception as e:
        print(f"Database error: {str(e)}")
        return []

def build_prompt_with_live_prices(user_question, portfolio=None):
    """Build a prompt for Gemini with portfolio context"""
    base_instruction = (
        "You are a financial assistant. Give concise, smart, and friendly advice. "
        "Use the user's portfolio and live market data to provide insights.\n\n"
    )

    if portfolio:
        portfolio_str = "\n".join([
            f"- {item['quantity']} shares of {item['ticker']} at ${item['buy_price']} (Current: ${item['current_price']})"
            for item in portfolio
        ])
        context = f"User Portfolio with Live Prices:\n{portfolio_str}\n\n"
        prompt = base_instruction + context + f"User Question: {user_question}\n"
    else:
        prompt = base_instruction + f"User Question: {user_question}\n"

    return prompt

def extract_stock_ticker(user_question, portfolio):
    for item in portfolio:
        if item['stock_symbol'].lower() in user_question.lower():
            return item['stock_symbol']
    return None

def predict_future_price_sarimax(ticker, steps=1, show_plot=False):
    # Download 6 months of historical data
    stock = yf.Ticker(ticker)
    df = stock.history(period="6mo")

    if df.empty or 'Close' not in df:
        print(f"⚠️ Data for {ticker} not available.")
        return None, 0.0

    df = df[["Close"]].dropna()

    if len(df) < 30:
        print(f"⚠️ Not enough data for {ticker} to train SARIMAX.")
        return None, 0.0

    # Train/Test Split (90% train, 10% test)
    split_idx = int(len(df) * 0.9)
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]

    # Train SARIMAX model
    try:
        model = SARIMAX(train['Close'], order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
        results = model.fit(disp=False)
    except Exception as e:
        print(f"❌ SARIMAX training failed for {ticker}: {e}")
        return None, 0.0

    # Forecast on test set to evaluate
    try:
        forecast = results.forecast(steps=len(test))
        forecast.index = test.index
        mape = mean_absolute_percentage_error(test['Close'], forecast)
        accuracy = round((1 - mape) * 100, 2)
    except:
        accuracy = 0.0

    # # Optional plot
    # if show_plot:
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(train['Close'], label="Train")
    #     plt.plot(test['Close'], label="Actual")
    #     if 'forecast' in locals():
    #         plt.plot(forecast, label="Forecast", linestyle="--")
    #     plt.title(f"{ticker} Forecast - Accuracy: {accuracy}%")
    #     plt.legend()
    #     plt.show()

    # Predict next day(s)
    try:
        future_forecast = results.forecast(steps=steps)
        if len(future_forecast) == 0:
            raise ValueError("Empty forecast output.")
        predicted_price = round(future_forecast.iloc[-1], 2)
    except Exception as e:
        print(f"❌ Forecasting failed for {ticker}: {e}")
        predicted_price = None

    return predicted_price, accuracy

def predict_future_price(ticker):
    predicted_price, acc = predict_future_price_sarimax(ticker)
    if predicted_price is None:
        # fallback naive logic
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")["Close"]
        if len(hist) > 0:
            predicted_price = round(hist.mean() * 1.02, 2)
        else:
            predicted_price = "Unavailable"
    return predicted_price

def buy_hold_sell_analysis(user_question, username):

    portfolio = get_user_portfolio(username)

    ticker_in_question = extract_stock_ticker(user_question, portfolio)
    if ticker_in_question:
        ticker_in_question = ticker_in_question.lower()

    relevant_stocks = [
        stock for stock in portfolio
        if ticker_in_question is None or stock['stock_symbol'].lower() == ticker_in_question
    ]

    if not relevant_stocks:
        return f"❌ No matching stock found for '{ticker_in_question}' in your portfolio."

    summaries = []

    for stock in relevant_stocks:
        ticker = stock['stock_symbol']
        buy_price = stock['avg_price']
        current_price = stock['current_price']
        predicted_price = predict_future_price(ticker)
        sentiment = get_news_sentiment(ticker)

        if predicted_price == "Unavailable":
            action = "HOLD"
            reason = "Prediction data not available."
        elif current_price < buy_price and predicted_price < buy_price and sentiment <= 0:
            action = "SELL"
            reason = "Prices are below buy price, and news sentiment is negative."
        elif predicted_price > current_price * 1.1 and sentiment > 0:
            action = "BUY"
            reason = "Future price is strong and sentiment is positive."
        elif sentiment < 0 and current_price > buy_price:
            action = "SELL"
            reason = "News sentiment is negative despite profit. Consider reducing risk."
        else:
            action = "HOLD"
            reason = "Market data and news suggest holding for now."

        summary = (
            f"Stock: {ticker}\n"
            f"Buy Price: ₹{buy_price}, Current Price: ₹{current_price}, Predicted Price: ₹{predicted_price}, "
            f"Sentiment Score: {sentiment}\n"
            f"Action: {action}\n"
            f"Reason: {reason}\n"
        )
        summaries.append(summary)

    combined_summary = "\n---\n".join(summaries)

    # ✅ Send to Gemini API for natural response generation
    gemini_prompt = f"""
You are an intelligent financial assistant.

A user asked: '{user_question}'

Below is their portfolio analysis for the relevant stock(s). Provide a **precise, helpful, and conversational** recommendation **only based on the data provided**.
Avoid mentioning other stocks unless they are explicitly part of the user's question.
If the stock mentioned is not in the portfolio, mention that politely and suggest the user to provide details.

--- Portfolio Analysis ---
{combined_summary}
---------------------------

🎯 Give a clear buy/sell/hold recommendation for the relevant stock(s) above, with a 1-2 sentence reasoning.
Avoid repeating raw data. Respond in a natural, assistant-like tone. Be concise and helpful.
"""

    response = model.generate_content(gemini_prompt).text
    return response.strip()

def general_qa_agent(user_question):
    """Handle general financial questions"""
    prompt = f"""
    You are a financial expert who explains concepts simply and concisely.

    Guidelines:
    1. Answer in 2-3 short lines maximum
    2. Use simple language
    3. Include a quick example if helpful
    4. Format with line breaks for readability

    Question: {user_question}

    Answer:
    """

    try:
        response = model.generate_content(prompt)
        answer = response.text.strip()
        if len(answer.split('\n')) > 3:
            answer = '\n'.join(answer.split('\n')[:3])
        return answer
    except Exception as e:
        return f"Sorry, I couldn't process that. Please try again. ({str(e)})"

def portfolio_insight_agent(user_question, portfolio):
    """Provide insights based on user's portfolio"""
    prompt = build_prompt_with_live_prices(user_question, portfolio)
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Sorry, I couldn't analyze your portfolio. Please try again. ({str(e)})"

def answer_user_query(user_question, username):
    """Main function to handle user queries"""
    question_lower = user_question.lower()
    # 1. Check if the question is about Buy/Hold/Sell decision
    decision_keywords = ['buy', 'sell', 'hold', 'should i buy', 'should i sell', 'is it time to sell']
    if any(keyword in question_lower for keyword in decision_keywords):
        try:
            return buy_hold_sell_analysis(user_question, username)
        except Exception as e:
            return f"⚠️ Could not evaluate buy/sell decision: {str(e)}"
        
    # Check if question is about portfolio
    portfolio_keywords = ['portfolio', 'stocks', 'investments', 'holdings', 'how is my portfolio']
    if any(keyword in user_question.lower() for keyword in portfolio_keywords):
        portfolio = get_user_portfolio(username)
        if portfolio:
            # Calculate portfolio metrics
            total_value = sum(float(stock['current_price']) * float(stock['quantity']) for stock in portfolio)
            total_investment = sum(float(stock['total_investment']) for stock in portfolio)
            total_profit_loss = total_value - total_investment
            
            # Calculate daily change using profit/loss percentage
            daily_change = portfolio[0]['profit_loss_pct'] if portfolio else 0
            
            # Get user's full name from database
            try:
                conn = get_db_connection()
                c = conn.cursor()
                c.execute("SELECT full_name FROM users WHERE username = ?", (username,))
                user = c.fetchone()
                user_name = user['full_name'] if user and user['full_name'] else username
                conn.close()
            except Exception as e:
                print(f"Error fetching user name: {str(e)}")
                user_name = username
            
            # Build detailed portfolio context
            portfolio_context = f"""
            Portfolio Analysis for {user_name}

            Portfolio Summary:
            • Total Investment: ${total_investment:,.2f}
            • Current Value: ${total_value:,.2f}
            • Total Profit/Loss: ${total_profit_loss:,.2f} ({daily_change:.2f}%)
            
            Individual Holdings:
            """
            
            for stock in portfolio:
                stock_profit_loss = (float(stock['current_price']) - float(stock['total_investment'])) * float(stock['quantity'])
                stock_profit_loss_percentage = ((float(stock['current_price']) - float(stock['total_investment'])) / float(stock['total_investment'])) * 100
                portfolio_context += f"""
                • {stock['stock_symbol']}: {stock['quantity']} shares
                  Buy Price: ${float(stock['total_investment']):.2f}
                  Current Price: ${float(stock['current_price']):.2f}
                  Profit/Loss: ${stock_profit_loss:,.2f} ({stock_profit_loss_percentage:.2f}%)
                """
            
            prompt = f"""
            You are a financial advisor analyzing a user's portfolio. Provide a detailed analysis based on the following information:

            {portfolio_context}

            User's Question: {user_question}

            Guidelines:
            1. Start with a clear summary of overall performance
            2. Highlight significant gains or losses
            3. Mention any concerning trends
            4. Provide specific insights about individual holdings if relevant
            5. Keep the response professional but conversational
            6. Format with clear sections and bullet points for readability
            7. Use consistent formatting throughout the response
            8. Address the user by their name ({user_name})
            """
            
            try:
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                return f"Sorry, I couldn't analyze your portfolio. Please try again. ({str(e)})"
    
    # For general questions
    return general_qa_agent(user_question)

def main():
    st.set_page_config(
        page_title="InvestEasy",
        page_icon="📈",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            margin-top: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("📈 InvestEasy - Your AI-Powered Investment Assistant")

    # Authentication
    if not st.session_state.user:
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.subheader("Login")
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login"):
                success, message = login(username, password)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

        with tab2:
            st.subheader("Register")
            new_username = st.text_input("New Username", key="reg_username")
            new_password = st.text_input("New Password", type="password", key="reg_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            
            if st.button("Register"):
                if not new_username or not new_password:
                    st.error("Please fill in all fields")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    success, message = register(new_username, new_password)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)

    else:
        # Main Dashboard
        st.sidebar.title(f"Welcome, {st.session_state.user}!")
        
        # Display API usage in sidebar
        st.sidebar.info(f"AlphaVantage API Usage: {st.session_state.api_request_count}/25 requests today")
        
        # Navigation
        page = st.sidebar.radio(
            "Navigation",
            ["Dashboard", "Portfolio", "Financial Goals", "News & Alerts", "Chat", "Settings", "Recommendations"]
        )

        if page == "Dashboard":
            st.header("Dashboard")
            
            # Get portfolio data
            portfolio = get_user_portfolio(st.session_state.user)
            
            if portfolio:
                # Calculate portfolio metrics
                total_value = sum(float(stock['current_price']) * float(stock['quantity']) for stock in portfolio)
                total_investment = sum(float(stock['total_investment']) for stock in portfolio)
                total_profit_loss = total_value - total_investment
                
                # Calculate daily change using profit/loss percentage
                daily_change = portfolio[0]['profit_loss_pct'] if portfolio else 0
                
                # Portfolio Summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Portfolio Value", f"${total_value:,.2f}")
                with col2:
                    st.metric("Daily Change", f"{daily_change:.2f}%")
                with col3:
                    pl_color = "green" if total_profit_loss >= 0 else "red"
                    st.metric("Total Profit/Loss", f"${total_profit_loss:,.2f}", 
                             delta=f"{(total_profit_loss/total_investment)*100:.2f}%")

            else:
                st.info("Your portfolio is empty. Add some stocks to get started!")

        elif page == "Portfolio":
            st.header("Portfolio Management")
            
            # Add new stock
            st.subheader("Add New Stock")
            col1, col2, col3 = st.columns(3)
            with col1:
                stock_symbol = st.text_input("Stock Symbol").upper()
            with col2:
                quantity = st.number_input("Quantity", min_value=1, value=1)
            with col3:
                avg_price = st.number_input("Average Buy Price", min_value=0.01, value=1.0)
            
            if st.button("Add Stock"):
                if stock_symbol and quantity and avg_price:
                    success, message = add_stock_to_portfolio(st.session_state.user, stock_symbol, quantity, avg_price)
                    if success:
                        st.success(message)
                    else:
                        st.error(f"Failed to add stock: {message}")
                else:
                    st.warning("Please fill in all fields")

            # Current Portfolio
            st.subheader("Current Portfolio")
            portfolio = get_user_portfolio(st.session_state.user)
            
            if portfolio:
                # Calculate total portfolio value and metrics
                total_value = sum(float(stock['current_price']) * float(stock['quantity']) for stock in portfolio)
                total_investment = sum(float(stock['total_investment']) for stock in portfolio)
                total_profit_loss = total_value - total_investment
                
                # Display portfolio metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Portfolio Value", f"${total_value:,.2f}")
                with col2:
                    st.metric("Total Investment", f"${total_investment:,.2f}")
                with col3:
                    pl_color = "green" if total_profit_loss >= 0 else "red"
                    st.metric("Total Profit/Loss", f"${total_profit_loss:,.2f}", 
                             delta=f"{(total_profit_loss/total_investment)*100:.2f}%")
                
                # Display portfolio table
                df = pd.DataFrame(portfolio)
                df['Current Value'] = df['current_price'] * df['quantity']
                df['Profit/Loss'] = df['profit_loss']
                df['P/L %'] = df['profit_loss_pct']
                
                st.dataframe(
                    df[[
                        'stock_symbol', 'quantity', 'total_investment', 
                        'current_price', 'Current Value', 'Profit/Loss', 'P/L %'
                    ]].style.format({
                        'total_investment': '${:.2f}',
                        'current_price': '${:.2f}',
                        'Current Value': '${:,.2f}',
                        'Profit/Loss': '${:,.2f}',
                        'P/L %': '{:.2f}%'
                    })
                )
            else:
                st.info("Your portfolio is empty. Add some stocks to get started!")

        elif page == "Financial Goals":
            st.header("Financial Goals")
            
            # Get current financial goals
            current_goals = get_user_financial_goals(st.session_state.user)
            
            if current_goals:
                st.subheader("Current Financial Goals")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Investment Amount", f"${current_goals['investment_amount']:,.2f}")
                with col2:
                    st.metric("Target Return", f"{current_goals['target_return']}%")
                with col3:
                    st.metric("Time Period", f"{current_goals['time_period']} years")
                with col4:
                    risk_value = float(current_goals['risk_tolerance'])
                    risk_label = "Conservative" if risk_value < 0.33 else "Moderate" if risk_value < 0.66 else "Aggressive"
                    st.metric("Risk Tolerance", risk_label)
                
                # st.write(f"Last Updated: {current_goals['created_at'].strftime('%Y-%m-%d %H:%M:%S')}")
                created_at_str = current_goals['created_at']
                created_at_dt = datetime.strptime(created_at_str, '%Y-%m-%d %H:%M:%S')  # adjust format if needed
                st.write(f"Last Updated: {created_at_dt.strftime('%Y-%m-%d %H:%M:%S')}")

            else:
                st.info("Financial goals have not been set yet.")
            
            # Update Financial Goals Form
            st.subheader("Update Financial Goals")
            with st.form("financial_goals_form"):
                # Convert decimal values to float for the form
                current_investment = float(current_goals['investment_amount']) if current_goals else 0.0
                current_return = float(current_goals['target_return']) if current_goals else 0.0
                current_period = int(current_goals['time_period']) if current_goals else 1
                current_risk = float(current_goals['risk_tolerance']) if current_goals else 0.5
                
                col1, col2 = st.columns(2)
                with col1:
                    investment_amount = st.number_input(
                        "Investment Amount ($)",
                        min_value=0.0,
                        value=current_investment,
                        step=1000.0,
                        format="%.2f"
                    )
                    target_return = st.number_input(
                        "Target Return (%)",
                        min_value=0.0,
                        value=current_return,
                        step=1.0,
                        format="%.2f"
                    )
                with col2:
                    time_period = st.number_input(
                        "Time Period (years)",
                        min_value=1,
                        value=current_period,
                        step=1
                    )
                    risk_tolerance = st.slider(
                        "Risk Tolerance",
                        min_value=0.0,
                        max_value=1.0,
                        value=current_risk,
                        step=0.01,
                        format="%.2f",
                        help="0 = Conservative, 1 = Aggressive"
                    )
                
                # Risk tolerance explanation
                risk_value = risk_tolerance
                risk_label = "Conservative" if risk_value < 0.33 else "Moderate" if risk_value < 0.66 else "Aggressive"
                st.info(f"Selected Risk Level: {risk_label}")
                
                submitted = st.form_submit_button("Update Goals")
                if submitted:
                    if investment_amount <= 0 or target_return <= 0 or time_period <= 0:
                        st.error("Please enter valid values for all fields")
                    else:
                        success, message = update_financial_goals(
                            st.session_state.user,
                            float(investment_amount),
                            float(target_return),
                            int(time_period),
                            float(risk_tolerance)
                        )
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)

        elif page == "News & Alerts":
            st.header("News & Alerts")
            
            # Stock News
            stock_symbol = st.text_input("Enter Stock Symbol", key="news_stock_symbol")
            if stock_symbol:
                try:
                    # Fetch news articles
                    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                    url = f"https://newsapi.org/v2/everything?q={stock_symbol}&from={yesterday}&language=en&sortBy=relevancy&apiKey={NEWS_API_KEY}"
                    response = requests.get(url, timeout=5)
                    news_data = response.json()
                    
                    if 'articles' in news_data:
                        articles = news_data['articles'][:5]  # Get top 5 articles
                        
                        if articles:
                            st.subheader(f"Latest News for {stock_symbol}")
                            
                            for article in articles:
                                with st.expander(article['title']):
                                    st.write(f"**Source:** {article['source']['name']}")
                                    st.write(f"**Published:** {article['publishedAt'][:10]}")
                                    st.write(article['description'])
                                    if article['url']:
                                        st.markdown(f"[Read more]({article['url']})")
                                    
                                    # Analyze sentiment
                                    try:
                                        sentiment = sia.polarity_scores(article['title'])['compound']
                                        print(sentiment)
                                        # sentiment = TextBlob(article['title']).sentiment.polarity
                                        sentiment_label = "Positive" if sentiment > 0.3 else "Negative" if sentiment < -0.3 else "Neutral"
                                        st.write(f"**Sentiment:** {sentiment_label} ({sentiment:.2f})")
                                    except:
                                        st.write("**Sentiment:** Unable to analyze")
                        else:
                            st.info("No news articles found for this stock symbol.")
                    else:
                        st.error("Error fetching news. Please try again later.")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("This might be due to API rate limits. Please try again later.")

        elif page == "Chat":
            st.header("💬 Financial Assistant")
            
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask me anything about finance or your portfolio"):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Get AI response
                with st.chat_message("assistant"):
                    response = answer_user_query(prompt, st.session_state.user)
                    st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

        elif page == "Settings":
            st.header("Settings")
            
            # Profile Settings
            st.subheader("Profile Settings")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            
            if st.button("Update Password"):
                if new_password == confirm_password:
                    # Implement password update logic
                    st.success("Password updated successfully!")
                else:
                    st.error("Passwords do not match")

            # Logout
            if st.button("Logout"):
                st.session_state.user = None
                st.session_state.token = None
                st.session_state.chat_history = []
                st.rerun()

        elif page == "Recommendations":
            st.header("Recommendations")

if __name__ == "__main__":
    main() 


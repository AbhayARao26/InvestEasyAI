import requests
import numpy as np
from datetime import datetime
import google.generativeai as genai
from collections import defaultdict

# APIs
STOCKS_API_KEY = "W44FCQUAEIK0ZC5J"
NEWS_API_KEY = "cd55e690d94d4a0ab3cd1c71bd20465f"
GEMINI_API_KEY = "AIzaSyA0yyjabsJPAORwkaxHm6jd3mxXgxDDtfY"

# Gemini setup
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# it tries to answer simple question about the stocks and also get the live price of the stocks which you have
# Created a database rough so that you can integrate it with.

import sqlite3

# 📦 Create and populate a sample user portfolio database
def create_demo_portfolio():
    conn = sqlite3.connect("portfolio.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_portfolio (
            user_id TEXT,
            stock_ticker TEXT,
            buy_price REAL,
            quantity INTEGER
        )
    """)
    # Sample data
    sample_data = [
        ("user123", "AAPL", 150.0, 10),
        ("user123", "GOOGL", 270.0, 5),
        ("user123", "TSLA", 700.0, 3)
    ]
    cursor.executemany("INSERT INTO user_portfolio VALUES (?, ?, ?, ?)", sample_data)
    conn.commit()
    conn.close()

create_demo_portfolio()

import google.generativeai as genai
from datetime import datetime
import yfinance as yf

# API keys initialization

genai.configure(api_key='AIzaSyA0yyjabsJPAORwkaxHm6jd3mxXgxDDtfY')
model = genai.GenerativeModel('gemini-2.0-flash')

# Function which fetches the users stocks with the current price and sends the data in the form
# ticker buy_price, quantity, current price.

def fetch_user_portfolio_with_prices(user_id):
    conn = sqlite3.connect("portfolio.db")  # try to add the sql part her
    cursor = conn.cursor()
    cursor.execute("SELECT stock_ticker, buy_price, quantity FROM user_portfolio WHERE user_id=?", (user_id,)) # change it according to your table structure
    raw_portfolio = cursor.fetchall()
    conn.close()

    enriched_portfolio = []
    for ticker, buy_price, qty in raw_portfolio:
        stock = yf.Ticker(ticker)
        current_price = stock.info.get("regularMarketPrice", None)
        if current_price is None:
            current_price = stock.history(period="1d")["Close"].iloc[-1]  # fallback
        enriched_portfolio.append({
            "ticker": ticker,
            "buy_price": buy_price,
            "quantity": qty,
            "current_price": round(current_price, 2)
        })

    return enriched_portfolio

# Prompt engineering for the financial code and if their is protfolio then it takes the protfolio and answer your question or else no and return the prompt

def build_prompt_with_live_prices(user_question, portfolio=None):
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


# Ask the gemini the prompt passed from above

def ask_gemini(prompt):
    response = model.generate_content(prompt)
    return response.text

# It is the main where the chatbot starts.

def answer_user_question_live(user_id, question):
    portfolio = fetch_user_portfolio_with_prices(user_id)
    prompt = build_prompt_with_live_prices(question, portfolio)
    response = ask_gemini(prompt)
    return response

# sample test case where the user ask the question and the ai gives answer.

user_id = "user123"
question = "How is my portfolio?"

response = answer_user_question_live(user_id, question)
print("💬 Gemini's Answer:\n", response)

"""# Creating agents ( one for simple answer thing , one for the protfolio analysis and the third is the buy/sell/or hold )"""

# General agent for simple question and answer.

def general_qa_agent(user_question):
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
        # Post-process to ensure brevity
        answer = response.text.strip()
        if len(answer.split('\n')) > 3:  # If too long, take first part
            answer = '\n'.join(answer.split('\n')[:3])
        return answer
    except Exception as e:
        return f"Sorry, I couldn't process that. Please try again. ({str(e)})"

# Second agent that is based on the protfolio analysis which is connected to above

def portfolio_insight_agent(user_question, portfolio):
    prompt = build_prompt_with_live_prices(user_question, portfolio)
    response = model.generate_content(prompt).text
    return response.strip()



"""**Sarimanx model is build to predict the future price**"""

# Installing the header files

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# The sarimax code for future price prediction

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

    # Optional plot
    if show_plot:
        plt.figure(figsize=(10, 5))
        plt.plot(train['Close'], label="Train")
        plt.plot(test['Close'], label="Actual")
        if 'forecast' in locals():
            plt.plot(forecast, label="Forecast", linestyle="--")
        plt.title(f"{ticker} Forecast - Accuracy: {accuracy}%")
        plt.legend()
        plt.show()

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

# Example for the above thing

ticker = "TCS.NS"  # Or "TSLA", "INFY.NS", etc.
predicted_price, acc = predict_future_price_sarimax(ticker, steps=1, show_plot=True)

print(f"📈 Predicted next close price for {ticker}: ₹{predicted_price}")
print(f"✅ Forecast accuracy: {acc}%")



"""Third agent which recomends whether to buy or sell or hold stocks."""

# using the above sarimax model in the third agent

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

# trying to extract the stocks in the question asked to refine it within that thing

import re

# Utility: Extract specific ticker mentioned in the question
def extract_stock_ticker(user_question, portfolio):
    for item in portfolio:
        if item['ticker'].lower() in user_question.lower():
            return item['ticker']
    return None

# for the sentiment analysis model we are using transformer so large model

# sentiment analysis for below code to work

from transformers import pipeline
sentiment_analyzer = pipeline("sentiment-analysis")


# model trained for the sentiment analysis of the news and it also require the hugging face api
from transformers import pipeline
import requests
from newspaper import Article # Import Article from newspaper3k
import time
from datetime import datetime, timedelta

# Load Hugging Face sentiment pipeline (no need to handle tokenizer/model separately)
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def get_news_headlines_from_api(symbol, days_back=7, max_articles=5):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    url = (
        f"https://newsapi.org/v2/everything?q={symbol}&from={start_date.date()}&"
        f"to={end_date.date()}&language=en&sortBy=publishedAt&pageSize={max_articles}&apiKey={NEWS_API_KEY}"
    )

    try:
        response = requests.get(url)
        data = response.json()
        if 'articles' in data:
            return [article['title'] for article in data['articles'] if 'title' in article]
        return []
    except Exception as e:
        print(f"Error fetching news for {symbol}: {str(e)}")
        return []

def analyze_sentiment(text):
    try:
        result = sentiment_pipeline(text[:512])[0]  # Take only the first 512 chars for speed
        label = result['label']
        score_map = {
            "1 star": -1.0,
            "2 stars": -0.5,
            "3 stars": 0.0,
            "4 stars": 0.5,
            "5 stars": 1.0
        }
        return score_map.get(label.lower(), 0.0)
    except Exception as e:
        print(f"Sentiment error on: {text} -> {e}")
        return 0.0

def get_news_sentiment(symbol):
    headlines = get_news_headlines_from_api(symbol)
    if not headlines:
        return 0.0
    scores = [analyze_sentiment(h) for h in headlines]
    return sum(scores) / len(scores)

def buy_sell_hold_agent(user_question, portfolio):
    if not portfolio:
        return "❌ No portfolio data available."

    ticker_in_question = extract_stock_ticker(user_question, portfolio)
    if ticker_in_question:
        ticker_in_question = ticker_in_question.lower()

    relevant_stocks = [
        stock for stock in portfolio
        if ticker_in_question is None or stock['ticker'].lower() == ticker_in_question
    ]

    if not relevant_stocks:
        return f"❌ No matching stock found for '{ticker_in_question}' in your portfolio."

    summaries = []

    for stock in relevant_stocks:
        ticker = stock['ticker']
        buy_price = stock['buy_price']
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

def extract_stock_ticker(user_question, portfolio):
    # ✅ Hardcoded company name to ticker mapping
    company_to_ticker = {
        "apple": "AAPL",
        "microsoft": "MSFT",
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "amazon": "AMZN",
        "tesla": "TSLA",
        "meta": "META",
        "facebook": "META",
        "nvidia": "NVDA",
        "netflix": "NFLX"
    }

    question_lower = user_question.lower()

    # Step 1: Check for company names in the question
    for company_name, ticker in company_to_ticker.items():
        if company_name in question_lower:
            return ticker

    # Step 2: Check if user directly mentioned a ticker in portfolio
    portfolio_tickers = [stock['ticker'].lower() for stock in portfolio]
    for word in user_question.split():
        if word.lower() in portfolio_tickers:
            return word.upper()

    # Step 3: No match found
    return None



"""Question classifier so that the user question can be divide into three thing

1. General question
2. Question related to protfolio
3. Buy Hold sell stocks

"""



# the interface where the classifcation is used

def answer_user_query(user_question, user_id):
    question_type = classify_question(user_question)
    portfolio = fetch_user_portfolio_with_prices(user_id)

    if question_type == "general_qa":
        return general_qa_agent(user_question)
    elif question_type == "portfolio_insight":
        return portfolio_insight_agent(user_question, portfolio)
    elif question_type == "buy_sell_hold":
        return buy_sell_hold_agent(user_question, portfolio)
    else:
        return "I'm not sure how to handle that. Can you rephrase?"

"""# test case for the above thing"""

user_id = "user123"

# Try different questions:
test_questions = [
    "What is AAPI?",
    "How is my portfolio doing?",
    "Should I sell my google shares?",
    "Explain what mutual fund is.",
    "Am I making any profit?"
]

for q in test_questions:
    print(f"🧑‍💼 User: {q}")
    print(f"🤖 Bot: {answer_user_query(q, user_id)}\n{'-'*50}")
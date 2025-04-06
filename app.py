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
import mysql.connector
from mysql.connector import Error
import bcrypt
import jwt

# Load environment variables
load_dotenv()

# JWT configuration
JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key')  # Add this to your .env file
JWT_ALGORITHM = 'HS256'

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=1)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
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
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        
        # Get user from database
        cursor.execute("""
            SELECT id, username, hashed_password 
            FROM users 
            WHERE username = %s
        """, (username,))
        
        user = cursor.fetchone()
        
        if user and bcrypt.checkpw(password.encode('utf-8'), user['hashed_password'].encode('utf-8')):
            # Create token for the user
            token = create_access_token(data={"sub": username})
            st.session_state.token = token
            st.session_state.user = username
            return True
        return False
    except Exception as e:
        st.error(f"Error during login: {str(e)}")
        return False
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def register(username, email, password, investor_type):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        
        # Check if username already exists
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            return False, "Username already exists"
        
        # Check if email already exists
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            return False, "Email already exists"
        
        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # Insert new user - using hashed_password instead of password
        cursor.execute(
            """
            INSERT INTO users (username, email, hashed_password, investor_type)
            VALUES (%s, %s, %s, %s)
            """,
            (username, email, hashed_password.decode('utf-8'), investor_type)
        )
        
        conn.commit()
        
        # Get the new user's ID
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        
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
        if 'cursor' in locals():
            cursor.close()
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
        # Check API limit before making request
        if not check_api_limit():
            return None
            
        response = requests.get(
            f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={os.getenv('ALPHA_VANTAGE_API_KEY')}"
        )
        data = response.json()
        
        # Debug the API response
        print(f"AlphaVantage API Response for {symbol}:", data)
        
        # Check for various rate limit messages
        if any(key in data for key in ["Note", "Information"]):
            message = data.get("Note", data.get("Information", ""))
            if "rate limit" in message.lower() or "api call frequency" in message.lower():
                st.session_state.api_request_count = 25  # Force rate limit
                st.warning("AlphaVantage API rate limit reached. Using cached data if available.")
                return None
        
        # Increment request counter only for successful requests
        if "Time Series (Daily)" in data:
            st.session_state.api_request_count += 1
            
            # Get the most recent data point (first key in the time series)
            latest_timestamp = list(data["Time Series (Daily)"].keys())[0]
            latest_data = data["Time Series (Daily)"][latest_timestamp]
            
            return {
                "symbol": symbol,
                "price": float(latest_data["4. close"]),  # Using close price as current price
                "change": float(latest_data["4. close"]) - float(latest_data["1. open"]),
                "change_percent": ((float(latest_data["4. close"]) - float(latest_data["1. open"])) / float(latest_data["1. open"])) * 100,
                "volume": int(latest_data["5. volume"])
            }
        else:
            st.warning(f"No data available for symbol: {symbol}")
            return None
            
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return None

def get_news_sentiment(symbol):
    if not newsapi:
        st.warning("News API not configured")
        return []
    
    try:
        news = newsapi.get_everything(
            q=symbol,
            language='en',
            sort_by='publishedAt',
            page_size=5
        )
        return news['articles']
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []

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

def get_user_portfolio(username, force_refresh=False):
    try:
        # Security check - verify the requested username matches the logged-in user
        if username != st.session_state.user:
            st.error("Unauthorized access attempt")
            return []

        # Check if we can use cached data
        current_time = datetime.now()
        if (not force_refresh and 
            st.session_state.cached_portfolio is not None and 
            st.session_state.portfolio_last_updated is not None and
            (current_time - st.session_state.portfolio_last_updated).total_seconds() < 300):  # Cache for 5 minutes
            st.info("Using cached portfolio data. Last updated: " + 
                   st.session_state.portfolio_last_updated.strftime("%H:%M:%S"))
            return st.session_state.cached_portfolio

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        
        # First get the user's ID
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        if not user:
            st.error("User not found")
            return []
        
        # Get user's portfolio with current market prices using user_id
        cursor.execute("""
            SELECT p.*, u.username 
            FROM portfolio p 
            JOIN users u ON p.user_id = u.id 
            WHERE p.user_id = %s
        """, (user['id'],))
        
        portfolio = cursor.fetchall()
        
        # Get current prices for each stock
        api_limit_reached = False
        for stock in portfolio:
            stock_data = get_stock_data(stock['stock_symbol'])
            if stock_data:
                # Convert decimal to float for calculations
                avg_price = float(stock['average_buy_price'])
                current_price = stock_data['price']
                quantity = float(stock['quantity'])
                
                stock['current_price'] = current_price
                stock['profit_loss'] = (current_price - avg_price) * quantity
                stock['profit_loss_percentage'] = ((current_price - avg_price) / avg_price) * 100
            else:
                # API limit reached or error occurred
                api_limit_reached = True
                # Convert decimal to float for fallback calculations
                avg_price = float(stock['average_buy_price'])
                quantity = float(stock['quantity'])
                
                # Use the last known price from the database if available
                stock['current_price'] = avg_price
                stock['profit_loss'] = 0
                stock['profit_loss_percentage'] = 0
        
        if api_limit_reached:
            st.warning("⚠️ Some stock prices couldn't be updated due to API rate limit. Showing last known prices.")
        
        # Cache the portfolio data
        st.session_state.cached_portfolio = portfolio
        st.session_state.portfolio_last_updated = current_time
        
        return portfolio
    except Exception as e:
        st.error(f"Error fetching portfolio: {str(e)}")
        return []
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def add_stock_to_portfolio(username, stock_symbol, quantity, avg_price):
    try:
        # Security check - verify the requested username matches the logged-in user
        if username != st.session_state.user:
            return False, "Unauthorized access attempt"

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        
        # Get user ID
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        if not user:
            return False, "User not found"
        
        # Check if stock already exists in portfolio
        cursor.execute(
            "SELECT * FROM portfolio WHERE user_id = %s AND stock_symbol = %s",
            (user['id'], stock_symbol)
        )
        existing_stock = cursor.fetchone()
        
        if existing_stock:
            # Update existing stock
            cursor.execute(
                """
                UPDATE portfolio 
                SET quantity = %s, average_buy_price = %s, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = %s AND stock_symbol = %s
                """,
                (quantity, avg_price, user['id'], stock_symbol)
            )
        else:
            # Add new stock
            cursor.execute(
                """
                INSERT INTO portfolio (user_id, stock_symbol, quantity, average_buy_price)
                VALUES (%s, %s, %s, %s)
                """,
                (user['id'], stock_symbol, quantity, avg_price)
            )
        
        conn.commit()
        
        # Invalidate cache after adding/updating stock
        st.session_state.cached_portfolio = None
        st.session_state.portfolio_last_updated = None
        
        return True, "Stock added successfully"
    except Exception as e:
        return False, str(e)
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def get_user_financial_goals(username):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        
        # Get user ID
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        if not user:
            return None
        
        # Get financial goals
        cursor.execute("""
            SELECT * FROM financial_goals 
            WHERE user_id = %s 
            ORDER BY created_at DESC 
            LIMIT 1
        """, (user['id'],))
        
        return cursor.fetchone()
    except Exception as e:
        st.error(f"Error fetching financial goals: {str(e)}")
        return None
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def update_financial_goals(username, investment_amount, target_return, time_period, risk_tolerance):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        
        # Get user ID
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        if not user:
            return False, "User not found"
        
        # Check if goals exist
        cursor.execute("SELECT id FROM financial_goals WHERE user_id = %s", (user['id'],))
        existing_goals = cursor.fetchone()
        
        if existing_goals:
            # Update existing goals
            cursor.execute("""
                UPDATE financial_goals 
                SET investment_amount = %s, target_return = %s, time_period = %s, risk_tolerance = %s, created_at = CURRENT_TIMESTAMP
                WHERE user_id = %s
            """, (investment_amount, target_return, time_period, risk_tolerance, user['id']))
        else:
            # Insert new goals
            cursor.execute("""
                INSERT INTO financial_goals (user_id, investment_amount, target_return, time_period, risk_tolerance)
                VALUES (%s, %s, %s, %s, %s)
            """, (user['id'], investment_amount, target_return, time_period, risk_tolerance))
        
        conn.commit()
        return True, "Financial goals updated successfully"
    except Exception as e:
        return False, str(e)
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

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
                if login(username, password):
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")

        with tab2:
            st.subheader("Register")
            new_username = st.text_input("New Username", key="reg_username")
            new_email = st.text_input("Email", key="reg_email")
            new_password = st.text_input("New Password", type="password", key="reg_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            investor_type = st.selectbox("Investor Type", ["beginner", "amateur"])
            
            if st.button("Register"):
                if not new_username or not new_email or not new_password:
                    st.error("Please fill in all fields")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    success, message = register(new_username, new_email, new_password, investor_type)
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
            ["Dashboard", "Portfolio", "Financial Goals", "News & Alerts", "Chat", "Settings"]
        )

        if page == "Dashboard":
            st.header("Dashboard")
            
            # Get portfolio data
            portfolio = get_user_portfolio(st.session_state.user)
            
            if portfolio:
                # Calculate portfolio metrics
                total_value = sum(float(stock['current_price']) * float(stock['quantity']) for stock in portfolio)
                total_investment = sum(float(stock['average_buy_price']) * float(stock['quantity']) for stock in portfolio)
                total_profit_loss = sum(float(stock['profit_loss']) for stock in portfolio)
                
                # Calculate daily change (using the first stock's change percentage as an example)
                daily_change = portfolio[0]['profit_loss_percentage'] if portfolio else 0
                
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

                # Stock Chart
                st.subheader("Portfolio Performance")
                # Add chart implementation here
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
                total_investment = sum(float(stock['average_buy_price']) * float(stock['quantity']) for stock in portfolio)
                total_profit_loss = sum(float(stock['profit_loss']) for stock in portfolio)
                
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
                df['P/L %'] = df['profit_loss_percentage']
                
                st.dataframe(
                    df[[
                        'stock_symbol', 'quantity', 'average_buy_price', 
                        'current_price', 'Current Value', 'Profit/Loss', 'P/L %'
                    ]].style.format({
                        'average_buy_price': '${:.2f}',
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
                
                st.write(f"Last Updated: {current_goals['created_at'].strftime('%Y-%m-%d %H:%M:%S')}")
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
            stock_symbol = st.text_input("Enter Stock Symbol")
            if stock_symbol:
                news = get_news_sentiment(stock_symbol)
                for article in news:
                    with st.expander(article['title']):
                        st.write(article['description'])
                        sentiment = analyze_sentiment(article['title'])
                        st.write(f"Sentiment: {sentiment}")

        elif page == "Chat":
            st.header("AI Investment Assistant")
            
            # Display chat history
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.write("You:", message["content"])
                else:
                    st.write("Assistant:", message["content"])
            
            # Chat interface
            user_input = st.text_input("Ask me anything about investments...", key="chat_input")
            if user_input:
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                # Get AI response
                ai_response = get_ai_response(user_input)
                
                # Add AI response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                
                # Clear input and rerun to update chat
                st.rerun()

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

if __name__ == "__main__":
    main() 


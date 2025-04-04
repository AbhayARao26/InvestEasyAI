import google.generativeai as genai
import os
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-pro')

def analyze_news_sentiment(news_text):
    """
    Analyze sentiment of news text using Gemini
    """
    try:
        prompt = f"""
        Analyze the sentiment of this financial news text and return a JSON with the following structure:
        {{
            "sentiment": "positive", "negative", or "neutral",
            "confidence": float between 0 and 1,
            "key_points": list of 2-3 key points from the text
        }}
        
        Text: {news_text}
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error analyzing sentiment: {str(e)}")
        return None

def get_investment_recommendation(stock_data, news_sentiment, user_profile):
    """
    Generate investment recommendation based on stock data, news sentiment, and user profile
    """
    try:
        prompt = f"""
        Based on the following information, provide an investment recommendation:
        
        Stock Data:
        - Current Price: {stock_data.get('price')}
        - Change: {stock_data.get('change')}%
        - Volume: {stock_data.get('volume')}
        
        News Sentiment: {news_sentiment}
        
        User Profile:
        - Risk Tolerance: {user_profile.get('risk_tolerance')}
        - Investment Horizon: {user_profile.get('investment_horizon')}
        - Experience Level: {user_profile.get('experience_level')}
        
        Provide a JSON response with:
        {{
            "recommendation": "buy", "sell", or "hold",
            "confidence": float between 0 and 1,
            "reasoning": "detailed explanation",
            "risk_level": "low", "medium", or "high"
        }}
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating recommendation: {str(e)}")
        return None

def analyze_portfolio_performance(portfolio_data):
    """
    Analyze portfolio performance and provide insights
    """
    try:
        prompt = f"""
        Analyze this portfolio data and provide insights:
        
        Portfolio Holdings:
        {portfolio_data}
        
        Provide a JSON response with:
        {{
            "overall_performance": "summary of portfolio performance",
            "top_performers": "list of best performing stocks",
            "underperformers": "list of underperforming stocks",
            "recommendations": "list of specific recommendations for improvement"
        }}
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error analyzing portfolio: {str(e)}")
        return None

def predict_stock_movement(stock_data, market_conditions):
    """
    Predict stock movement based on historical data and market conditions
    """
    try:
        # Prepare features
        features = pd.DataFrame({
            'price': stock_data['price'],
            'volume': stock_data['volume'],
            'market_cap': market_conditions.get('market_cap', 0),
            'sector_performance': market_conditions.get('sector_performance', 0),
            'market_sentiment': market_conditions.get('market_sentiment', 0)
        })
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Train a simple model (in practice, you'd want to use more sophisticated models)
        model = RandomForestClassifier(n_estimators=100)
        # Note: This is a placeholder. In a real application, you'd need historical data
        # to train the model properly
        
        return {
            "prediction": "up" if np.random.random() > 0.5 else "down",
            "confidence": float(np.random.random()),
            "factors": ["market sentiment", "volume", "price momentum"]
        }
    except Exception as e:
        print(f"Error predicting stock movement: {str(e)}")
        return None

def generate_market_summary(market_data):
    """
    Generate a comprehensive market summary using AI
    """
    try:
        prompt = f"""
        Generate a comprehensive market summary based on this data:
        
        Market Data:
        {market_data}
        
        Provide a JSON response with:
        {{
            "market_overview": "general market sentiment and performance",
            "key_events": "list of significant market events",
            "sector_analysis": "performance of different sectors",
            "future_outlook": "short-term market outlook",
            "risks": "potential risks to watch out for"
        }}
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating market summary: {str(e)}")
        return None 
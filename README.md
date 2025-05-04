# InvestEasy - AI-Powered Financial Assistant

InvestEasy is a comprehensive financial assistant platform that helps users make informed investment decisions using AI and real-time market data.

## Features

- User Authentication & Onboarding
- Personalized Investment Dashboard
- AI-Powered Chatbot Assistant
- Real-time Stock Market Data
- News Sentiment Analysis
- Portfolio Management
- Investment Recommendations

## Tech Stack

- Frontend: Streamlit
- Backend: FastAPI
- Database: MySQL
- APIs: AlphaVantage, News API
- AI: Google Gemini, Custom ML Models

## Setup Instructions

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your API keys:
   ```
   ALPHA_VANTAGE_API_KEY=your_key_here
   NEWS_API_KEY=your_key_here
   GEMINI_API_KEY=your_key_here
   DATABASE_URL=mysql://user:password@localhost:3306/investeasy
   JWT_SECRET_KEY=your_secret_key_here
   ```
5. Set up MySQL database and create the required tables
6. Run the application:
   ```bash
   streamlit run app.py
   ```

## Project Structure

```
investeasy/
├── app.py                 # Main Streamlit application
├── backend/              # FastAPI backend
│   └── main.py
├── database/            # Database scripts
│   ├── schema.sql
│   └── create_test_user.sql
├── models/             # AI models and utilities
│   ├── chatbot.py
│   └── sentiment.py
├── utils/              # Utility functions
│   ├── api.py
│   └── helpers.py
└── requirements.txt    # Project dependencies
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

def predict_future_price_prophet(ticker, steps=1, show_plot=False):
    # Download 6 months of historical data
    stock = yf.Ticker(ticker)
    df = stock.history(period="6mo")

    if df.empty or 'Close' not in df:
        print(f"⚠️ Data for {ticker} not available.")
        return None, 0.0

    df = df[["Close"]].dropna().reset_index()

    if len(df) < 30:
        print(f"⚠️ Not enough data for {ticker} to train Prophet.")
        return None, 0.0

    # Prepare data for Prophet
    df = df.rename(columns={"Date": "ds", "Close": "y"})
    df['ds'] = df['ds'].dt.tz_localize(None)  # Remove timezone

    # Train/Test Split
    split_idx = int(len(df) * 0.9)
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]

    # Train Prophet model
    try:
        model = Prophet(daily_seasonality=True)
        model.fit(train)
    except Exception as e:
        print(f"❌ Prophet training failed for {ticker}: {e}")
        return None, 0.0

    # Predict future (test + steps)
    future = model.make_future_dataframe(periods=len(test) + steps)

    try:
        forecast = model.predict(future)
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        forecast_df = forecast[['ds', 'yhat']].set_index('ds')

        # Align forecast to test dates (inner join)
        test_index = pd.to_datetime(test['ds'])
        common_dates = forecast_df.index.intersection(test_index)
        test_forecast = forecast_df.loc[common_dates]
        actual_test = test.set_index('ds').loc[common_dates]

        mape = mean_absolute_percentage_error(actual_test['y'], test_forecast['yhat'])
        accuracy = round((1 - mape) * 100, 2)
    except Exception as e:
        print(f"❌ Forecasting/evaluation failed for {ticker}: {e}")
        accuracy = 0.0
        test_forecast = pd.DataFrame()

    # Plot
    if show_plot:
        plt.figure(figsize=(10, 5))
        plt.plot(train['ds'], train['y'], label="Train")
        plt.plot(test['ds'], test['y'], label="Actual")
        if not test_forecast.empty:
            plt.plot(test_forecast.index, test_forecast['yhat'], label="Forecast", linestyle="--")
        plt.title(f"{ticker} Prophet Forecast - Accuracy: {accuracy}%")
        plt.legend()
        plt.show()

    # Predict next day(s)
    try:
        future_price = forecast_df.iloc[-steps:]['yhat'].values[-1]
        predicted_price = round(future_price, 2)
    except Exception as e:
        print(f"❌ Future price prediction failed: {e}")
        predicted_price = None

    return predicted_price, accuracy


# Example
ticker = "AAPL"
predicted_price, acc = predict_future_price_prophet(ticker, steps=1, show_plot=True)
print(f"📈 Predicted next close price for {ticker}: ₹{predicted_price}")
print(f"✅ Forecast accuracy: {acc}%")

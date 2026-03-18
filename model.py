import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from data import fetch_stock_data, engineer_features, get_feature_columns

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def train_model(ticker: str, prediction_days: int = 252) -> dict:
    """
    Train a Random Forest model for a given ticker.
    Saves the model + scaler to disk and returns evaluation metrics.
    """
    print(f"Fetching data for {ticker}...")
    df = fetch_stock_data(ticker, period="10y")

    print("Engineering features...")
    df = engineer_features(df, prediction_days=prediction_days)

    features = get_feature_columns()
    X = df[features].values
    y = df["Target"].values

    # Train/test split — keep chronological order (no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    print("Training model...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    # Directional accuracy: did we predict up/down correctly?
    actual_direction = (y_test > X_test[:, 0]).astype(int)   # Close is index 0
    pred_direction = (preds > X_test[:, 0]).astype(int)
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100

    # Save model + scaler — separate file per horizon
    model_path = os.path.join(MODELS_DIR, f"{ticker}_{prediction_days}d_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"{ticker}_{prediction_days}d_scaler.pkl")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")

    metrics = {
        "ticker": ticker,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "directional_accuracy_pct": round(directional_accuracy, 2),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
    }

    print(f"Metrics: {metrics}")
    return metrics


def predict_price(ticker: str, prediction_days: int = 252) -> dict:
    """
    Load a saved model and predict the price ~1 year from now.
    Trains a new model if one doesn't exist yet.
    """
    model_path = os.path.join(MODELS_DIR, f"{ticker}_{prediction_days}d_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"{ticker}_{prediction_days}d_scaler.pkl")

    # Auto-train if model doesn't exist for this specific horizon
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"No model found for {ticker} ({prediction_days}d), training now...")
        train_model(ticker, prediction_days)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Get latest data — use fresh 5d fetch for current price to avoid stale cache
    df = fetch_stock_data(ticker, period="3y")
    df = engineer_features(df, prediction_days=prediction_days)

    features = get_feature_columns()
    latest_row = df[features].iloc[-1].values.reshape(1, -1)
    latest_scaled = scaler.transform(latest_row)

    predicted_return = model.predict(latest_scaled)[0]  # % return as decimal

    # Get truly fresh current price from a separate short-period fetch
    fresh_df = fetch_stock_data(ticker, period="5d")
    current_price = fresh_df["Close"].iloc[-1]

    # Convert predicted % return back to a price
    predicted_price = current_price * (1 + predicted_return)
    expected_return_pct = predicted_return * 100

    return {
        "ticker": ticker.upper(),
        "current_price": round(float(current_price), 2),
        "predicted_price_1yr": round(float(predicted_price), 2),
        "expected_return_pct": round(float(expected_return_pct), 2),
        "prediction_horizon_days": prediction_days,
    }


if __name__ == "__main__":
    # Quick test
    metrics = train_model("AAPL")
    print(metrics)
    result = predict_price("AAPL")
    print(result)

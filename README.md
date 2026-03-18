# 📈 Stock Price Predictor API

A REST API that uses machine learning to predict stock prices approximately **1 year into the future**, trained on real historical data from Yahoo Finance.

---

## 🚀 Features

- Pulls **real historical stock data** via `yfinance`
- Engineers **technical indicators**: Moving Averages, RSI, MACD, Volatility, Lag features
- Trains a **Random Forest Regressor** to predict price 252 trading days out
- Serves predictions via a clean **FastAPI** REST API
- Auto-trains a model if one doesn't exist for a ticker yet

---

## 🛠 Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the API
```bash
uvicorn main:app --reload
```

### 3. Open interactive docs
Visit [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📡 Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/predict?ticker=AAPL` | Predict price 1 year from now |
| POST | `/train?ticker=AAPL` | Train/retrain the model for a ticker |
| GET | `/history?ticker=AAPL&period=1y` | Get historical closing prices |
| GET | `/info?ticker=AAPL` | Get company info (sector, market cap, P/E) |

---

## 💡 Example Response — `/predict?ticker=AAPL`

```json
{
  "ticker": "AAPL",
  "current_price": 189.50,
  "predicted_price_1yr": 214.30,
  "expected_return_pct": 13.08,
  "prediction_horizon_days": 252,
  "timestamp": "2024-06-01T12:00:00"
}
```

---

## 🔧 Tech Stack

- **Python** — core language
- **FastAPI** — REST API framework
- **yfinance** — real stock data
- **pandas / numpy** — data manipulation
- **scikit-learn** — ML model (Random Forest)
- **joblib** — model serialization

---

## ⚠️ Disclaimer

Stock predictions are inherently uncertain. This project is for **educational and portfolio purposes only** and should not be used for real investment decisions.

---

## 🔮 Ideas to Extend

- Add LSTM/neural network model for comparison
- Add a `/compare?tickers=AAPL,MSFT,GOOGL` endpoint
- Deploy to Railway or Render (free hosting)
- Build a simple frontend dashboard with Chart.js
- Add a caching layer (Redis) to avoid re-fetching data
- Store predictions in a database (SQLite/PostgreSQL) to track accuracy over time

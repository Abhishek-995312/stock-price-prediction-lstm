from flask import Flask, request, render_template
import yfinance as yf
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ---- Load trained model + scaler once at startup (not per request) ----
model = load_model('lstm_model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Must match what you used while training in the notebook
LOOKBACK = 60


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form.get('ticker', '').strip().upper()

    if not ticker:
        return render_template('index.html', error="Please enter a ticker symbol.")

    try:
        # Fetch recent data live — this is what keeps predictions "current"
        # even though the model weights themselves are fixed/pre-trained.
        df = yf.download(ticker, period='6mo', progress=False)

        if df.empty or len(df) < LOOKBACK:
            return render_template(
                'index.html',
                error=f"Not enough data found for '{ticker}'. Check the symbol and try again."
            )

        # Use the last LOOKBACK closing prices as model input
        closing_prices = df['Close'].values[-LOOKBACK:].reshape(-1, 1)

        # Scale using the SAME scaler fitted during training (transform, not fit_transform)
        scaled_input = scaler.transform(closing_prices)
        X = scaled_input.reshape(1, LOOKBACK, 1)

        # Inference on the fixed, pre-trained model
        pred_scaled = model.predict(X)
        pred_price = scaler.inverse_transform(pred_scaled)[0][0]

        return render_template(
            'index.html',
            predicted_price=round(float(pred_price), 2),
            ticker=ticker
        )

    except Exception as e:
        return render_template('index.html', error=f"Something went wrong: {e}")


if __name__ == '__main__':
    app.run(debug=True)
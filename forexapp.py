from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
import talib
import plotly.graph_objects as go
from itertools import product

app = Flask(__name__)
socketio = SocketIO(app)

# Load pre-trained model
model_path = "forex_model.h5"
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    model = None

def get_forex_data(symbol, timeframe, n_candles):
    if not mt5.initialize():
        return None
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_candles)
    mt5.shutdown()
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['close'])
    df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(df['close'])
    df['VWAP'] = (df['high'] + df['low'] + df['close']) / 3
    return df

def optimize_hyperparameters():
    best_params = None
    best_loss = float('inf')
    param_grid = {
        'epochs': [50, 100, 150],
        'batch_size': [32, 64, 128],
        'learning_rate': [0.0001, 0.0005, 0.001]
    }
    for params in product(*param_grid.values()):
        # Placeholder for actual optimization
        loss = np.random.rand()
        if loss < best_loss:
            best_loss = loss
            best_params = params
    return best_params

def predict_next_close(data):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['close']])
    last_sequence = data_scaled[-60:].reshape(1, 60, 1)
    predicted_price = scaler.inverse_transform(model.predict(last_sequence))
    return predicted_price[0][0]

def execute_trade(symbol, action, lot_size=0.1):
    if not mt5.initialize():
        return False
    price = mt5.symbol_info_tick(symbol).ask if action == "buy" else mt5.symbol_info_tick(symbol).bid
    order_type = mt5.ORDER_BUY if action == "buy" else mt5.ORDER_SELL
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "deviation": 10,
        "magic": 0,
        "comment": "AI Trade",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    order_result = mt5.order_send(request)
    mt5.shutdown()
    return order_result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded."})
    
    symbol = request.json.get("symbol", "EURUSD")
    timeframe = request.json.get("timeframe", mt5.TIMEFRAME_M15)
    candles = request.json.get("candles", 5000)
    
    data = get_forex_data(symbol, timeframe, candles)
    if data is None or data.empty:
        return jsonify({"error": "Failed to retrieve data."})
    
    predicted_price = predict_next_close(data)
    socketio.emit('prediction', {'price': predicted_price})
    return jsonify({"predicted_price": predicted_price})

@app.route('/trade', methods=['POST'])
def trade():
    symbol = request.json.get("symbol", "EURUSD")
    action = request.json.get("action", "buy")
    lot_size = request.json.get("lot_size", 0.1)
    trade_result = execute_trade(symbol, action, lot_size)
    return jsonify({"trade_result": trade_result})

@app.route('/chart', methods=['GET'])
def chart():
    symbol = request.args.get("symbol", "EURUSD")
    timeframe = mt5.TIMEFRAME_M15
    candles = 100
    data = get_forex_data(symbol, timeframe, candles)
    if data is None or data.empty:
        return jsonify({"error": "Failed to retrieve data."})
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data['time'], open=data['open'], high=data['high'], low=data['low'], close=data['close'], name='Candles'))
    return fig.to_json()

@app.route('/optimize', methods=['GET'])
def optimize():
    best_params = optimize_hyperparameters()
    return jsonify({"best_params": best_params})

if __name__ == '__main__':
    socketio.run(app, debug=True)

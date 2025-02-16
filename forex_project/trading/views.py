from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.decorators import api_view, permission_classes
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import talib
from trading.tasks import execute_trade_async
import plotly.graph_objects as go
import logging

# Load the pre-trained model at startup
model = load_model("forex_model.h5")
logger = logging.getLogger(__name__)

# Helper function to get forex data
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

# Prediction function
def predict_next_close(data):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['close']])
    last_sequence = data_scaled[-60:].reshape(1, 60, 1)
    predicted_price = scaler.inverse_transform(model.predict(last_sequence))
    return predicted_price[0][0]

# API to predict next close price
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def predict(request):
    symbol = request.data.get("symbol", "EURUSD")
    timeframe = request.data.get("timeframe", mt5.TIMEFRAME_M15)
    candles = request.data.get("candles", 5000)
    
    data = get_forex_data(symbol, timeframe, candles)
    if data is None or data.empty:
        return JsonResponse({"error": "Failed to retrieve data."}, status=500)
    
    predicted_price = predict_next_close(data)
    return JsonResponse({"predicted_price": predicted_price})

# API to execute trade asynchronously
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def trade(request):
    symbol = request.data.get("symbol", "EURUSD")
    action = request.data.get("action", "buy")
    lot_size = request.data.get("lot_size", 0.1)
    
    trade_result = execute_trade_async.apply_async(args=[symbol, action, lot_size])
    return JsonResponse({"trade_result": "Trade initiated", "task_id": trade_result.id})

# Simple login view (for demo purposes, replace with a real authentication mechanism)
@api_view(['POST'])
def login(request):
    # Use JWT to create access tokens
    refresh = RefreshToken.for_user(user)  # `user` should be authenticated
    return JsonResponse({
        "access_token": str(refresh.access_token)
    })

# API for the chart (using Plotly)
@api_view(['GET'])
def chart(request):
    symbol = request.query_params.get("symbol", "EURUSD")
    timeframe = mt5.TIMEFRAME_M15
    candles = 100
    data = get_forex_data(symbol, timeframe, candles)
    if data is None or data.empty:
        return JsonResponse({"error": "Failed to retrieve data."}, status=500)
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data['time'], open=data['open'], high=data['high'], low=data['low'], close=data['close'], name='Candles'))
    return JsonResponse({"chart": fig.to_json()})

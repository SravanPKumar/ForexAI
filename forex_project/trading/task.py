from forex_project.forex_project.celery import shared_task
import MetaTrader5 as mt5

@shared_task
def execute_trade_async(symbol, action, lot_size=0.1):
    if not mt5.initialize():
        return {"error": "Failed to initialize MT5"}
    
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

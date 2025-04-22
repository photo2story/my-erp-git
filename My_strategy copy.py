# My_strategy.py


import datetime
from typing import Union, Any
import requests 
import yaml
import os, sys
import pandas_ta as ta
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Strategy_buy import strategy_buy
from Strategy_sell import strategy_sell
from get_signal import calculate_ppo_buy_sell_signals
import config

def load_asset_classification():
    try:
        file_path = os.path.join(config.STATIC_IMAGES_PATH, "results_relative_divergence.csv")
        data = pd.read_csv(file_path)
        
        def classify_volatility(vol_adj):
            if vol_adj < 1.3:
                return "Low Volatility"
            elif vol_adj < 1.7:
                return "Medium Volatility"
            else:
                return "High Volatility"
        
        data['Volatility_Class'] = data['Volatility_Adjustment'].apply(classify_volatility)
        return dict(zip(data['Ticker'], data['Volatility_Class']))
    except Exception as e:
        print(f"[WARNING] Failed to load asset classification: {e}")
        return {}

asset_classification = load_asset_classification()

def get_volatility_class(ticker):
    return asset_classification.get(ticker, "Medium Volatility")

def is_korean_stock(stock_ticker, sector):
    stock_ticker = str(stock_ticker)
    if '.K' in stock_ticker or sector == 'KRX':
        return True
    return False

def my_strategy(stock_data, option_strategy='hybrid'):
    """전략 실행 함수"""
    result = []
    portfolio_value = 0
    cash = config.INITIAL_INVESTMENT
    deposit = 0
    invested_amount = config.INITIAL_INVESTMENT
    monthly_investment = config.MONTHLY_INVESTMENT
    shares = 0
    recent_high = 0
    hedge_50_active = False
    prev_month = None
    currency = 1
    signal = ''  # 신호 초기화
    
    print(f"[DEBUG] Starting strategy: {option_strategy}")
    first_trading_day = stock_data.index[0]

    for i, row in stock_data.iterrows():
        current_date = row.name
        
        old_cash = cash
        cash, invested_amount, deposit_signal, prev_month = config.monthly_deposit(
            current_date, prev_month, monthly_investment, cash, invested_amount
        )
        if cash != old_cash:
            signal = deposit_signal + (' ' + signal if signal else '')  # 월 적립 신호 추가
            print(f"[DEBUG] Monthly deposit - Date: {current_date}, New cash: ${cash:.2f}, Invested: ${invested_amount:.2f}")

        price = row['Close'] * currency
        buy_price = price * 1.005
        sell_price = price * 0.995
        mri = row.get('MRI', 0)
        rsi_ta = row.get('RSI_14', None)
        ppo_histogram = row.get('ppo_histogram', 0)
        performance = (price - recent_high) / recent_high if recent_high > 0 else 0        
        recent_high = max(recent_high, price)
        ticker = row['Stock']
        account_balance = cash + (shares * price)

        # 매도 로직
        shares_to_sell, sell_signal, hedge_50_active = strategy_sell(
            shares=shares,
            cash=cash,
            sell_price=sell_price,
            option_strategy=option_strategy,
            mri=mri,
            hedge_50_active=hedge_50_active,
            current_date=current_date,
            rsi_ta=rsi_ta,
            ppo_histogram=ppo_histogram,
            account_balance=account_balance,
            ticker=ticker
        )
        
        if shares_to_sell > 0:
            shares -= shares_to_sell
            cash += shares_to_sell * sell_price
            # 현금 비율 추가
            cash_ratio = cash / account_balance if account_balance > 0 else 0
            signal = f"cash_{cash_ratio:.0%}_{sell_signal}" + (' ' + signal if signal else '')  # 매도 신호 추가
            portfolio_value = shares * price  # 매도 후 포트폴리오 가치 업데이트
            print(f"[DEBUG] Sell executed - Date: {current_date}, Shares: {shares}, Cash: ${cash:.2f}, Portfolio: ${portfolio_value:.2f}")

        # 포트폴리오 가치 계산
        portfolio_value = shares * price

        # 매수 로직 (매일 가능)
        shares_to_buy, buy_signal, hedge_50_active = strategy_buy(
            cash=cash,
            price=buy_price,
            option_strategy=option_strategy,
            mri=mri,
            ppo_histogram=ppo_histogram,
            hedge_50_active=hedge_50_active,
            account_balance=account_balance,
            ticker=ticker,
            rsi_ta=rsi_ta,
            portfolio_value=portfolio_value
        )
        
        if shares_to_buy > 0:
            shares += shares_to_buy
            cash -= shares_to_buy * buy_price
            # 현금 비율 추가
            cash_ratio = cash / account_balance if account_balance > 0 else 0
            signal = f"cash_{cash_ratio:.0%}_{buy_signal}" + (' ' + signal if signal else '')  # 매수 신호 추가
            portfolio_value = shares * price  # 매수 후 포트폴리오 가치 업데이트
            print(f"[DEBUG] Buy executed - Date: {current_date}, Shares: {shares}, Cash: ${cash:.2f}, Portfolio: ${portfolio_value:.2f}")

        # hedge_50_active 해제 조건 (현금 비중 관리 강화)
        if hedge_50_active and option_strategy == 'hybrid':
            cash_ratio = cash / account_balance if account_balance > 0 else 0
            if cash_ratio > 0.5 and mri < 0.45:  # 현금 50% 초과 시 해제 및 매수 유도
                hedge_50_active = False
                print(f"[DEBUG] Hedge deactivated - Date: {current_date}, Cash ratio: {cash_ratio:.2f}, MRI: {mri:.2f}")
            elif cash < monthly_investment:
                hedge_50_active = False
                print(f"[DEBUG] Hedge deactivated - Low cash: ${cash:.2f}")

        # 결과 저장 (signal은 모든 신호가 누적된 형태)
        result.append([
            current_date, price/currency, row['Open'], row['High'], row['Low'],
            row['Close'], row['Volume'], portfolio_value + cash + deposit,
            deposit, cash, portfolio_value, shares, 
            ((portfolio_value + cash + deposit) / invested_amount - 1) * 100,
            invested_amount, signal, rsi_ta,
            row.get('SMA_5', None), row.get('SMA_20', None),
            row.get('SMA_60', None), row.get('SMA_120', None),
            row.get('SMA_240', None), recent_high,
            row.get('ppo_histogram', None), row['Stock'],
            mri, row.get('MRI_Signal', 'Hold')
        ])
        
        signal = ''  # 다음 날을 위해 신호 초기화

    return pd.DataFrame(result, columns=[
        'Date', 'price', 'Open', 'High', 'Low', 'Close', 'Volume',
        'account_balance', 'deposit', 'cash', 'portfolio_value', 'shares',
        'rate', 'invested_amount', 'signal', 'rsi_ta', 'sma05_ta',
        'sma20_ta', 'sma60_ta', 'sma120_ta', 'sma240_ta', 'Recent_high',
        'ppo_histogram', 'stock_ticker', 'MRI', 'MRI_Signal'
    ])

if __name__ == "__main__":
    import pandas as pd
    import os
    import config
    
    ticker = 'TSLA'  # VOO 대신 TSLA로 테스트
    file_path = os.path.join(config.STATIC_DATA_PATH, f"data_{ticker}.csv")

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
    else:
        stock_data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        if 'Stock' not in stock_data.columns:
            stock_data['Stock'] = ticker

        # 전략 테스트
        for strategy in ['default', 'modified_mri_seasonal', 'hedge_50', 'hybrid', 'shannon', 'buy_and_hold']:
            result_df = my_strategy(stock_data, strategy)
            if result_df.empty:
                print(f"No data processed by my_strategy for {ticker} with {strategy}.")
            else:
                print(f"\n=== {strategy} Results for {ticker} ===")
                print(f"Data rows: {len(result_df)}")
                final_row = result_df.iloc[-1]
                print(f"Final Date: {final_row['Date']}")
                print(f"Final Account Balance: ${final_row['account_balance']:,.2f}")
                print(f"Total Invested: ${final_row['invested_amount']:,.2f}")
                print(f"Total Return Rate: {final_row['rate']:.2f}%")
                print(f"Final Shares: {final_row['shares']}")
                print(f"Final Cash: ${final_row['cash']:,.2f}")
                print(f"Final Portfolio Value: ${final_row['portfolio_value']:,.2f}")
                print(f"Stock Price Change: {(final_row['price'] / stock_data['Close'].iloc[0] - 1) * 100:.2f}%")
                
# python My_strategy.py
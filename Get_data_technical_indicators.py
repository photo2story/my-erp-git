# Get_data_technical_indicators.py

import pandas as pd
import numpy as np
import pandas_ta as ta

def calculate_sma(prices, periods=[5, 10, 20, 60, 120, 240]):
    """
    여러 기간의 단순이동평균 계산
    """
    result = pd.DataFrame()
    for period in periods:
        result[f'SMA_{period}'] = prices.rolling(window=period).mean().round(4)
    return result

def calculate_rsi(prices, window=14):
    """
    RSI 지표 계산
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.round(4)

def calculate_ppo(prices, short_window=12, long_window=26, signal_window=9):
    """
    PPO 지표 계산
    """
    short_ema = prices.ewm(span=short_window, adjust=False).mean()
    long_ema = prices.ewm(span=long_window, adjust=False).mean()
    ppo = ((short_ema - long_ema) / long_ema) * 100
    ppo_signal = ppo.ewm(span=signal_window, adjust=False).mean()
    ppo_histogram = ppo - ppo_signal
    return ppo.round(4), ppo_signal.round(4), ppo_histogram.round(4)

def calculate_bollinger_bands(prices, window=20, num_std_dev=2):
    """
    볼린저 밴드 계산
    """
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band.round(4), rolling_mean.round(4), lower_band.round(4)

def calculate_mfi(high, low, close, volume, length=14):
    """
    MFI(Money Flow Index) 계산
    """
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    positive_flow = pd.Series(0, index=money_flow.index)
    negative_flow = pd.Series(0, index=money_flow.index)
    
    # Calculate positive and negative money flow
    positive_flow[typical_price > typical_price.shift(1)] = money_flow[typical_price > typical_price.shift(1)]
    negative_flow[typical_price < typical_price.shift(1)] = money_flow[typical_price < typical_price.shift(1)]
    
    positive_mf = positive_flow.rolling(window=length).sum()
    negative_mf = negative_flow.rolling(window=length).sum()
    
    mfi = 100 * (positive_mf / (positive_mf + negative_mf))
    return mfi.round(4)

def calculate_all_indicators(df):
    """
    모든 기술적 지표 계산
    """
    result = df.copy()
    
    # SMA 계산
    sma_df = calculate_sma(df['Close'])
    result = pd.concat([result, sma_df], axis=1)
    
    # RSI 계산
    result['RSI_14'] = calculate_rsi(df['Close'])
    
    # PPO 계산
    ppo, ppo_signal, ppo_hist = calculate_ppo(df['Close'])
    result['PPO'] = ppo
    result['PPO_Signal'] = ppo_signal
    result['PPO_Histogram'] = ppo_hist
    
    # 볼린저 밴드 계산
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['Close'])
    result['BB_Upper'] = bb_upper
    result['BB_Middle'] = bb_middle
    result['BB_Lower'] = bb_lower
    
    # MFI 계산
    if all(col in df.columns for col in ['High', 'Low', 'Close', 'Volume']):
        result['MFI_14'] = calculate_mfi(df['High'], df['Low'], df['Close'], df['Volume'])
    
    return result 
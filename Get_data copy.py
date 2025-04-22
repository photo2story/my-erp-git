## Get_data.py

import pandas_ta as ta
import pandas as pd
import requests
import numpy as np
import FinanceDataReader as fdr
import os
import sys
import config
from config import (
    is_market_open, 
    get_us_last_trading_day, 
    is_us_market_holiday,
    is_cache_valid
)
import yfinance as yf
from datetime import datetime, timedelta
import pytz
from aiohttp import ClientSession
import asyncio
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
from fredapi import Fred
import traceback
from config import FRED_API_KEY


# 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
CSV_PATH = config.CSV_PATH

NaN = np.nan

from git_operations import move_files_to_images_folder

# 기술 지표 계산 함수들 (모듈화)
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ppo(prices, short_window=12, long_window=26, signal_window=9):
    short_ema = prices.ewm(span=short_window, adjust=False).mean()
    long_ema = prices.ewm(span=long_window, adjust=False).mean()
    ppo = ((short_ema - long_ema) / long_ema) * 100
    ppo_signal = ppo.ewm(span=signal_window, adjust=False).mean()
    ppo_histogram = ppo - ppo_signal
    return ppo, ppo_signal, ppo_histogram

def calculate_bollinger_bands(prices, window=20, num_std_dev=2):
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()

    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)

    return upper_band, rolling_mean, lower_band

def calculate_mfi(high_prices, low_prices, close_prices, volumes, length=14):
    typical_prices = (high_prices + low_prices + close_prices) / 3
    raw_money_flows = typical_prices * volumes

    positive_flows = []
    negative_flows = []

    for i in range(1, len(typical_prices)):
        if typical_prices[i] > typical_prices[i-1]:
            positive_flows.append(raw_money_flows[i])
            negative_flows.append(0)
        else:
            positive_flows.append(0)
            negative_flows.append(raw_money_flows[i])

    mfi_values = []

    for i in range(length, len(typical_prices)):
        positive_mf_sum = np.sum(positive_flows[i-length:i])
        negative_mf_sum = np.sum(negative_flows[i-length:i])

        if negative_mf_sum == 0:
            mfi = 100
        else:
            mr = positive_mf_sum / negative_mf_sum
            mfi = 100 - (100 / (1 + mr))

        mfi_values.append(mfi)

    mfi_values_full = np.empty(len(typical_prices))
    mfi_values_full[:] = np.nan
    mfi_values_full[-len(mfi_values):] = mfi_values
    return mfi_values_full

# 비동기적으로 주가 데이터를 가져오는 함수 (네트워크 효율 개선)
async def fetch_data_async(session, url):
    async with session.get(url) as response:
        if response.status != 200:
            return None
        return await response.json()

import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import yfinance as yf

import yfinance as yf
import FinanceDataReader as fdr
import pandas as pd
import numpy as np

def get_daily_market_cap(ticker, start_date, end_date):
    """
    Fetches daily market cap for a given stock ticker using Yahoo Finance and FinanceDataReader.

    Args:
        ticker (str): The stock ticker.
        start_date (str): The start date in YYYY-MM-DD format.
        end_date (str): The end date in YYYY-MM-DD format.

    Returns:
        pd.Series: A series containing the market cap data.
        str: First available date in the data.
        str: Last available date in the data.
    """
    # Download stock data using yfinance
    stock_data = yf.download(ticker, start=start_date, end=end_date).astype('float64')

    # Remove rows where 'Close' price is 0
    stock_data = stock_data[stock_data['Close'] > 0]

    if stock_data.empty:
        print(f"No data returned for {ticker}.")
        return pd.Series(dtype='float64'), None, None

    # Create a Ticker object for additional info
    ticker_obj = yf.Ticker(ticker)

    # Handling for Korean stocks using FinanceDataReader
    if '.KS' in ticker or '.KQ' in ticker:
        try:
            # Use FDR to fetch outstanding shares
            fdr_data = fdr.StockListing('KOSPI' if '.KS' in ticker else 'KOSDAQ')
            stock_row = fdr_data[fdr_data['Symbol'] == ticker.split('.')[0]]

            if not stock_row.empty:
                shares_outstanding = stock_row['Shares'].values[0]  # Outstanding shares
                stock_data['Market Cap'] = stock_data['Close'] * shares_outstanding
                print(f"Market Cap calculated for {ticker} using FDR data.")
            else:
                stock_data['Market Cap'] = np.nan
                print(f"No shares outstanding data for {ticker} in FDR.")
        except Exception as e:
            print(f"Error retrieving FDR data for {ticker}: {e}")
            stock_data['Market Cap'] = np.nan
    else:
        # Handling for Yahoo Finance data
        if 'USD' in ticker:  # Cryptocurrency
            circulating_supply = ticker_obj.info.get('circulatingSupply')
            if circulating_supply:
                stock_data['Market Cap'] = stock_data['Close'] * circulating_supply
            else:
                stock_data['Market Cap'] = np.nan
        else:  # Regular stocks
            shares_outstanding = ticker_obj.info.get('sharesOutstanding')
            if shares_outstanding:
                stock_data['Market Cap'] = stock_data['Close'] * shares_outstanding
            else:
                stock_data['Market Cap'] = np.nan

    # Forward fill NaN values in Market Cap
    stock_data['Market Cap'] = stock_data['Market Cap'].ffill()

    # Force the index to datetime format
    stock_data.index = pd.to_datetime(stock_data.index, errors='coerce')

    # Extract Market Cap as a Series
    market_cap_series = stock_data['Market Cap']

    # Get the first and last available dates
    first_available_date = market_cap_series.index[0].strftime('%Y-%m-%d') if not market_cap_series.empty else None
    last_available_date = market_cap_series.index[-1].strftime('%Y-%m-%d') if not market_cap_series.empty else None

    return market_cap_series, first_available_date, last_available_date


# 비동기적으로 주가 데이터를 가져오는 함수 (네트워크 효율 개선)
async def get_stock_data(ticker, start_date, end_date):
    """비동기적으로 주가 데이터를 가져오는 함수"""
    original_ticker = ticker
    safe_ticker = f'{ticker}.KS' if len(ticker) == 6 and ticker.isdigit() else ticker.replace('/', '-')
    folder_path = config.STATIC_DATA_PATH

    # 폴더 확인
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Directory {folder_path} not found.")
    
    file_path = os.path.join(folder_path, f'data_{original_ticker}.csv')
    existing_data = None
    
    print(f"Starting data query for {original_ticker}, Target end date: {end_date}")

    # 현재 시간, 날짜 설정 (미국 동부 시간)
    us_eastern = pytz.timezone('US/Eastern')
    current_time = datetime.now(us_eastern)
    market_close_time = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
    
    # 마지막 거래일 설정
    if is_us_market_holiday(current_time):
        print("Today is a US market holiday. Adjusting end date to last trading day.")
        last_trading_day = get_us_last_trading_day(current_time)
    else:
        if current_time < market_close_time:
            last_trading_day = get_us_last_trading_day(current_time.date() - timedelta(days=1))
        else:
            last_trading_day = get_us_last_trading_day(current_time.date())
    
    end_date = last_trading_day.strftime('%Y-%m-%d')
    print(f"Adjusted end date: {end_date}")

    # 캐시된 파일 검증 및 로드
    if os.path.exists(file_path):
        try:
            existing_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            if existing_data.empty or existing_data.index.isna().all():
                print("Cached data is empty or invalid; fetching new data.")
                existing_data = None
            else:
                cached_start = existing_data.index.min().strftime('%Y-%m-%d')
                cached_end = existing_data.index.max().strftime('%Y-%m-%d')
                print(f"Cached data range: {cached_start} to {cached_end}, Required range: {start_date} to {end_date}")
                
                # 캐시가 최신 상태가 아닌 경우에만 새로운 데이터를 가져옴
                if (cached_end < end_date) or (cached_start > start_date):
                    print("Cached data is outdated or incomplete; fetching additional data.")
                    name = safe_ticker
                    ticker_obj = yf.Ticker(name)
                    
                    # 하루 더 여유를 두고 데이터 가져오기
                    next_day = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    
                    df = ticker_obj.history(
                        start=start_date,
                        end=next_day,
                        interval='1d',
                        auto_adjust=True,
                        prepost=True
                    )
                    
                    if df.empty:
                        print(f"No data available for {ticker}")
                        return pd.DataFrame(), None, None
                    
                    # 시간대 처리
                    df.index = df.index.tz_localize(None)
                    
                    print(f"Retrieved data for {ticker}: {len(df)} rows")
                    print(f"Date range: {df.index.min()} to {df.index.max()}")
                    
                    # 데이터 처리 및 저장
                    processed_data = await process_data(df, original_ticker)
                    processed_data.to_csv(file_path, na_rep='NaN')
                    
                    return processed_data, df.index[0].strftime('%Y-%m-%d'), df.index[-1].strftime('%Y-%m-%d')
                else:
                    print("Using cached data. No additional data fetch required.")
                    return existing_data, cached_start, cached_end
        except Exception as e:
            print(f"Error reading cached data for {original_ticker}: {e}")
            existing_data = None

    # 새 데이터 가져오기 (캐시가 없거나 유효하지 않은 경우)
    try:
        name = safe_ticker
        ticker_obj = yf.Ticker(name)
        
        # GPT-USD의 경우 시작일을 2023년부터로 설정
        if ticker == 'GPT-USD':
            start_date = '2023-01-01'
        
        df = ticker_obj.history(
            start=start_date,
            end=end_date,
            interval='1d',
            auto_adjust=True
        )
        print(f"Retrieved data for {ticker}: {len(df)} rows")
        
        if df.empty:
            print(f"No data found for {original_ticker}.")
            return pd.DataFrame(), None, None

        # 데이터 처리
        df.index = pd.to_datetime(df.index, errors='coerce').tz_localize(None)
        
        try:
            start_date = pd.Timestamp(start_date)
            end_date = pd.Timestamp(end_date)
            filtered_df = df[(df.index >= start_date) & (df.index <= end_date)]
            print(f"Filtered data range: {filtered_df.index[0].strftime('%Y-%m-%d')} to {filtered_df.index[-1].strftime('%Y-%m-%d')}")
            
            # 데이터 처리 및 저장
            processed_data = await process_data(filtered_df, original_ticker)
            processed_data.to_csv(file_path, na_rep='NaN')
            
            return processed_data, filtered_df.index[0].strftime('%Y-%m-%d'), filtered_df.index[-1].strftime('%Y-%m-%d')
        
        except Exception as e:
            print(f"Error processing data: {e}")
            return pd.DataFrame(), None, None

    except Exception as e:
        print(f"Error fetching data for {original_ticker}: {e}")
        return pd.DataFrame(), None, None


# VOO 데이터를 저장할 전역 변수 추가
voo_mri_data = None

async def fetch_market_risk_indicator(start_date, end_date, stock_index):
    """
    VIX, 금리, 장기채, 단기채 데이터를 활용하여 일별 Market Risk Indicator (MRI)를 계산.
    """
    global voo_mri_data
    
    try:
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        start_date = max(start_date, pd.Timestamp('2010-01-01'))

        cache_dir = config.STATIC_DATA_PATH
        mri_cache_file = os.path.join(cache_dir, 'market_risk_indicator.csv')

        # 캐시 파일 검증 및 업데이트
        if os.path.exists(mri_cache_file):
            try:
                mri_data = pd.read_csv(mri_cache_file, index_col=0, parse_dates=True)
                cached_start = mri_data.index.min()
                cached_end = mri_data.index.max()
                
                print(f"[DEBUG] Cached MRI data: {cached_start} to {cached_end}")
                print(f"[DEBUG] Required MRI data: {start_date} to {end_date}")
                
                # 캐시 데이터가 필요한 기간을 모두 포함하는지 확인
                if cached_start <= start_date and cached_end >= end_date:
                    print("[DEBUG] Using existing cached MRI data")
                    mri_daily = mri_data['MRI'].reindex(stock_index).ffill()
                    signal = np.where(mri_daily > 0.7, 'Sell',
                                    np.where(mri_daily < 0.3, 'Buy', 'Hold'))
                    return {
                        'mri': mri_daily.round(4),
                        'signal': signal
                    }
                else:
                    print("[DEBUG] Cache needs update - fetching new data")
            except Exception as e:
                print(f"[ERROR] Error reading cached MRI data: {e}")
                print("[DEBUG] Fetching new MRI data")

        # 새로운 MRI 데이터 계산
        print(f"[DEBUG] Calculating new MRI data from {start_date} to {end_date}")
        fred = Fred(api_key=FRED_API_KEY)
        
        # FRED API 데이터 가져오기
        series_data = {}
        for series_id in ['VIXCLS', 'FEDFUNDS', 'DGS10', 'DGS3MO']:
            try:
                series = fred.get_series(
                    series_id,
                    observation_start=start_date.strftime('%Y-%m-%d'),
                    observation_end=end_date.strftime('%Y-%m-%d')
                )
                if series is None or series.empty:
                    print(f"[WARNING] No data available for {series_id}")
                    return {
                        'mri': pd.Series(index=stock_index, data=np.nan),
                        'signal': np.array(['Hold'] * len(stock_index))
                    }
                series_data[series_id] = series
            except Exception as e:
                print(f"[ERROR] Error fetching {series_id} from FRED: {e}")
                return {
                    'mri': pd.Series(index=stock_index, data=np.nan),
                    'signal': np.array(['Hold'] * len(stock_index))
                }

        # DataFrame 생성 및 데이터 처리
        mri_data = pd.DataFrame({
            'VIX': series_data.get('VIXCLS'),
            'FEDFUNDS': series_data.get('FEDFUNDS'),
            'DGS10': series_data.get('DGS10'),
            'DGS3MO': series_data.get('DGS3MO')
        })
        
        if mri_data.empty:
            print("[WARNING] No data received from FRED API")
            return {
                'mri': pd.Series(index=stock_index, data=np.nan),
                'signal': np.array(['Hold'] * len(stock_index))
            }

        # 결측치 처리
        mri_data = mri_data.ffill().interpolate(method='linear')
        mri_data['Spread'] = mri_data['DGS10'] - mri_data['DGS3MO']

        # 정규화 및 MRI 계산
        for col in ['VIX', 'FEDFUNDS', 'Spread']:
            min_val = mri_data[col].min()
            max_val = mri_data[col].max()
            if max_val > min_val:
                mri_data[f'{col}_norm'] = (mri_data[col] - min_val) / (max_val - min_val)
            else:
                mri_data[f'{col}_norm'] = 0

        mri_data['MRI'] = (
            0.4 * mri_data['VIX_norm'] +
            0.3 * mri_data['FEDFUNDS_norm'] +
            0.3 * mri_data['Spread_norm']
        )

        print(f"[DEBUG] New MRI data range: {mri_data.index.min()} to {mri_data.index.max()}")
        print(f"[DEBUG] Sample MRI values:\n{mri_data[['MRI']].tail()}")

        # 캐시 저장
        mri_data.index.name = 'Date'  # 인덱스 이름을 'Date'로 설정
        mri_data.to_csv(mri_cache_file, na_rep='NaN')
        print(f"[DEBUG] Saved updated MRI data to cache: {mri_cache_file}")

        # 결과 반환
        mri_daily = mri_data['MRI'].reindex(stock_index).ffill()
        signal = np.where(mri_daily > 0.7, 'Sell',
                         np.where(mri_daily < 0.3, 'Buy', 'Hold'))

        return {
            'mri': mri_daily.round(4),
            'signal': signal
        }

    except Exception as e:
        print(f"[ERROR] Error calculating Market Risk Indicator: {str(e)}")
        traceback.print_exc()
        return {
            'mri': pd.Series(index=stock_index, data=np.nan),
            'signal': np.array(['Hold'] * len(stock_index))
        }
        
# process_data 함수를 async로 변경
async def process_data(stock_data, ticker):
    """데이터 처리 함수"""
    global voo_mri_data

    print(f"[DEBUG] Processing data for {ticker}")
    print(f"[DEBUG] Data range: {stock_data.index[0]} to {stock_data.index[-1]}")

    # VOO 데이터 처리 순서 보장
    if ticker != 'VOO':
        print(f"[DEBUG] Checking VOO MRI data for {ticker}")
        try:
            # VOO MRI 데이터 직접 로드
            mri_cache_file = os.path.join(config.STATIC_DATA_PATH, 'market_risk_indicator.csv')
            if os.path.exists(mri_cache_file):
                voo_mri_data = pd.read_csv(mri_cache_file, index_col=0, parse_dates=True)
                print(f"[DEBUG] Loaded VOO MRI data from cache: {len(voo_mri_data)} rows")
            else:
                print("[DEBUG] VOO MRI cache file not found")
                voo_data, _, _ = await get_stock_data('VOO', 
                    stock_data.index[0].strftime('%Y-%m-%d'),
                    stock_data.index[-1].strftime('%Y-%m-%d')
                )
                if not voo_data.empty:
                    await process_data(voo_data, 'VOO')
        except Exception as e:
            print(f"[ERROR] Error processing VOO data: {e}")

    if len(stock_data) < 20:
        print(f"[WARNING] Not enough data for {ticker}. Minimum 20 data points required.")
        return stock_data

    # 중복된 날짜 제거
    stock_data = stock_data[~stock_data.index.duplicated(keep='first')]

    # Close 값 처리
    stock_data['Close'] = stock_data['Close'].ffill()
    stock_data['Close'] = stock_data['Close'].bfill()

    # 기술적 지표 계산
    stock_data.loc[:, 'RSI_14'] = calculate_rsi(stock_data['Close'], window=14)
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(stock_data['Close'])
    stock_data.loc[:, 'bb_upper_ta'] = bb_upper.round(4)
    stock_data.loc[:, 'bb_middle_ta'] = bb_middle.round(4)
    stock_data.loc[:, 'bb_lower_ta'] = bb_lower.round(4)

    # Aroon, MFI
    stock_data.ta.aroon(length=25, append=True)
    stock_data.loc[:, 'MFI_14'] = calculate_mfi(
        stock_data['High'].values, 
        stock_data['Low'].values, 
        stock_data['Close'].values, 
        stock_data['Volume'].values, 
        length=14
    ).round(4)

    # SMA (이동평균선)
    sma_periods = [5, 10, 20, 60, 120, 240]
    for period in sma_periods:
        stock_data.ta.sma(close='Close', length=period, append=True)
        sma_column = f'SMA_{period}'
        if sma_column in stock_data.columns:
            stock_data[sma_column] = stock_data[sma_column].round(4)

    # Stochastic
    stock_data.ta.stoch(high='High', low='Low', k=20, d=10, append=True)
    stock_data.ta.stoch(high='High', low='Low', k=14, d=3, append=True)
    stock_data['STOCHk_20_10_3'] = stock_data['STOCHk_20_10_3'].round(4)
    stock_data['STOCHd_20_10_3'] = stock_data['STOCHd_20_10_3'].round(4)

    # PPO
    try:
        ppo, ppo_signal, ppo_histogram = calculate_ppo(stock_data['Close'])
        stock_data.loc[:, 'ppo'] = ppo.round(4)
        stock_data.loc[:, 'ppo_signal'] = ppo_signal.round(4)
        stock_data.loc[:, 'ppo_histogram'] = ppo_histogram.round(4)
    except Exception as e:
        print(f"[ERROR] Error calculating PPO for {ticker}: {e}")
        stock_data.loc[:, 'ppo'] = np.nan
        stock_data.loc[:, 'ppo_signal'] = np.nan
        stock_data.loc[:, 'ppo_histogram'] = np.nan

    # Stock 정보 추가
    stock_data['Stock'] = str(ticker)
    stock_data['Stock'] = stock_data['Stock'].astype('object')

    # Market Cap 처리
    market_cap_data, first_available_date, last_available_date = get_daily_market_cap(
        ticker, 
        stock_data.index[0], 
        stock_data.index[-1]
    )
    market_cap_data = market_cap_data.round(4)

    # Market Cap 데이터 길이 맞춤
    market_cap_data.index = pd.to_datetime(market_cap_data.index, errors='coerce')
    if len(market_cap_data) != len(stock_data):
        market_cap_data = market_cap_data.reindex(stock_data.index, method='ffill')

    stock_data['Market Cap'] = market_cap_data.values

    # MRI 데이터 처리
    try:
        print(f"[DEBUG] Fetching MRI data for {ticker}")
        mri_data = await fetch_market_risk_indicator(
            max(stock_data.index[0], pd.Timestamp('2010-01-01')),
            stock_data.index[-1],
            stock_data.index
        )
        
        if mri_data and not mri_data['mri'].isna().all():
            stock_data['MRI'] = mri_data['mri']
            stock_data['MRI_Signal'] = mri_data['signal']
            print(f"[DEBUG] Successfully added MRI data for {ticker}")
            print(f"[DEBUG] MRI data range: {stock_data['MRI'].min():.4f} to {stock_data['MRI'].max():.4f}")
            
            if ticker == 'VOO':
                voo_mri_data = mri_data
                print("[DEBUG] Stored VOO's MRI data in memory")
        else:
            print(f"[WARNING] No valid MRI data for {ticker}")
            stock_data['MRI'] = np.nan
            stock_data['MRI_Signal'] = 'Hold'
            
    except Exception as e:
        print(f"[ERROR] Error adding MRI data for {ticker}: {str(e)}")
        stock_data['MRI'] = np.nan
        stock_data['MRI_Signal'] = 'Hold'

    return stock_data

# get_post_market_prices 함수 수정
async def get_post_market_prices(ticker):
    try:
        print(f"[DEBUG] Fetching current price for {ticker}")
        
        # BTC-USD와 같은 암호화폐 처리
        if '-USD' in ticker:
            yf_ticker = ticker
        else:
            yf_ticker = ticker.upper()
            
        ticker_obj = yf.Ticker(yf_ticker)
        
        # 종가 정보 가져오기
        prev_close = ticker_obj.info.get('previousClose')
        if prev_close is None:
            history = ticker_obj.history(period='2d')
            if not history.empty and len(history) > 1:
                prev_close = history['Close'].iloc[-2]
            else:
                prev_close = None

        # 현재가 가져오기
        current_price = ticker_obj.info.get('regularMarketPrice')
        if current_price is None:
            current_price = ticker_obj.history(period='1d')['Close'].iloc[-1]

        # 변동률 계산
        change = 0.0
        if prev_close and current_price:
            change = ((current_price - prev_close) / prev_close) * 100

        print(f"[DEBUG] Current price for {yf_ticker}: ${current_price:.2f} (Δ{change:.2f}%)")
        
        #소수2자리로 변환
        change = round(change, 2)

        return {
            'price': current_price,
            'previousClose': prev_close,
            'change': change #소수2자리
        }

    except Exception as e:
        print(f"Error fetching prices for {ticker}: {e}")
        return None

# 테스트 함수
async def test_fetch_and_process_stock_data():
    tickers = ['457480.KS']
    start_date = '2015-01-02'
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    us_eastern = pytz.timezone('US/Eastern')
    
    print(f"\nTest parameters:")
    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")
    print(f"Current time (EST): {datetime.now(us_eastern).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Is market open: {is_market_open()}")
    print(f"Is holiday: {is_us_market_holiday(datetime.now(us_eastern))}")
    
    last_trading_day = get_us_last_trading_day(datetime.now())
    print(f"Last trading day: {last_trading_day.strftime('%Y-%m-%d')}")

    async with ClientSession() as session:
        for ticker in tickers:
            try:
                stock_data, first_date, last_date = await get_stock_data(ticker, start_date, end_date)
                if stock_data.empty:
                    print(f"No data for {ticker}")
                    continue

                print(f"\nData fetched for {ticker}:")
                print(f"Total rows: {len(stock_data)}")
                
                if first_date and last_date:
                    print(f"Date range: {first_date} to {last_date}")
                    
                    # MRI 데이터 계산 또는 가져오기
                    try:
                        mri_data = await fetch_market_risk_indicator(
                            max(stock_data.index[0], pd.Timestamp('2010-01-01')),
                            stock_data.index[-1],
                            stock_data.index
                        )
                        if mri_data:
                            stock_data['MRI'] = mri_data['mri']
                            stock_data['MRI_Signal'] = mri_data['signal']
                            
                            # VOO인 경우 전역 변수에 저장
                            if ticker == 'VOO':
                                global voo_mri_data
                                voo_mri_data = mri_data
                                print("Stored VOO's MRI data in memory")
                        
                        print(f"\nLast few dates in data:")
                        last_rows = stock_data.tail()
                        print(f"Index (Date), Close price, MRI, MRI_Signal:")
                        for idx, row in last_rows.iterrows():
                            print(f"{idx.strftime('%Y-%m-%d')}: Close={row['Close']:.2f}, MRI={row['MRI']:.4f}, MRI_Signal={row['MRI_Signal']}")
                    except Exception as e:
                        print(f"Error calculating MRI for {ticker}: {str(e)}")
                        traceback.print_exc()
            except Exception as e:
                print(f"Error processing {ticker}: {str(e)}")
                traceback.print_exc()

if __name__ == "__main__":
    import asyncio
    
    async def update_mri_data():
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(years=5)
        stock_index = pd.date_range(start=start_date, end=end_date, freq='D')
        
        print("[DEBUG] Updating MRI data...")
        mri_data = await fetch_market_risk_indicator(start_date, end_date, stock_index)
        print("[DEBUG] MRI data update completed")
        
    asyncio.run(update_mri_data())

## python Get_data.py    

# if __name__ == "__main__":
#     asyncio.run(test_fetch_and_process_stock_data())
#     ticker = '457480.KS'
#     asyncio.run(get_post_market_prices(ticker))
#     ticker = 'SHIB-USD'
#     data = yf.download(ticker, start='2020-01-01', end='2024-12-31')
#     print(data)
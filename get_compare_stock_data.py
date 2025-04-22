# get_compare_stock_data.py

import os
import sys
import pandas as pd
import numpy as np
import requests
import io
import asyncio
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
import config_assets  # 섹터 정보 가져오기 위해 추가
from get_ticker import get_stock_info
from git_operations import move_files_to_images_folder

GITHUB_RAW_BASE_URL = config.STATIC_IMAGES_PATH
folder_path = config.STATIC_IMAGES_PATH
alpha_path = config.STATIC_DATA_PATH

def fetch_csv(ticker):
    file_path = os.path.join(GITHUB_RAW_BASE_URL, f"result_VOO_{ticker}.csv")
    
    # 로컬 파일 존재 여부 확인
    if not os.path.exists(file_path):
        print(f"No data file found for {ticker}. Skipping...")
        return None  # 파일이 없을 경우 None 반환

    try:
        # 로컬 파일을 직접 읽음
        df = pd.read_csv(file_path)
        # print(df.head())
        return df
    
    except pd.errors.EmptyDataError:
        print(f"No data found in the file for {ticker}.")
        return None
    except Exception as e:
        print(f"Failed to load data for {ticker}: {e}")
        return None

def get_sector(ticker):
    """ config_asset에서 티커의 섹터를 찾고, 찾지 못하면 Yahoo Finance에서 가져옵니다. """
    # 먼저 config_asset에서 섹터를 찾습니다.
    for sector, tickers in config_assets.STOCKS.items():
        if ticker in tickers:
            return sector

    # config_asset에서 찾지 못한 경우 Yahoo Finance에서 섹터 정보를 가져옵니다.
    stock_info = get_stock_info(ticker)
    sector_from_yahoo = stock_info.get('Sector')
    
    if sector_from_yahoo:
        print(f"Sector for {ticker} found via Yahoo Finance: {sector_from_yahoo}")
        return sector_from_yahoo
    else:
        print(f"No sector data found for {ticker}")
        return 'Unknown'  # 최종적으로도 없을 경우 'Unknown' 반환


def calculate_volatility_adjustment(df):
    df['Relative_Divergence'] = df['Relative_Divergence'].fillna(0)
    window_size = min(480, len(df))
    df[f'Mean_Relative_Divergence_{window_size}'] = df['Relative_Divergence'].rolling(window=window_size, min_periods=1).mean()
    df[f'Percentile_85_Relative_Divergence_{window_size}'] = df['Relative_Divergence'].rolling(window=window_size, min_periods=1).quantile(0.9)
    df['Difference_Percentile_vs_Mean'] = df[f'Percentile_85_Relative_Divergence_{window_size}'] - df[f'Mean_Relative_Divergence_{window_size}']
    df['Volatility_Adjustment'] = (
        df[f'Percentile_85_Relative_Divergence_{window_size}'] / df[f'Mean_Relative_Divergence_{window_size}']
    ).clip(upper=10, lower=0.1)
    df['Volatility_Adjustment'] = df['Volatility_Adjustment'].ffill().fillna(1.0).round(2)
    return df

def calculate_expected_return(df, ticker):
    # Rolling_Divergence 계산 추가
    window_size = min(480, len(df))
    df['Rolling_Divergence'] = df['Divergence'].rolling(window=window_size, min_periods=1).mean()
    df['Rolling_Divergence'] = df['Rolling_Divergence'].ffill().fillna(0)

    # Expected Return 계산
    epsilon = 1e-2  # 최소값 설정
    df['Expected_Return'] = df['Rolling_Divergence'] / (df['Relative_Divergence'] / 100).clip(lower=epsilon)
    df['Expected_Return'] = df['Expected_Return'].fillna(0).round(2)
    
    # S&P 500 성과보다 낮은 경우 기대 수익률을 0으로 설정
    df.loc[df[f'rate_{ticker}_5D'] < df['rate_VOO_20D'], 'Expected_Return'] = 0
    
    return df

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

def calculate_relative_strength_features(df, ticker):
    """상대강도 관 특성들을 계산하는 함수"""
    # 기본 상대강도 (S&P 500 대비)
    df['RS_vs_Market'] = df[f'rate_{ticker}_5D'] / df['rate_VOO_20D']
    
    # 상대강도 변화율 (20일)
    df['RS_Change_20D'] = df['RS_vs_Market'].pct_change(20)
    
    # 상대강도 모멘텀 (상대강도의 이동평균 대비)
    df['RS_Momentum'] = df['RS_vs_Market'] / df['RS_vs_Market'].rolling(60).mean()
    
    # 상대강도 변동성
    df['RS_Volatility'] = df['RS_vs_Market'].rolling(20).std()
    
    # 상대강도 추세 (현재 상대강도가 60일 이동평균보다 높은지)
    df['RS_Trend'] = (df['RS_vs_Market'] > df['RS_vs_Market'].rolling(60).mean()).astype(int)
    
    return df

def calculate_growth_adjustment(relative_divergence):
    """상대 디벌전스에 따른 성장 조정 계수 계산 (연속적 지수감소)"""
    # 기본 파라미터
    base = 1.0  # 초기값 (상대 디벌전스가 0일 때)
    decay_rate = 0.015  # 0.035에서 0.015 감소율 낮춤 (더 완만한 감소)
    
    # 지수감소 함수: base * exp(-decay_rate * x)
    adjustment = base * np.exp(-decay_rate * relative_divergence)
    
    # 최소값도 상향 조정
    min_adjustment = 0.2  # 0.1에서 0.2로 상향
    
    return max(adjustment, min_adjustment)

def calculate_polynomial_features(df):
    """롤링값의 다항식 특성을 계산하는 함수"""
    # 480일 롤링값의 추세를 다항식으로 피팅
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['Rolling_Divergence'].values.reshape(-1, 1)
    
    # 4차 다항식으로 변경 (더 복잡한 패턴 포착)
    poly_model = make_pipeline(
        PolynomialFeatures(degree=4),
        LinearRegression()
    )
    poly_model.fit(X, y)
    
    # 다항식 예측값 계산
    df['Poly_Trend'] = poly_model.predict(X)
    
    # 실제값과 예측값의 차이 (잔차)
    df['Poly_Residual'] = df['Rolling_Divergence'] - df['Poly_Trend']
    
    # 추세 가속도 (2차 미분)
    df['Trend_Acceleration'] = df['Poly_Trend'].diff().diff()
    
    # 성장 조정 계수 추가
    df['Growth_Adjustment'] = df['Relative_Divergence'].apply(calculate_growth_adjustment)
    
    # 조정된 추세
    df['Adjusted_Trend'] = df['Poly_Trend'] * df['Growth_Adjustment']
    
    return df

def prepare_training_data(df, ticker):
    """학습 데이터를 준비하는 함수"""
    # 다항식 특성 추가
    df = calculate_polynomial_features(df)
    
    # 상대강도 특성 추가
    df = calculate_relative_strength_features(df, ticker)
    
    # 특성 선택
    features = pd.DataFrame()
    features['Rolling_Divergence'] = df['Rolling_Divergence']
    features['Relative_Divergence'] = df['Relative_Divergence'] / 100.0
    features['Poly_Trend'] = df['Poly_Trend']
    features['Adjusted_Trend'] = df['Adjusted_Trend']  # 조정된 추세 추가
    features['Growth_Adjustment'] = df['Growth_Adjustment']  # 성장 조정 계수 추가
    features['Poly_Residual'] = df['Poly_Residual']
    features['Trend_Acceleration'] = df['Trend_Acceleration']
    features['RS_vs_Market'] = df['RS_vs_Market']
    features['RS_Momentum'] = df['RS_Momentum']
    features['RS_Volatility'] = df['RS_Volatility']
    
    target = df['Expected_Return']
    
    # 결측값 처리
    features = features.fillna(0)
    target = target.fillna(0)
    
    # 학습/검증 이터 분리
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_dynamic_model(X_train, y_train):
    # 무한대 값이나 NaN 값 처리
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_train = X_train.fillna(0)

    y_train = y_train.replace([np.inf, -np.inf], np.nan)
    y_train = y_train.fillna(0)

    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gbr.fit(X_train, y_train)
    return gbr

def predict_relative_max(df, model):
    # Simulate input for Relative_Divergence=100
    input_data = df[['Rolling_Divergence']].copy()
    input_data['Relative_Divergence'] = 100  # Fix Relative Divergence at max value

    # Predict Dynamic Expected Return
    predictions = model.predict(input_data)
    df['Dynamic_Expected_Return'] = predictions.round(2)

    return df

def calculate_volatility(df, ticker):
    """종가(Close) 기준 연간 표준편차와 rate 수익률 사용"""
    try:
        # VOO 데이터 가져오기
        voo_file_path = os.path.join(GITHUB_RAW_BASE_URL, f"result_VOO_VOO.csv")
        if os.path.exists(voo_file_path):
            voo_df = pd.read_csv(voo_file_path)
            voo_df['Date'] = pd.to_datetime(voo_df['Date'])
            voo_df = voo_df.set_index('Date')
        else:
            return 1.0
        
        # 날짜를 인덱스로 설정
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        
        # 종가 기준 연간 표준편차 계산
        close_returns = df['Close'].pct_change()
        annual_std = close_returns.std() * np.sqrt(252)
        
        # rate 값 사용 (480일 롤링 평균)
        rolling_window = 480
        rolling_rate = df[f'rate_{ticker}_5D'].rolling(window=rolling_window).mean()
        voo_rolling = df['rate_VOO_20D'].rolling(window=rolling_window).mean()
        
        return annual_std
        
    except Exception as e:
        return 1.0

def apply_dynamic_expected_return(df, model, ticker):
    """Dynamic Expected Return 계산"""
    try:
        # 다항식 특성과 상대강도 특성 추가
        df = calculate_polynomial_features(df)
        df = calculate_relative_strength_features(df, ticker)
        
        # 종가 기준 변동성 계산 (참고용)
        volatility_ratio = calculate_volatility(df, ticker)
        
        # 480일 롤링 평균 계산
        rolling_window = 480
        base_predictions = df['Rolling_Divergence'].rolling(window=rolling_window).mean()
        
        # 마지막 값을 Relative_Divergence로 기
        base_predictions = base_predictions / (df['Relative_Divergence'] / 100).clip(lower=0.01)
        base_predictions = base_predictions.round(2)
        
        # Growth Adjustment 그대로 사용
        adjusted_growth = df['Growth_Adjustment']
        
        # 최종 Dynamic Expected Return 계산
        df['Dynamic_Expected_Return'] = base_predictions * adjusted_growth
        
        # 디버깅을 위한 출력
        # print("\nGrowth Adjustment Analysis:")
        # print(f"Volatility Ratio: {volatility_ratio:.2f}")
        # print(f"Base predictions (last 5): \n{base_predictions[-5:]}")
        # print(f"Growth_Adjustment (last 5): \n{df['Growth_Adjustment'].tail()}")
        # print(f"Final Dynamic_Expected_Return (last 5): \n{df['Dynamic_Expected_Return'].tail()}\n")
        
        return df
        
    except Exception as e:
        print(f"Error in apply_dynamic_expected_return: {str(e)}")
        raise

def get_closest_business_day(df, target_date):
    """주어 날짜에서 가장 가까운 영업일을 찾는 함수"""
    df_dates = pd.to_datetime(df['Date'])
    target_date = pd.to_datetime(target_date)
    
    # 해당 날짜보다 작거나 같은 가장 최근 영업일 찾기
    closest_date = df_dates[df_dates <= target_date].max()
    
    if pd.isnull(closest_date):
        # 만약 이전 영업일이 없다면, 다음 영업일 찾기
        closest_date = df_dates[df_dates > target_date].min()
    
    return closest_date

def calculate_delta_divergence_periods(ticker):
    # 기간 정의 (월 단위로 통일)
    periods = {
        '1m': 1,
        '3m': 3,
        '6m': 6,
        '1y': 12,
        '2y': 24,
        '3y': 36,
        '5y': 60,
        '7y': 84,
        'max': None  # max 기간 추가
    }

    folder_path = config.STATIC_IMAGES_PATH
    data_file = os.path.join(folder_path, f'result_VOO_{ticker}.csv')

    result = {}
    initial_investment = config.INITIAL_INVESTMENT
    monthly_investment = config.MONTHLY_INVESTMENT

    try:
        df = pd.read_csv(data_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        first_date = df['Date'].min()
        last_date = df['Date'].max()
        # print(f"\nData range: {first_date.strftime('%Y-%m')} to {last_date.strftime('%Y-%m-%d')}")

        for period_name, months in periods.items():
            try:
                if period_name == 'max':
                    period_data = df.copy()  # 전체 데이터 사용
                else:
                    period_start = last_date - pd.DateOffset(months=months)
                    period_data = df[df['Date'] >= period_start].copy()

                if not period_data.empty:
                    # 매수 수수료 0.5% 적용
                    initial_price = float(period_data['Close'].iloc[0]) * 1.005
                    initial_voo_price = float(period_data['Close_vs'].iloc[0]) * 1.005
                    
                    total_shares = initial_investment / initial_price
                    total_voo_shares = initial_investment / initial_voo_price
                    total_investment = initial_investment

                    current_month = period_data['Date'].iloc[0].month

                    for _, row in period_data.iterrows():
                        if row['Date'].month != current_month:
                            current_month = row['Date'].month
                            monthly_price = float(row['Close']) * 1.005
                            monthly_voo_price = float(row['Close_vs']) * 1.005
                            
                            total_shares += monthly_investment / monthly_price
                            total_voo_shares += monthly_investment / monthly_voo_price
                            total_investment += monthly_investment

                    final_price = float(period_data['Close'].iloc[-1])
                    final_voo_price = float(period_data['Close_vs'].iloc[-1])
                    
                    final_value = total_shares * final_price
                    final_voo_value = total_voo_shares * final_voo_price

                    ticker_return = ((final_value - total_investment) / total_investment) * 100
                    voo_return = ((final_voo_value - total_investment) / total_investment) * 100
                    excess_return = ticker_return - voo_return

                    # print(f"Period: {period_name}, Ticker Return: {ticker_return:.2f}%, VOO Return: {voo_return:.2f}%, Excess Return: {excess_return:.2f}%")

                    result[period_name] = round(excess_return, 1)
                else:
                    result[period_name] = 0
            except Exception as e:
                print(f"Error calculating {period_name}: {str(e)}")
                result[period_name] = 0

        return result

    except Exception as e:
        print(f"Error in calculate_delta_divergence_periods: {str(e)}")
        return {period: 0 for period in periods}

async def save_simplified_csv(ticker):
    df = fetch_csv(ticker)
    
    if df is None:
        return None  # None 명시적 반��
    
    folder_path = config.STATIC_IMAGES_PATH
    simplified_file_path = os.path.join(folder_path, f'result_{ticker}.csv')

    # 기존 CSV 존재 여부 체크
    if os.path.exists(simplified_file_path):
        existing_df = pd.read_csv(simplified_file_path) # 기존 CSV 파일 읽기
        last_saved_date = pd.to_datetime(existing_df['Date'].max()) # 기존 CSV 파일의 마지막 날짜 추출
        last_available_date = pd.to_datetime(df['Date'].max()) # 현재 데이터의 마지막 날짜 추출
    else:
        if df is None or df.empty or len(df) < 240: # 데이터가 없거나 데이터가 부족한 경우
            # fallback_data 생성 및 반환
            fallback_data = df[['Date','rate','rate_vs']].copy()
            fallback_data = fallback_data.rename(columns={'rate': f'rate_{ticker}_5D', 'rate_vs': 'rate_VOO_20D'})
            fallback_data['Date'] = pd.to_datetime(fallback_data['Date'])
            fallback_data['Date'] = fallback_data['Date'].dt.strftime('%Y-%m')
            fallback_data.to_csv(simplified_file_path, index=False)
            await move_files_to_images_folder(simplified_file_path)
            return fallback_data  # fallback_data 반환

    # 필요한 열 선택 및 이격도 계산
    df = df[['Date', 'Close', 'rate', 'rate_vs', 'sma60_ta', 'sma120_ta', 'sma240_ta']].copy()
    df['Divergence'] = (df['rate'] - df['rate_vs']).round(1)
    
    # 누적 최대, 최소 이격도 계산
    df['Max_Divergence'] = df['Divergence'].cummax().round(1)
    df['Min_Divergence'] = df['Divergence'].cummin().round(1)
    cumTmax_divergence = df['Divergence'].fillna(0).cummax() - df['Divergence'].fillna(0).cummin()
    df['Relative_Divergence'] = ((df['Divergence'] - df['Divergence'].fillna(0).cummin()) / cumTmax_divergence * 100).round(1)
    df['Delta_Previous_Relative_Divergence'] = df['Relative_Divergence'].diff(periods=5).fillna(0).round(1)
    
    # 변동성 계산 및 기대 수익률 계산
    df = calculate_volatility_adjustment(df)
    df = df.rename(columns={'rate': f'rate_{ticker}_5D', 'rate_vs': 'rate_VOO_20D'})
    df = calculate_expected_return(df, ticker)
    
    # Dynamic_Expected_Return 계산
    X_train, X_test, y_train, y_test = prepare_training_data(df, ticker)
    dynamic_model = train_dynamic_model(X_train, y_train)
    df = apply_dynamic_expected_return(df, dynamic_model, ticker)

    # 월별 리샘플
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    monthly_df = df.resample('ME').last().reset_index() # 매월 마지막 데이터를 월별로 리샘플링
    monthly_df['Date'] = monthly_df['Date'].dt.strftime('%Y-%m')

    simplified_df = monthly_df[['Date', f'rate_{ticker}_5D', 'rate_VOO_20D']]
    
    if simplified_df.empty:
        #  DataFrame일 경우 fallback_data 생성 및 반환
        fallback_data = df.reset_index()[['Date', f'rate_{ticker}_5D', 'rate_VOO_20D']].copy()
        fallback_data['Date'] = fallback_data['Date'].dt.strftime('%Y-%m')
        # 소수점 4자리로 반올림
        fallback_data = fallback_data.round(4)
        fallback_data.to_csv(simplified_file_path, index=False)
        await move_files_to_images_folder(simplified_file_path)
        return fallback_data  # fallback_data 반환

    # 월별 데이터 존재 시 정식 파일 생성
    simplified_df = simplified_df.set_index('Date')
    # 소수점 4자리로 반올림
    simplified_df = simplified_df.round(4)
    simplified_df.to_csv(simplified_file_path, index=True)
    await move_files_to_images_folder(simplified_file_path)
    
    # 최신 데이터 전달
    latest_entry = df.iloc[-1]
    await collect_relative_divergence(ticker, latest_entry)

    return simplified_df  # 정상적인 simplified_df 반환


# collect_relative_divergence 함수 내에서의 수정 부분
async def collect_relative_divergence(ticker, latest_entry):
    try:
        # 기본 정보 가져오기
        sector = get_sector(ticker)
        latest_relative_divergence = latest_entry['Relative_Divergence'].round(1)
        latest_divergence = latest_entry['Divergence'].round(1)
        delta_previous_relative_divergence = latest_entry['Delta_Previous_Relative_Divergence'].round(1)
        max_divergence = latest_entry['Max_Divergence'].round(1)
        min_divergence = latest_entry['Min_Divergence'].round(1)
        volatility_adjustment = latest_entry['Volatility_Adjustment'].round(1)
        expected_return = latest_entry['Expected_Return'].round(1)
        dynamic_expected_return = latest_entry['Dynamic_Expected_Return'].round(1)
        total_return = latest_entry[f'rate_{ticker}_5D'].round(1)

        # 기간별 Delta 값 계산
        delta_values = calculate_delta_divergence_periods(ticker)

        # results_relative_divergence.csv 파일 업데이트
        results_file_path = os.path.join(config.STATIC_IMAGES_PATH, 'results_relative_divergence.csv')
        
        if os.path.exists(results_file_path):
            results = pd.read_csv(results_file_path)
        else:
            results = pd.DataFrame(columns=[
                'Rank', 'Ticker', 'Sector', 'Divergence', 'Relative_Divergence', 'Delta_Previous_Relative_Divergence',
                '1m', '3m', '6m', '1y', '2y', '3y', '5y', '7y', 'max',  # 10y를 max로 변경
                'Max_Divergence', 'Min_Divergence',
                'Volatility_Adjustment', 'Expected_Return', 'Dynamic_Expected_Return', 'Total_Return'
            ])

        # 존재하는 티커 데이터 제거
        results = results[results['Ticker'] != ticker]

        # 새로운 데이터 추가
        new_entry = pd.DataFrame({
            'Ticker': [ticker], 
            'Sector': [sector],
            'Divergence': [latest_divergence], 
            'Relative_Divergence': [latest_relative_divergence],
            'Delta_Previous_Relative_Divergence': [delta_previous_relative_divergence],
            '1m': [delta_values['1m']],
            '3m': [delta_values['3m']],
            '6m': [delta_values['6m']],
            '1y': [delta_values['1y']],
            '2y': [delta_values['2y']],
            '3y': [delta_values['3y']],
            '5y': [delta_values['5y']],
            '7y': [delta_values['7y']],
            'max': [delta_values['max']],  # 10y 대신 max 값 사용
            'Max_Divergence': [max_divergence],
            'Min_Divergence': [min_divergence],
            'Volatility_Adjustment': [volatility_adjustment],
            'Expected_Return': [expected_return],
            'Dynamic_Expected_Return': [dynamic_expected_return],
            'Total_Return': [total_return]
        })

        # 결과 업데이트
        updated_results = pd.concat([results, new_entry], ignore_index=True)

        def calculate_penalty(row):
            """음수 값에 대한 페널티 계산"""
            try:
                # 기간별 음수 패널티 계산
                ultra_short_term_periods = ['1m']              # 초단기: 20점
                short_term_periods = ['3m', '6m', '1y']       # 단기: 20점
                mid_term_periods = ['2y', '3y']               # 중기: 5점
                long_term_periods = ['5y', '7y', 'max']       # 장기: 20점
                
                period_penalty = 0
                
                # 초단기 페널티 (20점)
                for period in ultra_short_term_periods:
                    if float(row.get(period, 0)) < 0:
                        period_penalty += 5
                        # print(f"[DEBUG] {row['Ticker']} - {period} penalty: 30 (Ultra-Short-term)")
                
                # 단기 페널티 (20점)
                for period in short_term_periods:
                    if float(row.get(period, 0)) < 0:
                        period_penalty += 3
                        # print(f"[DEBUG] {row['Ticker']} - {period} penalty: 20 (Short-term)")
                
                # 중기 페널티 (5점)
                for period in mid_term_periods:
                    if float(row.get(period, 0)) < 0:
                        period_penalty += 3
                        # print(f"[DEBUG] {row['Ticker']} - {period} penalty: 5 (Mid-term)")
                
                # 장기 페널티 (20점)
                for period in long_term_periods:
                    if float(row.get(period, 0)) < 0:
                        period_penalty += 5
                        # print(f"[DEBUG] {row['Ticker']} - {period} penalty: 20 (Long-term)")
                
                # Alpha 패널티
                ticker = row['Ticker']
                alpha_file_path = os.path.join(config.STATIC_DATA_PATH, f'result_alpha_{ticker}.csv')
                try:
                    if os.path.exists(alpha_file_path):
                        alpha_df = pd.read_csv(alpha_file_path)
                        positive_alpha_count = sum(alpha_df['Alpha'] > 0)
                        # print(f"[DEBUG] {ticker} - Positive Alpha Count: {positive_alpha_count}")
                        alpha_penalty = (8 - positive_alpha_count) * 10
                        
                        total_penalty = period_penalty + alpha_penalty
                        # print(f"[DEBUG] {ticker} Penalties - Period: {period_penalty}, Alpha: {alpha_penalty}")
                        
                        return total_penalty
                    else:
                        # print(f"[WARNING] No alpha file found for {ticker}")
                        return 80  # 알파 파일이 없는 경우 최대 패널티
                except Exception as e:
                    print(f"Error calculating Alpha penalty for {ticker}: {e}")
                    return 80
                
            except Exception as e:
                print(f"Error in calculate_penalty for {row.get('Ticker', 'Unknown')}: {e}")
                return 80  # 최대 패널티

        # 순위 계산 및 정렬
        for column in ['Total_Return', 'Divergence', 'Expected_Return', 'Dynamic_Expected_Return']:
            rank_column = f'{column}_Rank'
            updated_results[rank_column] = pd.to_numeric(updated_results[column], errors='coerce').rank(ascending=False)

        # 페널티 계산
        updated_results['Penalty'] = updated_results.apply(calculate_penalty, axis=1)
        
        # 평균 순위 계산 (페널티 포함)
        updated_results['Average_Rank'] = (
            updated_results['Total_Return_Rank'] * 0.2 +
            updated_results['Divergence_Rank'] * 0.2 +
            updated_results['Expected_Return_Rank'] * 0.2 +
            updated_results['Dynamic_Expected_Return_Rank'] * 0.4 +
            updated_results['Penalty']
        ).round(4)

        # 최종 순위 정렬
        updated_results = updated_results.sort_values(by='Average_Rank', ascending=True).reset_index(drop=True)
        updated_results['Rank'] = updated_results.index + 1

        # 결과 저장 - 여기만 수정
        results_file_path = os.path.join(config.STATIC_IMAGES_PATH, 'results_relative_divergence.csv')
        updated_results.to_csv(results_file_path, index=False)
        await move_files_to_images_folder(results_file_path)

        # 알파 결과 일 이름 단순하게 유지
        alpha_file_path = os.path.join(config.STATIC_DATA_PATH, f'result_alpha_{ticker}.csv')
        return updated_results

    except Exception as e:
        print(f"Error updating results for {ticker}: {e}")
        raise

async def test_collect_relative_divergence():
    all_tickers = [ticker for tickers in config_assets.STOCKS.values() for ticker in tickers]
    print(f"\nTotal tickers to process: {len(all_tickers)}")

    for i, ticker in enumerate(all_tickers, 1):
        try:
            print(f"\n[{i}/{len(all_tickers)}] Processing {ticker}...")
            await save_simplified_csv(ticker)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
    
    """테스트 실행 함수"""
    # # 테스트할 티커 리스트
    tickers = [
    #     # 'DOGE-USD',  # Rank 1
    #     # 'ENPH',      # Rank 2
    #     # 'BTC-USD',   # Rank 3
    #     # 'AMD',       # Rank 4
    #     # 'NVDA',      # Rank 5
    #     # 'ETH-USD',   # Rank 7
    #     # 'ASML',      # Rank 8
    #     # 'AMAT',      # Rank 9
    #     # 'LLY',       # Rank 10
        'TSLA'       # Rank 6
    ]
    
   
    # for i, ticker in enumerate(tickers, 1):
    #     try:
    #         print(f"\n[{i}/{len(tickers)}] Processing {ticker}...")
    #         await save_simplified_csv(ticker)
    #     except Exception as e:
    #         print(f"Error processing {ticker}: {e}")
    #         continue


if __name__ == "__main__":
    asyncio.run(test_collect_relative_divergence())
    
        
            # python get_compare_stock_data.py
# get_compare_alpha.py

import os
import sys
import pandas as pd
import numpy as np
import asyncio
import traceback

# 상위 디렉토리로 경로 추가 후, config 모듈 불러오기
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from get_ticker import get_stock_info, get_market_cap
from git_operations import move_files_to_images_folder

# 경로 설정
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STATIC_DATA_PATH = os.path.join(PROJECT_ROOT, 'static', 'data')

async def calculate_max_drawdown(group):
    """계좌 잔고 기준 MDD 계산 (%)"""
    try:
        peak = group['account_balance'].iloc[0]
        max_drawdown = 0.0

        for balance in group['account_balance']:
            if balance > peak:
                peak = balance
            # 현재 잔고가 최고점 대비 얼마나 하락했는지 계산
            drawdown = ((peak - balance) / peak * 100) if peak != 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return round(max_drawdown, 2)
    except Exception as e:
        print(f"MDD calculation error: {e}")
        return 0.0

def initialize_period_values(data, start_date, end_date, column_name):
    """특정 기간의 누적값을 초기화하는 함수"""
    try:
        period_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)].copy()
        
        if len(period_data) == 0:
            return pd.Series()
            
        # VOO 데이터의 경우 rate_vs 컬럼 사용
        actual_column = 'rate_vs' if column_name == 'rate' and 'rate_vs' in period_data.columns else column_name
            
        if actual_column not in period_data.columns:
            print(f"Column '{actual_column}' not found. Available columns: {period_data.columns.tolist()}")
            return pd.Series()
            
        # 시작 시점의 값을 0으로 초기화
        initial_value = period_data[actual_column].iloc[0]
        period_data['initialized_value'] = period_data[actual_column] - initial_value
        
        return period_data['initialized_value']
        
    except Exception as e:
        print(f"Error in initialize_period_values: {e}")
        return pd.Series()

async def calculate_yearly_metrics(ticker, df):
    """연도별 지표 계산 함수"""
    try:
        # account_balance 계산 추가
        df['account_balance'] = df['shares'] * df['Close']  # 보유 주식 수 * 현재가
        
        # overall Beta 계산
        log_returns = np.log(df['rate'] / df['rate'].shift(1))
        log_voo_returns = np.log(df['rate_vs'] / df['rate_vs'].shift(1))
        
        valid_data = pd.DataFrame({
            'log_returns': log_returns,
            'log_voo_returns': log_voo_returns
        }).dropna()
        
        if len(valid_data) > 0:
            covariance = np.cov(valid_data['log_returns'], valid_data['log_voo_returns'])[0, 1]
            variance_voo = np.var(valid_data['log_voo_returns'])
            
            if variance_voo != 0 and not np.isnan(covariance):
                overall_beta = covariance / variance_voo
                print(f"\nBeta: {overall_beta:.2f}")
            else:
                overall_beta = 1.0
        else:
            overall_beta = 1.0

        # 연도별 지표 계산
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        
        # market_cap 컬럼이 없으면 추가
        if 'market_cap' not in df.columns:
            try:
                current_market_cap = get_market_cap(ticker)
                # Close 가격 비율로 과거 시가총액 추정
                latest_close = df['Close'].iloc[-1]
                df['market_cap'] = df['Close'].apply(lambda x: current_market_cap * (x / latest_close) if latest_close != 0 else current_market_cap)
            except Exception as e:
                print(f"Error getting market cap for {ticker}: {e}")
                df['market_cap'] = 0
        
        # 월별 데이터로 그룹화
        monthly_df = df.groupby(['Year', 'Month']).agg({
            'Date': 'last',
            'rate': 'last',
            'rate_vs': 'last',
            'account_balance': 'last',
            'market_cap': 'last',
            'Close': 'last'
        }).reset_index()

        yearly_metrics = []
        current_year = pd.Timestamp.now().year  # 2025

        # MSTY 하드코딩 부분을 조건문으로 변경
        if len(monthly_df) >= 6 and (monthly_df['Year'].min() > current_year - 2 or monthly_df['Year'].max() < current_year):
            # 데이터가 6개월 이상이고, 일반적인 3년 기간 계산이 어려운 경우
            start_year = current_year - 2
            start_date = monthly_df['Date'].iloc[0]
            end_date = monthly_df['Date'].iloc[-1]
            
            group = monthly_df[(monthly_df['Date'] >= start_date) & (monthly_df['Date'] <= end_date)]

            if len(group) >= 6:  # 6개월 이상의 데이터가 있으면
                print(f"\nPeriod {start_year}-{current_year}:")
                print(f"Start date: {start_date}")
                print(f"End date: {end_date}")
                print(f"Number of months in period: {len(group)}")
                
                # CAGR 계산 (초기화된 rate 사용)
                period_rates = initialize_period_values(group, start_date, end_date, 'rate')  # 티커의 rate
                period_voo_rates = initialize_period_values(group, start_date, end_date, 'rate_vs')  # VOO의 rate_vs

                try:
                    # 티커의 CAGR 계산
                    initial_rate = group[group['Date'] == start_date]['rate'].iloc[0]
                    final_rate = group[group['Date'] == end_date]['rate'].iloc[0]
                    ticker_cagr_change = final_rate - initial_rate
                except:
                    ticker_cagr_change = 0.0

                try:
                    # VOO의 CAGR 계산
                    initial_voo_rate = group[group['Date'] == start_date]['rate_vs'].iloc[0]
                    final_voo_rate = group[group['Date'] == end_date]['rate_vs'].iloc[0]
                    voo_cagr_change = final_voo_rate - initial_voo_rate
                except:
                    voo_cagr_change = 0.0

                alpha = ticker_cagr_change - voo_cagr_change

                # MDD 계산
                try:
                    prev_balance = None
                    deltas = []  # 변화율을 저장할 리스트
                    
                    for balance in group['account_balance']:
                        if prev_balance is not None and prev_balance != 0:
                            delta = ((balance - prev_balance) / abs(prev_balance)) * 100
                            deltas.append(delta)
                        prev_balance = balance
                    
                    # 음수 변화율 중 가장 큰 하락률 찾기
                    negative_deltas = [d for d in deltas if d < 0]
                    if negative_deltas:
                        mdd = abs(min(negative_deltas))  # 가장 큰 하락률의 절대값
                    else:
                        mdd = 0.0
                    
                except Exception as e:
                    print(f"MDD calculation error: {e}")
                    mdd = 0.0

                try:
                    # 해당 기간의 시가총액 계산
                    market_cap = group['market_cap'].iloc[-1] / 1e9
                    
                    market_cap = max(round(market_cap, 2), 0.0)  # 음수 방지
                except Exception as e:
                    print(f"Error calculating market cap for {ticker} in {start_year}-{current_year}: {e}")
                    market_cap = 0.0

                # 샤프 지수 계산 추가
                try:
                    # 월간 수익률 계산
                    monthly_returns = group['rate'].pct_change()
                    # 연간 변동성 계산 (월간 표준편차 * sqrt(12))
                    annual_std = monthly_returns.std() * np.sqrt(12)
                    # 초과 수익률 (CAGR - VOO CAGR)
                    excess_return = ticker_cagr_change - voo_cagr_change
                    # 샤프 지수 계산
                    sharpe = excess_return / annual_std if annual_std != 0 else 0.0
                except Exception as e:
                    sharpe = 0.0

                yearly_metrics.append({
                    "Year": f"{start_year}-{current_year}",
                    "CAGR": int(round(float(ticker_cagr_change), 0)),
                    "S&P": int(round(float(voo_cagr_change), 0)),
                    "Alpha": int(round(float(alpha), 0)),
                    "Beta": round(float(overall_beta), 2),
                    "Sharp": round(float(sharpe), 1),
                    "MDD": round(float(mdd), 1),
                    "Cap(B)": round(float(market_cap), 1)
                })

        # 다른 티커들은 기존 로직대로 처리
        else:
            # 기존 로직 유지
            for year in range(monthly_df['Year'].min() + 2, monthly_df['Year'].max() + 1):
                try:
                    # 시작 날짜 설정 (데이터가 있는 시점부터)
                    start_year = year - 2
                    if start_year < monthly_df['Year'].min():
                        start_date = monthly_df['Date'].iloc[0]  # 데이터 시작일 사용
                    else:
                        try:
                            start_date = monthly_df[monthly_df['Year'] == start_year]['Date'].iloc[0]
                        except:
                            start_date = monthly_df['Date'].iloc[0]  # 해당 연도 데이터가 없으면 시작일 사용

                    # 종료 날짜 설정
                    try:
                        end_date = monthly_df[monthly_df['Year'] == year]['Date'].iloc[-1]
                    except:
                        end_date = monthly_df['Date'].iloc[-1]  # 해당 연도 데이터가 없으면 마지막 데이터 사용
                    
                    group = monthly_df[(monthly_df['Date'] >= start_date) & (monthly_df['Date'] <= end_date)]

                    if len(group) < 6:  # 최소 6개월 데이터로 변경
                        continue

                    print(f"\nPeriod {year - 2}-{year}:")
                    print(f"Start date: {start_date}")
                    print(f"End date: {end_date}")
                    print(f"Number of months in period: {len(group)}")

                    # CAGR 계산 (초기화된 rate 사용)
                    period_rates = initialize_period_values(group, start_date, end_date, 'rate')  # 티커의 rate
                    period_voo_rates = initialize_period_values(group, start_date, end_date, 'rate_vs')  # VOO의 rate_vs

                    try:
                        # 티커의 CAGR 계산
                        initial_rate = group[group['Date'] == start_date]['rate'].iloc[0]
                        final_rate = group[group['Date'] == end_date]['rate'].iloc[0]
                        ticker_cagr_change = final_rate - initial_rate
                    except:
                        ticker_cagr_change = 0.0

                    try:
                        # VOO의 CAGR 계산
                        initial_voo_rate = group[group['Date'] == start_date]['rate_vs'].iloc[0]
                        final_voo_rate = group[group['Date'] == end_date]['rate_vs'].iloc[0]
                        voo_cagr_change = final_voo_rate - initial_voo_rate
                    except:
                        voo_cagr_change = 0.0

                    alpha = ticker_cagr_change - voo_cagr_change

                    # MDD 계산
                    try:
                        prev_balance = None
                        deltas = []  # 변화율을 저장할 리스트
                        
                        for balance in group['account_balance']:
                            if prev_balance is not None and prev_balance != 0:
                                delta = ((balance - prev_balance) / abs(prev_balance)) * 100
                                deltas.append(delta)
                            prev_balance = balance
                        
                        # 음수 변화율 중 가장 큰 하락률 찾기
                        negative_deltas = [d for d in deltas if d < 0]
                        if negative_deltas:
                            mdd = abs(min(negative_deltas))  # 가장 큰 하락률의 절대값
                        else:
                            mdd = 0.0
                        
                    except Exception as e:
                        print(f"MDD calculation error: {e}")
                        mdd = 0.0

                    try:
                        # 해당 기간의 시가총액 계산
                        market_cap = group['market_cap'].iloc[-1] / 1e9
                        
                        market_cap = max(round(market_cap, 2), 0.0)  # 음수 방지
                    except Exception as e:
                        print(f"Error calculating market cap for {ticker} in {year - 2}-{year}: {e}")
                        market_cap = 0.0

                    # 샤프 지수 계산 추가
                    try:
                        # 월간 수익률 계산
                        monthly_returns = group['rate'].pct_change()
                        # 연간 변동성 계산 (월간 표준편차 * sqrt(12))
                        annual_std = monthly_returns.std() * np.sqrt(12)
                        # 초과 수익률 (CAGR - VOO CAGR)
                        excess_return = ticker_cagr_change - voo_cagr_change
                        # 샤프 지수 계산
                        sharpe = excess_return / annual_std if annual_std != 0 else 0.0
                    except Exception as e:
                        sharpe = 0.0

                    # 구간별 Beta 계산 추가
                    period_start = f"{year-2}-01-01"
                    period_end = f"{year}-12-31"
                    period_data = df[(df['Date'] >= period_start) & (df['Date'] <= period_end)].copy()
                    
                    # 해당 구간의 일간 수익률 계산
                    period_returns = np.log(period_data['rate'] / period_data['rate'].shift(1))
                    period_voo_returns = np.log(period_data['rate_vs'] / period_data['rate_vs'].shift(1))
                    
                    # 유효한 데이터만 선택
                    valid_mask = ~(np.isnan(period_returns) | np.isnan(period_voo_returns))
                    period_returns = period_returns[valid_mask]
                    period_voo_returns = period_voo_returns[valid_mask]
                    
                    if len(period_returns) > 0:
                        # 공분산과 분산 계산
                        covariance = np.cov(period_returns, period_voo_returns)[0, 1]
                        variance_voo = np.var(period_voo_returns)
                        
                        # 베타 계산
                        if variance_voo != 0 and not np.isnan(covariance):
                            period_beta = covariance / variance_voo
                        else:
                            period_beta = 1.0
                    else:
                        period_beta = 1.0

                    yearly_metrics.append({
                        "Year": f"{year-2}-{year}",
                        "CAGR": int(round(float(ticker_cagr_change), 0)),
                        "S&P": int(round(float(voo_cagr_change), 0)),
                        "Alpha": int(round(float(alpha), 0)),
                        "Beta": round(float(period_beta), 2),  # 새로 계산된 베타 사용
                        "Sharp": round(float(sharpe), 1),
                        "MDD": round(float(mdd), 1),
                        "Cap(B)": round(float(market_cap), 1)
                    })

                except Exception as e:
                    print(f"Error processing period {year - 2}-{year}: {e}")
                    continue

        return yearly_metrics

    except Exception as e:
        print(f"Error in calculate_yearly_metrics: {e}")
        traceback.print_exc()  # 상세한 에러 정보 출력
        return []

async def save_yearly_metrics_csv(ticker, metrics):
    """결과를 CSV 파일로 저장"""
    if not metrics:
        return None
    
    result_df = pd.DataFrame(metrics)
    print(result_df)  # 결과 표 출력만 유지
    
    result_file = os.path.join(STATIC_DATA_PATH, f"result_alpha_{ticker}.csv")
    
    try:
        result_df.to_csv(result_file, index=False)
        return result_file
    except Exception as e:
        print(f"Error saving file: {e}")
        return None

async def process_alpha_beta(ticker, combined_df, voo_data_df):
    """메인 처리 함수"""
    try:
        if combined_df.empty:
            print(f"No data in combined_df for {ticker}. Skipping process.")
            return

        yearly_metrics = await calculate_yearly_metrics(ticker, combined_df)
        result_file = await save_yearly_metrics_csv(ticker, yearly_metrics)
        if not yearly_metrics:
            print(f"No metrics to move for {ticker}.")
        else:
            await move_files_to_images_folder(result_file)
            
    except Exception as e:
        print(f"Error in process_alpha_beta: {e}")

# 실행 테스트
if __name__ == "__main__":
    tickers = [
        'DOGE-USD',  # Rank 1
        # 'ENPH',      # Rank 2
        # 'BTC-USD',   # Rank 3
        # 'AMD',       # Rank 4
        # 'NVDA',      # Rank 5
        'TSLA',      # Rank 6
        # 'ETH-USD',   # Rank 7
        # 'ASML',      # Rank 8
        'QQQ',      # Rank 9
        'MSTY'        # Rank 10
    ]
    
    for ticker in tickers:
        try:
            ticker_file = os.path.join(config.STATIC_IMAGES_PATH, f'result_VOO_{ticker}.csv')
            voo_file = os.path.join(config.STATIC_IMAGES_PATH, 'result_VOO_VOO.csv')
            
            if os.path.exists(ticker_file) and os.path.exists(voo_file):
                df = pd.read_csv(ticker_file)
                voo_data_df = pd.read_csv(voo_file)
                asyncio.run(process_alpha_beta(ticker, df, voo_data_df))
            else:
                print(f"Data files not found for {ticker}")
                
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

# python get_compare_alpha.py
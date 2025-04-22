# Get_dividend.py

import yfinance as yf
import pandas as pd
import logging
import os
import sys
from datetime import datetime, timedelta
import numpy as np

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

def get_dividends_for_ticker(ticker):
    """
    특정 티커의 배당금 정보를 가져오는 함수
    
    Args:
        ticker (str): 주식 티커 심볼
    
    Returns:
        pd.DataFrame: Date와 Dividend 컬럼을 포함하는 데이터프레임
    """
    try:
        yf_ticker = yf.Ticker(ticker)
        dividends = yf_ticker.dividends  # Series(Date -> Dividend)
        
        if dividends.empty:
            logging.info(f"No dividend history found for {ticker}")
            return pd.DataFrame(columns=['Date', 'Dividend'])
            
        dividends = dividends.reset_index()
        dividends.columns = ['Date', 'Dividend']
        dividends['Date'] = pd.to_datetime(dividends['Date'])
        
        # 배당금 정보 로깅
        logging.info(f"Retrieved {len(dividends)} dividend records for {ticker}")
        logging.info(f"Latest dividend: ${dividends['Dividend'].iloc[-1]:.4f} on {dividends['Date'].iloc[-1].strftime('%Y-%m-%d')}")
        
        return dividends
        
    except Exception as e:
        logging.error(f"Failed to fetch dividends for {ticker}: {e}")
        return pd.DataFrame(columns=['Date', 'Dividend'])

def calculate_dividend_income(current_date, shares, dividend_df):
    """
    특정 날짜의 배당금 수입을 계산하는 함수
    
    Args:
        current_date (datetime): 현재 날짜
        shares (float): 보유 주식 수
        dividend_df (pd.DataFrame): 배당금 데이터프레임
    
    Returns:
        tuple: (배당금 금액, 배당금 지급 여부를 나타내는 문자열)
    """
    try:
        dividend_row = dividend_df[dividend_df['Date'].dt.date == current_date.date()]
        if not dividend_row.empty:
            dividend = dividend_row['Dividend'].values[0] * shares
            signal = f"💰 Dividend: ${dividend:.2f}"
            logging.info(f"Dividend payment on {current_date.date()}: ${dividend:.2f} ({shares} shares)")
            return dividend, signal
        return 0, ""
        
    except Exception as e:
        logging.error(f"Error calculating dividend income: {e}")
        return 0, ""

def get_annual_dividend_yield(ticker):
    """
    연간 배당 수익률을 계산하는 함수
    
    Args:
        ticker (str): 주식 티커 심볼
    
    Returns:
        float: 연간 배당 수익률 (%)
    """
    try:
        stock = yf.Ticker(ticker)
        current_price = stock.info.get('regularMarketPrice', 0)
        if not current_price:
            return 0
            
        # 최근 1년간의 배당금 합계 계산
        dividends = stock.dividends
        if dividends.empty:
            return 0
            
        annual_dividend = dividends[-4:].sum() if len(dividends) >= 4 else dividends.sum()
        dividend_yield = (annual_dividend / current_price) * 100
        
        logging.info(f"{ticker} Annual Dividend Yield: {dividend_yield:.2f}%")
        return round(dividend_yield, 2)
        
    except Exception as e:
        logging.error(f"Error calculating dividend yield for {ticker}: {e}")
        return 0

def get_dividend_history_summary(ticker, years=5):
    """
    배당금 히스토리 요약 정보를 제공하는 함수
    
    Args:
        ticker (str): 주식 티커 심볼
        years (int): 조회할 연도 수
    
    Returns:
        dict: 배당금 관련 요약 정보
    """
    try:
        dividend_df = get_dividends_for_ticker(ticker)
        if dividend_df.empty:
            return {
                'average_dividend': 0,
                'dividend_growth': 0,
                'payout_frequency': 0,
                'years_of_growth': 0
            }
            
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        filtered_df = dividend_df[dividend_df['Date'] >= start_date]
        if filtered_df.empty:
            return {
                'average_dividend': 0,
                'dividend_growth': 0,
                'payout_frequency': 0,
                'years_of_growth': 0
            }
            
        # 평균 배당금
        average_dividend = filtered_df['Dividend'].mean()
        
        # 배당금 성장률
        yearly_dividends = filtered_df.groupby(filtered_df['Date'].dt.year)['Dividend'].sum()
        if len(yearly_dividends) >= 2:
            dividend_growth = ((yearly_dividends.iloc[-1] / yearly_dividends.iloc[0]) ** (1/len(yearly_dividends)) - 1) * 100
        else:
            dividend_growth = 0
            
        # 연간 배당 횟수
        payout_frequency = len(filtered_df) / years
        
        # 연속 배당 성장 연수
        years_of_growth = 0
        yearly_dividends = yearly_dividends.sort_index()
        for i in range(len(yearly_dividends)-1, 0, -1):
            if yearly_dividends.iloc[i] > yearly_dividends.iloc[i-1]:
                years_of_growth += 1
            else:
                break
                
        summary = {
            'average_dividend': round(average_dividend, 4),
            'dividend_growth': round(dividend_growth, 2),
            'payout_frequency': round(payout_frequency, 1),
            'years_of_growth': years_of_growth
        }
        
        logging.info(f"Dividend summary for {ticker}:")
        for key, value in summary.items():
            logging.info(f"{key}: {value}")
            
        return summary
        
    except Exception as e:
        logging.error(f"Error getting dividend history summary for {ticker}: {e}")
        return {
            'average_dividend': 0,
            'dividend_growth': 0,
            'payout_frequency': 0,
            'years_of_growth': 0
        }

def calculate_dividend_tax(dividend_amount: float, country: str = 'US', tax_treaty: bool = True) -> dict:
    """
    배당금에 대한 세금을 계산하는 함수 (단순화된 버전: 30% 세금 공제)
    
    Args:
        dividend_amount (float): 배당금 금액
        country (str): 사용하지 않음
        tax_treaty (bool): 사용하지 않음
    
    Returns:
        dict: 세금 계산 결과
        {
            'gross_dividend': 총 배당금,
            'withholding_tax': 원천징수세,
            'local_tax': 지방세,
            'net_dividend': 실수령액
        }
    """
    try:
        result = {
            'gross_dividend': dividend_amount,
            'withholding_tax': dividend_amount * 0.3,  # 30% 세금
            'local_tax': 0.0,
            'net_dividend': dividend_amount * 0.7  # 70% 실수령
        }
        return result
        
    except Exception as e:
        logging.error(f"Error calculating dividend tax: {e}")
        return {
            'gross_dividend': dividend_amount,
            'withholding_tax': 0.0,
            'local_tax': 0.0,
            'net_dividend': dividend_amount
        }

def get_dividend_for_date(ticker: str, target_date: str, country: str = 'US', tax_treaty: bool = True) -> tuple:
    """
    특정 날짜의 배당금 정보를 조회하는 함수 (세금 계산 포함)
    
    Args:
        ticker (str): 주식 티커 심볼
        target_date (str): 조회할 날짜 (YYYY-MM-DD 형식)
        country (str): 주식 발행 국가
        tax_treaty (bool): 조세협약 적용 여부
    
    Returns:
        tuple: (배당금 존재 여부(bool), 주당 배당금(float), 세금 정보(dict))
    """
    try:
        # 날짜 형식 변환
        target_date = pd.to_datetime(target_date).date()
        
        # 배당금 정보 조회
        yf_ticker = yf.Ticker(ticker)
        dividends = yf_ticker.dividends
        
        if dividends.empty:
            return False, 0.0, None
            
        # Series를 DataFrame으로 변환
        dividends = dividends.reset_index()
        dividends.columns = ['Date', 'Dividend']
        dividends['Date'] = pd.to_datetime(dividends['Date']).dt.date
        
        # 해당 날짜의 배당금 확인
        dividend_row = dividends[dividends['Date'] == target_date]
        
        if dividend_row.empty:
            return False, 0.0, None
            
        dividend_amount = dividend_row['Dividend'].iloc[0]
        logging.info(f"Found dividend for {ticker} on {target_date}: ${dividend_amount:.4f}")
        
        tax_info = calculate_dividend_tax(dividend_amount, country, tax_treaty)
        return True, dividend_amount, tax_info
        
    except Exception as e:
        logging.error(f"Error getting dividend with tax info: {e}")
        return False, 0.0, None

def get_next_dividend_date(ticker: str) -> tuple:
    """
    다음 배당금 지급일과 예상 배당금을 조회하는 함수
    
    Args:
        ticker (str): 주식 티커 심볼
    
    Returns:
        tuple: (다음 배당일(str), 예상 배당금(float))
    """
    try:
        yf_ticker = yf.Ticker(ticker)
        
        # 다음 배당일 정보 조회
        calendar = yf_ticker.calendar
        if calendar is not None and 'Dividend Date' in calendar.columns:
            next_div_date = calendar['Dividend Date'].iloc[0]
            if pd.notnull(next_div_date):
                next_div_date = pd.to_datetime(next_div_date).strftime('%Y-%m-%d')
                
                # 최근 배당금으로 예상 배당금 추정
                recent_div = yf_ticker.dividends
                if not recent_div.empty:
                    expected_amount = recent_div.iloc[-1]
                    logging.info(f"Next dividend for {ticker}: ${expected_amount:.4f} on {next_div_date}")
                    return next_div_date, expected_amount
                    
        return None, 0.0
        
    except Exception as e:
        logging.error(f"Error getting next dividend date for {ticker}: {e}")
        return None, 0.0

if __name__ == "__main__":
    # 테스트 코드
    test_ticker = "AAPL"
    start_date = pd.to_datetime("2015-01-02")
    end_date = pd.to_datetime("2015-06-30")
    
    print(f"\n{test_ticker}의 배당금 테스트 ({start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')})")
    print("-" * 50)
    
    # 날짜 범위 생성
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 각 날짜별 배당금 확인 (세금 정보 포함)
    found_dividends = False
    total_gross = 0.0
    total_net = 0.0
    
    for current_date in date_range:
        date_str = current_date.strftime('%Y-%m-%d')
        has_dividend, amount, tax_info = get_dividend_for_date(test_ticker, date_str, 'US', True)
        
        if has_dividend:
            found_dividends = True
            total_gross += amount
            total_net += tax_info['net_dividend']
            
            print(f"\n날짜: {date_str}")
            print(f"주당 총 배당금: ${amount:.4f}")
            print(f"원천징수세: ${tax_info['withholding_tax']:.4f}")
            print(f"실수령액: ${tax_info['net_dividend']:.4f}")
    
    if found_dividends:
        print("\n기간 전체 합계:")
        print(f"총 배당금: ${total_gross:.4f}")
        print(f"실수령액: ${total_net:.4f}")
        print(f"총 세금: ${(total_gross - total_net):.4f}")
    else:
        print(f"해당 기간에 배당금 지급 내역이 없습니다.")
    
    print("-" * 50)
    
    # 다음 배당일 확인
    next_date, expected_amount = get_next_dividend_date(test_ticker)
    if next_date:
        print(f"\n{test_ticker}의 다음 배당일: {next_date}")
        print(f"예상 배당금: ${expected_amount:.4f}")
    else:
        print(f"\n{test_ticker}의 다음 배당일 정보를 찾을 수 없습니다.")
    
    # 배당금 정보 조회
    dividend_df = get_dividends_for_ticker(test_ticker)
    if not dividend_df.empty:
        print(f"\n{test_ticker} 최근 배당금 정보:")
        print(dividend_df.tail())
        
    # 배당 수익률 조회
    dividend_yield = get_annual_dividend_yield(test_ticker)
    print(f"\n{test_ticker} 연간 배당 수익률: {dividend_yield}%")
    
    # 배당금 히스토리 요약
    summary = get_dividend_history_summary(test_ticker)
    print(f"\n{test_ticker} 배당금 히스토리 요약:")
    for key, value in summary.items():
        print(f"{key}: {value}") 
        
# python Get_dividend.py
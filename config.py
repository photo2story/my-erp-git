# config.py
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
import pytz
import pandas_market_calendars as mcal
import yfinance as yf
from config_assets import STOCKS  # STOCKS를 별도의 파일에서 가져옴
# STOCKS = config_asset.STOCKS (import하여 이미 정의됨)


load_dotenv()

# Discord configuration
DISCORD_APPLICATION_TOKEN = os.getenv('DISCORD_APPLICATION_TOKEN', '')
DISCORD_CHANNEL_ID = int(os.getenv('DISCORD_CHANNEL_ID', '0'))
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL', '')
DISCORD_TR_WEBHOOK_URL = os.getenv('DISCORD_TR_WEBHOOK_URL', '')

# FRED API 키 설정(버핏지수)
FRED_API_KEY = os.getenv('FRED_API_KEY')

# 환율 설정
KRW_USD_EXCHANGE_RATE = float(os.getenv('KRW_USD_EXCHANGE_RATE', 1300.0))

# Investment and backtest configuration
START_DATE = os.getenv('START_DATE', '2015-01-02')
END_DATE = datetime.today().strftime('%Y-%m-%d')
INITIAL_INVESTMENT = int(os.getenv('INITIAL_INVESTMENT', 1000))
MONTHLY_INVESTMENT = int(os.getenv('MONTHLY_INVESTMENT', 1000))

# option_strategy = 'default'
# option_strategy = 'hedge_50'  # 예시: 헷지 전략 선택
# option_strategy = 'modified_mri_seasonal'  # 예시: 헷지 전략 선택
option_strategy = 'hybrid'  # 예시: 헷지 전략 선택



# 프로젝트 루트 경로 설정
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# static/images 폴더 경로 설정 (프로젝트 루트 기준)
STATIC_IMAGES_PATH = os.path.join(PROJECT_ROOT, 'static', 'images')

# static/data 폴더 경로 설정 (프로젝트 루트 기준)
STATIC_DATA_PATH = os.path.join(PROJECT_ROOT, 'static', 'data')

# VOO 캐시 파일 경로 설정
VOO_CACHE_FILE = os.path.join(STATIC_IMAGES_PATH, 'cached_voo_data.csv')

# 기타 CSV 파일 경로 설정 (예: stock_market.csv)
CSV_PATH = os.path.join(STATIC_IMAGES_PATH, 'stock_market.csv')

# Data URLs
# CSV_URL = os.getenv('CSV_URL', 'https://raw.githubusercontent.com/photo2story/my-flask-app/main/static/images/stock_market.csv')
# GITHUB_API_URL = os.getenv('GITHUB_API_URL', 'https://api.github.com/repos/photo2story/my-flask-app/contents/static/images')
CSV_URL = CSV_PATH
GITHUB_API_URL = STATIC_IMAGES_PATH

#  전략 설정

# 전략 모드 설정
STRATEGY_MODE = {
}



# 추가된 함수: 분석 검증
# 미국 동부 표준시(EST)로 시간 설정
us_eastern = pytz.timezone('US/Eastern')
korea_time = datetime.now().astimezone(pytz.timezone('Asia/Seoul'))
us_market_close = korea_time.astimezone(us_eastern).replace(hour=16, minute=0, second=0, microsecond=0)
us_market_close_in_korea_time = us_market_close.astimezone(pytz.timezone('Asia/Seoul'))






def is_gemini_analysis_complete(ticker):
    report_file_path = os.path.join(STATIC_IMAGES_PATH, f'report_{ticker}.txt')
    
    if not os.path.exists(report_file_path):
        return False
    
    try:
        with open(report_file_path, 'r', encoding='utf-8') as file:
            first_line = file.readline().strip()
            today_date_str = datetime.now().strftime('%Y-%m-%d')
            
            if today_date_str in first_line:
                return True
            else:
                return False
    except Exception as e:
        print(f"Error reading report file for {ticker}: {e}")
        return False
    


# 미국 동부 표준시 (EST)로 시간을 맞춤
us_eastern = pytz.timezone('US/Eastern')


def is_cache_valid(cache_file, start_date, end_date):
    """
    캐시 파일의 유효성을 검사합니다:
    - 파일의 시작 날짜가 원하는 시작 날짜와 일치하는지 확인.
    """



# 이 함수들을 봇의 다른 부분에서 호출하여 유효성을 검토할 수 있습니다.
if __name__ == '__main__':
    # 분석할 티커 설정
    print(f"Gemini analysis complete ")
    
# python config.py


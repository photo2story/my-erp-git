# get_earning_fmp.py 
import os
import requests
import pandas as pd
from dotenv import load_dotenv

# .env 파일의 환경 변수를 로드합니다.
load_dotenv()

API_KEY = os.getenv('FMP_API_KEY')  # .env 파일에서 API 키를 가져옵니다
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')  # .env 파일에서 디스코드 웹훅 URL을 가져옵니다

if not API_KEY:
    raise ValueError("API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")

if not DISCORD_WEBHOOK_URL:
    raise ValueError("DISCORD 웹훅 URL이 설정되지 않았습니다. .env 파일을 확인하세요.")

def get_cik_by_ticker(ticker):
    """FMP API에서는 CIK가 필요 없으므로 기본 값을 반환."""
    return "FMP"

def get_financial_data(ticker):
    url = f'https://financialmodelingprep.com/api/v3/earnings-surprises/{ticker}?apikey={API_KEY}'
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error retrieving data for {ticker}. Status code: {response.status_code}")
        return None
    
    try:
        data = response.json()
        return data
    except ValueError as e:
        print('Error parsing JSON:', e)
        return None

def get_recent_eps_and_revenue_fmp(ticker):
    earnings_data = get_financial_data(ticker)
    
    if not earnings_data:
        print(f"No earnings data found for {ticker}.")
        return None
    
    # 최근 5개 분기 데이터 추출
    recent_earnings = []
    for data in earnings_data[:5]:
        end_date = data['date']
        actual_eps = data['actualEarningResult']
        estimated_eps = data.get('estimatedEarning', 'N/A')  # 추정치
        revenue = data.get('revenue', 'N/A')  # 실제 매출 데이터
        estimated_revenue = data.get('estimatedRevenue', 'N/A')  # 추정 매출 데이터
        
        # revenue와 estimated_revenue 정보가 있으면 포함
        if revenue != 'N/A' and estimated_revenue != 'N/A':
            recent_earnings.append((end_date, actual_eps, estimated_eps, revenue, estimated_revenue))
        else:
            recent_earnings.append((end_date, actual_eps, estimated_eps))
    
    return recent_earnings

if __name__ == "__main__":
    # 테스트 코드
    results = get_recent_eps_and_revenue_fmp("NVDA")
    if results:
        print("\nQuarterly Results:")
        for entry in results:
            if len(entry) == 5:
                end, actual_eps, estimated_eps, revenue, estimated_revenue = entry
                print(f"{end}: EPS {actual_eps} (Estimated: {estimated_eps}), Revenue {revenue / 1e9:.2f} B$ (Estimated: {estimated_revenue / 1e9:.2f} B$)")
            else:
                end, actual_eps, estimated_eps = entry
                print(f"{end}: EPS {actual_eps} (Estimated: {estimated_eps})")
    else:
        print("No data found for TSM")

# python get_earning_fmp.py    
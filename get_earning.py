# get_earning.py 
import os
import requests
from datetime import datetime
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()
API_KEY = os.getenv('FMP_API_KEY')
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')

if not API_KEY:
    raise ValueError("API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
if not DISCORD_WEBHOOK_URL:
    raise ValueError("DISCORD 웹훅 URL이 설정되지 않았습니다. .env 파일을 확인하세요.")

def get_cik_by_ticker(ticker):
    search_url = "https://www.sec.gov/include/ticker.txt"
    headers = {
        "User-Agent": "Your Name (your_email@example.com)"
    }
    response = requests.get(search_url, headers=headers)
    if response.status_code == 200:
        for line in response.text.splitlines():
            if line.lower().startswith(ticker.lower()):
                return line.split('\t')[1].zfill(10)
    return None

def get_financial_data(cik, tag):
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
    headers = {
        "User-Agent": "Your Name (your_email@example.com)"
    }
    print(f"Full URL: {url}")
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    print(f"Error retrieving data for {tag}.")
    return None

def get_recent_eps_and_revenue(ticker):
    cik = get_cik_by_ticker(ticker)
    if not cik:
        print(f"Error: CIK not found for ticker {ticker}.")
        return None

    eps_data = get_financial_data(cik, "EarningsPerShareBasic")
    revenue_data = get_financial_data(cik, "RevenueFromContractWithCustomerExcludingAssessedTax")
    
    if not eps_data:
        print("Error retrieving EPS data. Checking alternative data points.")
        eps_data = get_financial_data(cik, "EarningsPerShareDiluted")

    if not revenue_data:
        print("Error retrieving Revenue data. Checking alternative data points.")
        revenue_data = get_financial_data(cik, "Revenues")

    if not eps_data:
        print("No EPS data available.")
        return None
    
    def extract_data(data, unit_key):
        if data and unit_key in data['units']:
            return data['units'][unit_key]
        print(f"No '{unit_key}' unit data found.")
        return []

    eps_facts = extract_data(eps_data, 'USD/shares')
    revenue_facts = extract_data(revenue_data, 'USD') if revenue_data else []

    def filter_quarterly_data(data):
        quarterly_data = []
        for fact in data:
            start = datetime.strptime(fact['start'], "%Y-%m-%d")
            end = datetime.strptime(fact['end'], "%Y-%m-%d")
            if (end - start).days <= 92:
                quarterly_data.append(fact)
        return quarterly_data
    
    eps_facts = filter_quarterly_data(eps_facts)
    revenue_facts = filter_quarterly_data(revenue_facts) if revenue_facts else []
    
    eps_facts = sorted(eps_facts, key=lambda x: x['end'], reverse=True)
    revenue_facts = sorted(revenue_facts, key=lambda x: x['end'], reverse=True) if revenue_facts else []

    quarterly_results = []
    revenue_dict = {rev['end']: rev for rev in revenue_facts}
    for eps in eps_facts:
        eps_val = eps.get('val', None)
        if eps_val is not None and isinstance(eps_val, (int, float)):
            revenue_val = None
            rev = revenue_dict.get(eps['end'], None)
            if rev:
                revenue_val = rev.get('val', None)
            quarterly_results.append((eps['end'], eps['filed'], eps_val, revenue_val))
        if len(quarterly_results) == 5:
            break

    return quarterly_results

def get_financial_data_fmp(ticker):
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
    earnings_data = get_financial_data_fmp(ticker)
    
    if not earnings_data:
        print(f"No earnings data found for {ticker}.")
        return None
    
    recent_earnings = []
    for data in earnings_data[:5]:
        end_date = data['date']
        actual_eps = data['actualEarningResult']
        estimated_eps = data.get('estimatedEarning', 'N/A')
        revenue = data.get('revenue', 'N/A')
        estimated_revenue = data.get('estimatedRevenue', 'N/A')
        
        if revenue != 'N/A' and estimated_revenue != 'N/A':
            recent_earnings.append((end_date, actual_eps, estimated_eps, revenue, estimated_revenue))
        else:
            recent_earnings.append((end_date, actual_eps, estimated_eps))
    
    return recent_earnings

def get_combined_eps_and_revenue(ticker):
    recent_earnings = get_recent_eps_and_revenue(ticker)
    if recent_earnings is None or all(entry[3] is None for entry in recent_earnings):
        print(f"Primary data source failed for {ticker}, attempting secondary source...")
        recent_earnings = get_recent_eps_and_revenue_fmp(ticker)
        if recent_earnings is None:
            raise Exception("No recent earnings data found from secondary source.")
    return recent_earnings

if __name__ == "__main__":
    # 테스트 코드
    ticker = "NVDA"
    results = get_combined_eps_and_revenue(ticker)
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



# python get_earning.py    
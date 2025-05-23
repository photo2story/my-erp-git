# -*- coding: utf-8 -*-
import pandas as pd
import FinanceDataReader as fdr
import csv, os, io, requests, sys
from github_operations import ticker_path  # stock_market.csv 파일 경로
import yfinance as yf
import investpy
import config  # config 파일을 import
from datetime import datetime
# 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
CSV_PATH = config.CSV_PATH


#ticker_path = os.getenv('CSV_URL', 'https://raw.githubusercontent.com/photo2story/my-flask-app/main/static/images/stock_market.csv')
CSV_URL = 'https://raw.githubusercontent.com/photo2story/my-flask-app/main/static/images/stock_market.csv'
# CSV_URL = config.CSV_PATH  # config.py에서 정의된 CSV_PATH를 사용
ticker_path = config.CSV_PATH  # 동일한 경로를 사용

def get_ticker_name(ticker):
    df = pd.read_csv(ticker_path, encoding='utf-8')  # stock_market.csv 파일을 읽음
    result = df.loc[df['Symbol'] == ticker, 'Name']
    if not result.empty:
        # CSV에서 이름을 찾은 경우
        return result.iloc[0]
    
    # CSV에서 찾지 못한 경우 yfinance를 통한 fallback
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        name = info.get('shortName') or info.get('longName')
        if name:
            return name
        else:
            return ticker  # shortName, longName 모두 없으면 티커 반환
    except Exception as e:
        print(f"Error fetching name for {ticker} from yfinance: {e}")
        return ticker  # 에러 발생 시 티커 반환

def get_ticker_market(ticker):
    df = pd.read_csv(ticker_path, encoding='utf-8')  # stock_market.csv 파일 경로 인코딩 설정
    result = df.loc[df['Symbol'] == ticker, 'Market']
    market = result.iloc[0] if not result.empty else None
    return market

def get_stock_info(ticker):
    info = yf.Ticker(ticker).info
    return {
        'Stock': ticker,
        'Industry': info.get('industry'),
        'Beta': info.get('beta'),
        'Sector': info.get('sector')
    }

def update_stock_market_csv(file_path, tickers_to_update):
    df = pd.read_csv(ticker_path, encoding='utf-8-sig')  # Specify encoding
    for i, row in df.iterrows():
        ticker = row['Symbol']
        if ticker in tickers_to_update:
            stock_info = get_stock_info(ticker)
            if stock_info:
                for key, value in stock_info.items():
                    df.at[i, key] = value
            else:
                # 데이터가 없는 경우 기본값을 설정합니다.
                df.at[i, 'Sector'] = 'Unknown'
                df.at[i, 'Stock'] = 'Unknown Stock'
                df.at[i, 'Industry'] = 'Unknown Industry'
                df.at[i, 'Beta'] = 0.0
                df.at[i, 'marketcap'] = 0.0                
    
    df.to_csv(ticker_path, index=False, encoding='utf-8-sig')  # Specify encoding

def load_tickers():
    ticker_dict = {}
    response = requests.get(CSV_URL)
    response.raise_for_status()
    csv_data = response.content.decode('utf-8')
    csv_reader = csv.reader(io.StringIO(csv_data))
    for rows in csv_reader:
        if len(rows) >= 2:
            ticker_dict[rows[1]] = rows[0]
    return ticker_dict

def search_tickers(stock_name, ticker_dict):
    stock_name_lower = stock_name.lower()
    return [(ticker, name) for name, ticker in ticker_dict.items() if stock_name_lower in name.lower()]

def search_ticker_list_KR():
    url = 'http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13'
    response = requests.get(url)
    response.encoding = 'euc-kr'  # Korean encoding
    df_listing = pd.read_html(response.text, header=0)[0]
    cols_ren = {
        '회사명': 'Name',
        '종목코드': 'Symbol',
        '업종': 'Sector',
    }
    df_listing = df_listing.rename(columns=cols_ren)
    df_listing['market'] = 'KRX'
    df_listing['Symbol'] = df_listing['Symbol'].apply(lambda x: '{:06d}'.format(x))
    df_KR = df_listing[['Symbol', 'Name', 'market', 'Sector']]
    return df_KR

def search_ticker_list_US():
    df_amex = fdr.StockListing('AMEX')
    df_nasdaq = fdr.StockListing('NASDAQ')
    df_nyse = fdr.StockListing('NYSE')
    try:
        df_ETF_US = fdr.StockListing("ETF/US")
        df_ETF_US['market'] = "us_ETF"
        columns_to_select = ['Symbol', 'Name', 'market']
        df_ETF_US = df_ETF_US[columns_to_select]
    except Exception as e:
        print(f"An error occurred while fetching US ETF listings: {e}")
        df_ETF_US = pd.DataFrame(columns=['Symbol', 'Name', 'market'])
    df_amex['market'] = "AMEX"
    df_nasdaq['market'] = "NASDAQ"
    df_nyse['market'] = "NYSE"
    columns_to_select = ['Symbol', 'Name', 'market']
    df_amex = df_amex[columns_to_select]
    df_nasdaq = df_nasdaq[columns_to_select]
    df_nyse = df_nyse[columns_to_select]
    data_frames_US = [df_nasdaq, df_nyse, df_amex, df_ETF_US]
    df_US = pd.concat(data_frames_US, ignore_index=True)
    df_US['Sector'] = 'none'
    df_US = df_US[['Symbol', 'Name', 'market', 'Sector']]
    return df_US

def search_ticker_list_US_ETF():
    df_etfs = investpy.etfs.get_etfs(country='united states')
    df_US_ETF = df_etfs[['symbol', 'name']].copy()
    df_US_ETF['market'] = 'US_ETF'
    df_US_ETF['Sector'] = 'US_ETF'
    df_US_ETF.columns = ['Symbol', 'Name', 'market', 'Sector']
    return df_US_ETF

def get_ticker_list_all():
    df_KR = search_ticker_list_KR()
    df_US = search_ticker_list_US()
    df_US_ETF = search_ticker_list_US_ETF()
    df_combined = pd.concat([df_KR, df_US, df_US_ETF], ignore_index=True)
    df_combined.to_csv(ticker_path, encoding='utf-8-sig', index=False)
    return df_combined

def get_ticker_from_korean_name(name):
    df_KR = search_ticker_list_KR()
    result = df_KR.loc[df_KR['Name'] == name, 'Symbol']
    ticker = result.iloc[0] if not result.empty else None
    return ticker

async def search_tickers_and_respond(ctx, query):
    ticker_dict = load_tickers()
    matching_tickers = search_tickers(query, ticker_dict)
    if not matching_tickers:
        await ctx.send("No search results.")
        return
    response_message = "Search results:\n"
    response_messages = []
    for symbol, name in matching_tickers:
        line = f"{symbol} - {name}\n"
        if len(response_message) + len(line) > 2000:
            response_messages.append(response_message)
            response_message = "Search results (continued):\n"
        response_message += line
    if response_message:
        response_messages.append(response_message)
    for message in response_messages:
        await ctx.send(message)
    print(f'Sent messages for query: {query}')

def is_valid_stock(stock):  # Check if the stock is in the stock market CSV
    try:
        # url = 'https://raw.githubusercontent.com/photo2story/my-flask-app/main/static/images/stock_market.csv'
        url = config.CSV_PATH
        stock_market_df = pd.read_csv(url, encoding='utf-8')  # Specify encoding
        return stock in stock_market_df['Symbol'].values
    except Exception as e:
        print(f"Error checking stock market CSV: {e}")
        return False
    
import pandas as pd
import yfinance as yf
import os

def get_market_cap(ticker):
    """티커의 시가총액을 가져오는 함수"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if ticker.endswith('-USD'):  # 암호화폐
            try:
                # 현재가격 가져오기
                current_price = info.get('regularMarketPrice', 0)
                if current_price == 0:
                    current_price = stock.history(period='1d')['Close'].iloc[-1]
                
                circulating_supply = info.get('circulatingSupply', 0)
                market_cap = current_price * circulating_supply
                print(f"Crypto {ticker} market cap: ${market_cap/1e9:.2f}B")
                return market_cap

            except Exception as e:
                print(f"Error getting crypto market cap for {ticker}: {e}")
                return 0

        elif any(ticker.startswith(prefix) for prefix in ['SPY', 'QQQ', 'VOO', 'MSTY']):  # ETF
            try:
                # netAssets를 우선적으로 사용
                net_assets = info.get('netAssets', 0)
                if net_assets == 0:
                    # netAssets가 없으면 totalAssets 사용
                    net_assets = info.get('totalAssets', 0)
                
                if net_assets == 0:
                    # 현재가와 shares outstanding으로 계산 시도
                    try:
                        current_price = info.get('regularMarketPrice', 0)
                        if current_price == 0:
                            current_price = stock.history(period='1d')['Close'].iloc[-1]
                        shares = info.get('sharesOutstanding', 0)
                        if shares > 0:
                            net_assets = current_price * shares
                    except Exception as e:
                        print(f"Error calculating ETF assets from price and shares: {e}")

                print(f"ETF {ticker} net assets: ${net_assets/1e9:.2f}B")
                return net_assets

            except Exception as e:
                print(f"Error getting ETF net assets for {ticker}: {e}")
                return 0

        else:  # 일반 주식
            try:
                market_cap = info.get('marketCap', 0)
                print(f"{ticker} market cap: ${market_cap/1e9:.2f}B")
                return market_cap

            except Exception as e:
                print(f"Error getting stock market cap for {ticker}: {e}")
                return 0

    except Exception as e:
        print(f"Error in get_market_cap for {ticker}: {e}")
        return 0

def update_market_cap_in_csv(csv_url):
    """
    CSV 파일을 읽어 티커의 시가총액을 업데이트하는 함수.
    NYSE와 NASDAQ 시장의 주식만 포함.
    :param csv_url: CSV 파일 URL
    """
    response = requests.get(csv_url)
    response.raise_for_status()
    csv_data = response.content.decode('utf-8')
    df = pd.read_csv(io.StringIO(csv_data))

    # NYSE와 NASDAQ 주식 필터링
    filtered_df = df[df['Market'].isin(['NYSE', 'NASDAQ'])]

    # marketCap 열이 없으면 새로 추가
    if 'marketCap' not in df.columns:
        df['marketCap'] = 0.0

    total_tickers = len(filtered_df)
    for index, row in filtered_df.iterrows():
        ticker = row['Symbol']
        market_cap = get_market_cap(ticker)
        df.at[index, 'marketCap'] = market_cap
        print(f"Processed {index + 1}/{total_tickers} - {ticker}")

    # 업데이트된 데이터프레임을 로컬에 저장 (원격에 저장하려면 추가 작업 필요)
    df.to_csv('updated_stock_market.csv', index=False)
    print(f"Updated CSV saved to updated_stock_market.csv")

import requests
from bs4 import BeautifulSoup
import pandas as pd

def fetch_finviz_data(ticker):
    """
    주어진 티커의 Finviz 팩터 데이터를 가져오는 함수
    :param ticker: 주식 티커
    :return: 팩터 데이터를 포함한 Pandas 데이터프레임
    """
    url = f'https://finviz.com/quote.ashx?t={ticker}'
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        # 페이지 요청
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # 요청이 성공했는지 확인

        # 페이지 파싱
        soup = BeautifulSoup(response.text, 'html.parser')

        # Finviz 테이블 확인
        tables = soup.find_all('table')
        print(f"Total tables found: {len(tables)}")  # 테이블 개수 확인

        # Finviz 테이블 출력 (디버그용)
        for idx, table in enumerate(tables):
            print(f"\n--- Table {idx + 1} ---")
            print(table.prettify())  # 테이블의 구조를 출력 (디버그용)

        if len(tables) < 8:
            print(f"Finviz 테이블이 제대로 로드되지 않았습니다. 티커: {ticker}")
            return pd.DataFrame()

        # 필요한 테이블 선택 (여기서는 8번째 테이블이 주로 데이터임)
        finviz_table = tables[7]

        # 데이터 파싱
        rows = finviz_table.find_all('tr')

        # 데이터를 저장할 딕셔너리
        data = {}

        # 행을 순회하며 데이터 추출
        for row in rows:
            columns = row.find_all('td')
            if len(columns) == 2:
                key = columns[0].text.strip()
                value = columns[1].text.strip()
                data[key] = value

        # 데이터프레임으로 변환
        df = pd.DataFrame(list(data.items()), columns=['Factor', 'Value'])

        return df

    except Exception as e:
        print(f"Error fetching data from Finviz for {ticker}: {e}")
        return pd.DataFrame()

def get_latest_erp_file():
    """가장 최근 날짜의 ERP 데이터 파일을 찾습니다."""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'data')
    erp_files = [f for f in os.listdir(data_path) if f.startswith('erp_data_') and f.endswith('.csv')]
    
    if not erp_files:
        raise FileNotFoundError("ERP 데이터 파일을 찾을 수 없습니다.")
    
    # 파일명에서 날짜를 추출하여 가장 최근 파일을 찾음
    latest_file = max(erp_files, key=lambda x: x.replace('erp_data_', '').replace('.csv', ''))
    return os.path.join(data_path, latest_file)

def get_project_name(proj_code):
    """
    사업코드로 사업명을 검색합니다.
    :param proj_code: 사업코드 (예: 'C20170001')
    :return: 사업명 또는 None
    """
    try:
        erp_path = get_latest_erp_file()
        df = pd.read_csv(erp_path)
        result = df.loc[df['사업코드'] == proj_code, '사업명']
        return result.iloc[0] if not result.empty else None
    except Exception as e:
        print(f"Error getting project name for {proj_code}: {e}")
        return None

def search_projects(search_term):
    """
    사업명이나 사업코드로 프로젝트를 검색합니다.
    :param search_term: 검색어 (사업명 일부 또는 사업코드)
    :return: [(사업코드, 사업명)] 형태의 리스트
    """
    try:
        erp_path = get_latest_erp_file()
        df = pd.read_csv(erp_path)
        
        # 사업코드와 사업명 컬럼만 선택하고 중복 제거
        df = df[['사업코드', '사업명']].drop_duplicates()
        
        # 검색어를 포함하는 사업코드나 사업명 검색
        mask = (df['사업코드'].str.contains(search_term, case=False, na=False) |
                df['사업명'].str.contains(search_term, case=False, na=False))
        
        results = df[mask].values.tolist()
        return [(code, name) for code, name in results]
    except Exception as e:
        print(f"Error searching projects: {e}")
        return []

async def search_projects_and_respond(ctx, query):
    """
    Discord 봇을 위한 프로젝트 검색 및 응답 함수
    :param ctx: Discord 컨텍스트
    :param query: 검색어
    """
    matching_projects = search_projects(query)
    if not matching_projects:
        await ctx.send("검색 결과가 없습니다.")
        return
    
    response_message = "검색 결과:\n"
    response_messages = []
    
    for proj_code, name in matching_projects:
        line = f"{proj_code} - {name}\n"
        if len(response_message) + len(line) > 2000:  # Discord 메시지 길이 제한
            response_messages.append(response_message)
            response_message = "검색 결과 (계속):\n"
        response_message += line
    
    if response_message:
        response_messages.append(response_message)
    
    for message in response_messages:
        await ctx.send(message)
    print(f'검색어 "{query}"에 대한 메시지를 전송했습니다.')

def is_valid_project(proj_code):
    """
    사업코드가 유효한지 확인합니다.
    :param proj_code: 확인할 사업코드
    :return: bool
    """
    try:
        erp_path = get_latest_erp_file()
        df = pd.read_csv(erp_path)
        return proj_code in df['사업코드'].values
    except Exception as e:
        print(f"Error checking project code: {e}")
        return False

if __name__ == "__main__":
    # 테스트 코드
    search_term = input("검색어를 입력하세요 (사업명 또는 사업코드): ")
    results = search_projects(search_term)
    
    if results:
        print(f"'{search_term}'에 대한 검색 결과:")
        for proj_code, name in results:
            print(f"사업코드: {proj_code}, 사업명: {name}")
    else:
        print(f"'{search_term}'에 대한 검색 결과가 없습니다.")

# python get_ticker.py
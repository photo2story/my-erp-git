# //my-flask-app/get_account_H_API.py

import mojito
import pandas as pd
import yfinance as yf
from get_account_util_API import send_to_discord, format_balance_message
import config
from dotenv import load_dotenv
import os
from pathlib import Path

# 환경변수 로드 - 명시적 경로 지정
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# 환경변수 디버깅
print("\n=== 환경변수 확인 ===")

print(f"H_ACCOUNT: {os.getenv('H_ACCOUNT')}")
print("=====================\n")

async def get_kis_account_balance():
    """한국투자증권 계좌 잔고를 조회합니다."""
    try:
        # .env에서 API 키 로드
        key = os.getenv('H_APIKEY')
        secret = os.getenv('H_SECRET')
        acc_no = os.getenv('H_ACCOUNT')
        
        if not all([key, secret, acc_no]):
            raise ValueError("API 키, 시크릿, 계좌번호가 모두 필요합니다.")

        broker = mojito.KoreaInvestment(
            api_key=key,
            api_secret=secret,
            acc_no=acc_no,
        )
      
        print(f"계좌번호: {acc_no}")
        
        # 미국주식
        balance_US = broker.fetch_present_balance()
        output1_list_US = [
            {
                'ticker': comp['pdno'],
                'name': comp['prdt_name'],
                'profit_amount': comp['frcr_evlu_amt2'],
                'average_price': comp['frcr_pchs_amt'],
                'holding_quantity': comp.get('ccld_qty_smtl1', '0'),
                'profit_rate': comp['evlu_pfls_rt1'],
                'current_price': float(comp.get('ovrs_now_pric1', 0)),
                'total_value': float(comp.get('ovrs_now_pric1', 0)) * float(comp.get('ccld_qty_smtl1', 0))
            }
            for comp in balance_US['output1']
        ]
        
        # 한국주식
        balance_KR = broker.fetch_balance()
        output1_list_KR = [
            {
                'ticker': comp['pdno'],
                'name': comp['prdt_name'],
                'profit_amount': comp['evlu_amt'],
                'average_price': comp['pchs_amt'],
                'holding_quantity': comp.get('ord_psbl_qty', '0'),
                'profit_rate': comp['evlu_pfls_rt'],
                'current_price': float(comp.get('prpr', 0)) / 1000,
                'total_value': float(comp.get('prpr', 0)) * float(comp.get('ord_psbl_qty', 0)) / 1000
            }
            for comp in balance_KR['output1']
        ]
        
        # Combine US and KR balances into one list
        combined_balance = output1_list_US + output1_list_KR
        
        # Calculate total value
        total_value = sum(float(item['total_value']) for item in combined_balance)
        
        # Get cash balance
        cash_balance = float(balance_US['output2'][0].get('nxdy_frcr_drwg_psbl_amt', '0'))  # 미국주식 예수금
        krw_balance = float(balance_KR['output2'][0].get('dnca_tot_amt', '0'))  # 한국주식 예수금
        
        # 미국주식 USD를 KRW로 변환
        exchange_rate = float(balance_US['output1'][0]['bass_exrt'])  # 환율
        usd_total_value = sum(float(item['total_value']) for item in output1_list_US)
        krw_total_value = sum(float(item['total_value']) for item in output1_list_KR)
        total_value_krw = (usd_total_value * exchange_rate) + krw_total_value
        
        # 콘솔 출력용 메시지 생성
        console_message = [
            "\n=== 한국투자증권 계좌 잔고 ===",
            f"전체 보유자산 평가금액: {total_value_krw + (cash_balance * exchange_rate) + krw_balance:,.2f} KRW",
            f"보유 종목 수: {len(combined_balance)}개\n"
        ]
        
        # 미국주식 출력
        if output1_list_US:
            console_message.append("[ 미국주식 ]")
            for stock in output1_list_US:
                console_message.extend([
                    f"• {stock['ticker']} ({stock['name']})",
                    f"  수량: {float(stock['holding_quantity']):,.4f}",
                    f"  현재가: ${stock['current_price']:,.2f}",
                    f"  평가금액: ${stock['total_value']:,.2f} (₩{stock['total_value'] * exchange_rate:,.0f})",
                    f"  수익률: {float(stock['profit_rate']):,.2f}%\n"
                ])

        # 한국주식 출력
        if output1_list_KR:
            console_message.append("[ 한국주식 ]")
            for stock in output1_list_KR:
                console_message.extend([
                    f"• {stock['ticker']} ({stock['name']})",
                    f"  수량: {float(stock['holding_quantity']):,.0f}",
                    f"  현재가: {stock['current_price']:,.0f} KRW",
                    f"  평가금액: {stock['total_value']:,.0f} KRW",
                    f"  수익률: {float(stock['profit_rate']):,.2f}%\n"
                ])

        # 예수금 정보 추가
        console_message.extend([
            "=== 한국투자증권 예수금 현황 ===",
            f"미국주식 예수금: ${cash_balance:,.2f} (₩{cash_balance * exchange_rate:,.0f})",
            f"한국주식 예수금: {krw_balance:,.0f} KRW",
            f"총 예수금: {(cash_balance * exchange_rate) + krw_balance:,.0f} KRW\n"
        ])
        
        # 콘솔에 출력
        print("\n".join(console_message))
        
        # Discord로 메시지 전송
        if os.getenv('DISCORD_WEBHOOK_URL'):
            await send_to_discord(os.getenv('DISCORD_WEBHOOK_URL'), "\n".join(console_message))
        
        return combined_balance
            
    except Exception as e:
        error_msg = f"잔고 조회 중 오류 발생: {e}"
        print(error_msg)
        return {'error': error_msg}

def calculate_buyable_balance(key, secret, acc_no):
    """매수 가능 금액을 계산합니다."""
    broker = mojito.KoreaInvestment(
        api_key=key,
        api_secret=secret,
        acc_no=acc_no,
    )
  
    balance = broker.fetch_present_balance()
  
    if not balance['output2']:
        us_psbl_amt = 0
    else:
        us_psbl_amt = float(balance['output2'][0].get('nxdy_frcr_drwg_psbl_amt', '0'))

    won_psbl_amt = float(balance['output3'].get('wdrw_psbl_tot_amt', '0').replace(',', ''))
    print(f'won_psbl_amt: {won_psbl_amt}')
    print(f'us_psbl_amt: {us_psbl_amt}')
  
    frst_bltn_exrt = float(balance['output1'][0]['bass_exrt'])
  
    buyable_balance = (us_psbl_amt + won_psbl_amt / frst_bltn_exrt)
    return won_psbl_amt, us_psbl_amt, frst_bltn_exrt, buyable_balance

async def get_ticker_info(key, secret, acc_no, exchange, ticker, price, quantity):
    """종목 정보를 조회합니다."""
    broker = mojito.KoreaInvestment(key, secret, acc_no, exchange)
    resp = await broker.create_limit_buy_order(ticker, price, quantity)
    return resp

async def get_ticker_price(key, secret, acc_no, exchange, ticker):
    """종목의 현재가를 조회합니다."""
    if 'KOSPI' in exchange or 'KOSDAQ' in exchange:
        ticker = ticker.replace(".KS", "").replace(".KQ", "")
        broker = mojito.KoreaInvestment(api_key=key, api_secret=secret, acc_no=acc_no)
        price_data = await broker.fetch_price(ticker)
        last_price = price_data['output']['stck_oprc']
    else:
        if exchange == 'NASDAQ':
            exchange = '나스닥'
        elif exchange == 'NYSE':
            exchange = '뉴욕'
        elif exchange == 'AMEX':
            exchange = '아멕스'

        broker = mojito.KoreaInvestment(api_key=key, api_secret=secret, acc_no=acc_no, exchange=exchange)
        price_data = await broker.fetch_price(ticker)
        print(price_data)
        last_price = price_data['output']['last']
        print(last_price)
        if last_price is None:
            stock = yf.Ticker(ticker)
            last_price = stock.history(period='1d')['Close'][0]

    print(f"Last price for {ticker} on {exchange}: {last_price}")
    return last_price

def get_market_from_ticker(ticker):
    """티커로부터 마켓 정보를 조회합니다."""
    df = pd.read_csv(config.CSV_PATH, na_values=['', 'NaN'])
    ticker = ticker.upper()
    row = df[df['Symbol'] == ticker]
    if not row.empty:
        market = row['Market'].values[0]
        return market
    else:
        return "알 수 없는 마켓"

def get_ticker_from_korean_name(korean_name):
    """한글 종목명으로부터 티커를 조회합니다."""
    df = pd.read_csv('stock_market.csv', na_values=['', 'NaN'])
    row = df[df['Name'].str.upper() == korean_name]
    if not row.empty:
        ticker = row['Symbol'].values[0]
        print(f"k_stock {ticker} : {ticker}")
        return ticker
    else:
        return None 

# 테스트 코드
if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("\n=== 한국투자증권 계좌 테스트 ===")
        try:
            result = await get_kis_account_balance()
            if isinstance(result, dict) and result.get('error'):
                print(f"❌ 오류 발생: {result['error']}")
            else:
                print("✅ 계좌 조회 성공")
        except Exception as e:
            print(f"❌ 예외 발생: {str(e)}")

    # 테스트 실행
    asyncio.run(test()) 
    
# python get_account_H_API.py
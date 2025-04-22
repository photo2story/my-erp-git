# //my-flask-app/get_account_balance_U.py
# 업비트 계좌 관리 전용

import os
import asyncio
import pandas as pd
from dotenv import load_dotenv
from get_account_upbit_API import (
    get_upbit_balance, 
    get_upbit_krw_balance, 
    place_upbit_order
)
from get_account_util_API import send_to_discord
import config

# Constants for rebalancing
MIN_ORDER_AMOUNT = 5000  # Minimum order amount in KRW
FEE_RATE = 0.0005  # 0.05% transaction fee
DECIMAL_PLACES = {
    'BTC': 8, 
    'ETH': 8, 
    'XRP': 2, 
    'DOGE': 2, 
    'SOL': 4
}  # Coin-specific decimal limits

def get_upbit_market_from_yahoo(yahoo_ticker):
    """Convert Yahoo ticker to Upbit market code (e.g., 'BTC-USD' -> 'KRW-BTC')."""
    if not yahoo_ticker.endswith('-USD'):
        return None
    coin = yahoo_ticker.split('-')[0]
    return f'KRW-{coin}'

def format_volume(volume, decimal_places):
    """Format order volume to Upbit standards with coin-specific decimal precision."""
    rounded = round(volume, decimal_places)
    formatted = f"{{:.{decimal_places}f}}".format(rounded).rstrip('0').rstrip('.')
    return formatted

async def fetch_balance_data():
    """Fetch Upbit balance and KRW data asynchronously."""
    upbit_balance = await get_upbit_balance()
    upbit_krw = await get_upbit_krw_balance()
    if not upbit_balance or not upbit_krw:
        return None, None
    return upbit_balance, upbit_krw

def calculate_holdings_and_targets(upbit_balance, total_asset_value, available_cash):
    """현재 보유 자산과 목표 금액 계산"""
    current_holdings = {}
    target_amounts = {}

    # 현재 보유 자산
    for coin in upbit_balance:
        ticker = f"{coin['ticker']}-USD"
        current_holdings[ticker] = {
            'value': coin['total_value'],
            'ratio': (coin['total_value'] / total_asset_value) * 100,
            'market': f"KRW-{coin['ticker']}"
        }
    current_holdings['CASH'] = {
        'value': available_cash,
        'ratio': (available_cash / total_asset_value) * 100,
        'market': 'KRW'
    }

    # 목표 금액 (config에서 설정된 비율 사용)
    for ticker, ratio in config.REBALANCING_CONFIG.items():
        if ticker == 'ALT_COINS':
            for alt_ticker, alt_ratio in ratio.items():
                target_amounts[alt_ticker] = {
                    'value': (alt_ratio / 100) * total_asset_value,
                    'ratio': alt_ratio
                }
        else:
            target_amounts[ticker] = {
                'value': (ratio / 100) * total_asset_value,
                'ratio': ratio
            }

    return current_holdings, target_amounts

async def print_balance():
    """Fetch and display Upbit account balance with asset ratios."""
    load_dotenv()
    upbit_balance, upbit_krw = await fetch_balance_data()
    if not upbit_balance or not upbit_krw:
        return "Error: Failed to fetch balance data."

    total_asset_value = sum(coin['total_value'] for coin in upbit_balance) + float(upbit_krw.get('available', 0))
    message = [
        "=== 업비트 계좌 현황 ===",
        f"전체 보유자산 평가금액: {total_asset_value:,.2f} KRW",
        f"보유 종목 수: {len(upbit_balance)}개\n"
    ]
    
    for coin in upbit_balance:
        ratio = (coin['total_value'] / total_asset_value) * 100
        message.extend([
            f"• {coin['ticker']}",
            f"  주문가능: {coin['holding_quantity']:.8f}",
            f"  주문중: {coin['locked_quantity']:.8f}",
            f"  매수평균가: {coin['avg_buy_price']:,.2f} {coin['unit_currency']}",
            f"  현재가: {coin['current_price']:,.2f} {coin['unit_currency']}",
            f"  평가금액: {coin['total_value']:,.2f} KRW",
            f"  비중: {ratio:.2f}%",
            f"  수익률: {coin['profit_rate']:+.2f}%",
            f"  24시간 변동률: {coin['change_rate']:+.2f}%\n"
        ])

    message.append(f"• 현금: {float(upbit_krw['available']):,.2f} KRW ({(float(upbit_krw['available']) / total_asset_value * 100):.2f}%)")
    return "\n".join(message)

async def plan_rebalancing():
    """리밸런싱 계획을 수립하고 디스코드로 전송"""
    load_dotenv()
    webhook_url = getattr(config, 'DISCORD_WEBHOOK_URL', None)

    upbit_balance, upbit_krw = await fetch_balance_data()
    if not upbit_balance or not upbit_krw:
        error_msg = "Error: Failed to fetch balance data."
        if webhook_url:
            await send_to_discord(webhook_url, error_msg)
        return error_msg

    total_asset_value = sum(coin['total_value'] for coin in upbit_balance) + float(upbit_krw.get('available', 0))
    available_cash = float(upbit_krw.get('available', 0))

    current_holdings, target_amounts = calculate_holdings_and_targets(upbit_balance, total_asset_value, available_cash)
    
    message = [
        "=== 업비트 자산별 리밸런싱 플랜 ===",
        f"📊 전체 자산: {total_asset_value:,.0f} KRW",
        f"💰 사용 가능한 현금: {available_cash:,.0f} KRW\n",
        "=== 현재 자산 상태 ==="
    ]

    needs_rebalancing = False
    for ticker, holding in current_holdings.items():
        if ticker == 'CASH':
            continue
            
        target = target_amounts.get(ticker, {'value': 0, 'ratio': 0})
        diff = holding['value'] - target['value']
        diff_ratio = abs(diff / target['value'] * 100) if target['value'] > 0 else 0
        
        if diff_ratio > 10 and abs(diff) >= MIN_ORDER_AMOUNT:
            needs_rebalancing = True
            
        status = "✅ 목표 내" if diff_ratio <= 10 else "❌ 조정 필요"
        message.extend([
            f"\n• {ticker}:",
            f"  현재: {holding['value']:,.0f} KRW ({holding['ratio']:.1f}%)",
            f"  목표: {target['value']:,.0f} KRW ({target['ratio']:.1f}%)",
            f"  차이: {diff:,.0f} KRW",
            f"  상태: {status}"
        ])

    if not needs_rebalancing:
        message.append("\n✨ 모든 자산이 목표 ±10% 내에 있어 리밸런싱 불필요.")

    plan_message = "\n".join(message)
    if webhook_url:
        await send_to_discord(webhook_url, plan_message)
    return plan_message

async def execute_rebalancing():
    """Execute rebalancing with market orders."""
    load_dotenv()
    webhook_url = getattr(config, 'DISCORD_WEBHOOK_URL', None)

    upbit_balance, upbit_krw = await fetch_balance_data()
    if not upbit_balance or not upbit_krw:
        error_msg = "Error: Failed to fetch balance data."
        if webhook_url:
            await send_to_discord(webhook_url, error_msg)
        return error_msg

    total_asset_value = sum(coin['total_value'] for coin in upbit_balance) + float(upbit_krw.get('available', 0))
    available_cash = float(upbit_krw.get('available', 0))

    current_holdings, target_amounts = calculate_holdings_and_targets(upbit_balance, total_asset_value, available_cash)
    
    orders = []
    for ticker, holding in current_holdings.items():
        if ticker == 'CASH':
            continue
            
        target = target_amounts.get(ticker, {'value': 0, 'ratio': 0})
        diff = target['value'] - holding['value']
        diff_ratio = abs(diff / target['value'] * 100) if target['value'] > 0 else 0
        
        if diff_ratio > 10 and abs(diff) >= MIN_ORDER_AMOUNT:
            market = get_upbit_market_from_yahoo(ticker)
            if not market:
                continue
                
            order = {
                'market': market,
                'side': 'bid' if diff > 0 else 'ask',
                'amount': abs(diff)
            }
            orders.append(order)

    if not orders:
        message = "✨ 모든 자산이 목표 ±10% 내에 있어 리밸런싱 불필요."
        if webhook_url:
            await send_to_discord(webhook_url, message)
        return message

    results = []
    for order in orders:
        if order['side'] == 'ask':
            # 매도 주문
            price = next((coin['current_price'] for coin in upbit_balance 
                         if f"KRW-{coin['ticker']}" == order['market']), None)
            if not price:
                continue
                
            volume = format_volume(order['amount'] / price, 
                                 DECIMAL_PLACES.get(order['market'].split('-')[1], 8))
            result = await place_upbit_order(
                order['market'], 'ask', volume, None, 'market'
            )
            results.append(result)
        else:
            # 매수 주문
            if available_cash < MIN_ORDER_AMOUNT:
                break
                
            buy_amount = min(order['amount'], available_cash * (1 - FEE_RATE))
            if buy_amount < MIN_ORDER_AMOUNT:
                continue
                
            result = await place_upbit_order(
                order['market'], 'bid', None, buy_amount, 'price'
            )
            if not result.get('error'):
                available_cash -= buy_amount * (1 + FEE_RATE)
            results.append(result)

    execution_message = "\n".join([str(r) for r in results]) if results else "주문 실행 실패"
    if webhook_url:
        await send_to_discord(webhook_url, execution_message)
    return execution_message

async def test():
    """Test the Upbit account functionality"""
    print("\n=== 업비트 계좌 테스트 시작 ===")
    
    print("\n1. 계좌 잔고 조회 테스트")
    balance_result = await print_balance()
    print(balance_result)
    
    print("\n2. 리밸런싱 계획 수립 테스트")
    plan_result = await plan_rebalancing()
    print(plan_result)
    
    print("\n=== 테스트 완료 ===")

def get_signal_info(ticker):
    """CSV 파일에서 시그널과 MRI 정보를 읽어옵니다."""
    try:
        csv_path = os.path.join(config.STATIC_IMAGES_PATH, f'result_VOO_{ticker}.csv')
        if not os.path.exists(csv_path):
            print(f"Warning: Signal file not found for {ticker}")
            return None, None
            
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"Warning: Empty signal file for {ticker}")
            return None, None
            
        # 마지막 유효한 시그널과 MRI 값 찾기
        last_valid_row = df.iloc[-1]
        signal = last_valid_row['signal']
        mri = float(last_valid_row['MRI'])
        
        # 시그널 분석
        cash_ratio = 0  # 기본값
        if isinstance(signal, str):
            if 'cash_' in signal:
                # 기존 cash_XX% 형식
                cash_ratio = int(signal.split('cash_')[1].split('%')[0])
            elif 'Monthly invest' in signal:
                # Monthly invest XX% 형식
                investment_ratio = float(signal.split('Monthly invest ')[1].split('%')[0])
                cash_ratio = 100 - investment_ratio  # 투자 비율의 반대가 현금 비율
            else:
                print(f"Warning: Unknown signal format for {ticker}: {signal}")
            
        return cash_ratio, mri
    except Exception as e:
        print(f"Error reading signal file for {ticker}: {e}")
        return None, None

if __name__ == "__main__":
    asyncio.run(test())

# python get_account_balance_U.py 
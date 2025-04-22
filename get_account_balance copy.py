# //my-flask-app/get_account_balance.py
# 업비트 계좌 관리 전용

import os
import argparse
import asyncio
import aiohttp
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
DECIMAL_PLACES = {'BTC': 8, 'ETH': 8, 'XRP': 2, 'DOGE': 2, 'SOL': 4}  # Coin-specific decimal limits

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

    # 목표 금액
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

def generate_orders(current_holdings, target_amounts):
    """매도/매수 주문 생성"""
    sell_orders, buy_orders = [], []
    needs_rebalancing = False

    for ticker, target in target_amounts.items():
        current = current_holdings.get(ticker, {'value': 0, 'ratio': 0, 'market': get_upbit_market_from_yahoo(ticker)})
        diff = target['value'] - current['value']
        diff_ratio = abs(diff / target['value'] * 100) if target['value'] > 0 else 0

        if diff_ratio > 10 and abs(diff) >= MIN_ORDER_AMOUNT:
            needs_rebalancing = True
            order = {'market': current['market'], 'amount': abs(diff), 'ticker': ticker}
            (sell_orders if diff < 0 else buy_orders).append(order)

    return sell_orders, buy_orders, needs_rebalancing

def format_detailed_plan(total_asset_value, available_cash, current_holdings, target_amounts, sell_orders, buy_orders, needs_rebalancing):
    """자산별 리밸런싱 플랜을 상세히 포맷팅"""
    message = [
        "=== 업비트 자산별 리밸런싱 플랜 ===",
        f"📊 전체 자산: {total_asset_value:,.0f} KRW",
        f"💰 사용 가능한 현금: {available_cash:,.0f} KRW\n",
        "=== 현재 자산 상태 ==="
    ]

    # 현재 자산 상태
    for ticker, holding in current_holdings.items():
        target = target_amounts.get(ticker, {'value': 0, 'ratio': 0})
        diff = holding['value'] - target['value']
        
        # target['value']가 0인 경우 처리
        if target['value'] == 0:
            status = "⚠️ 설정 없음"
            diff_percentage = 0
        else:
            diff_percentage = abs(diff / target['value'] * 100)
            status = "✅ 목표 내" if diff_percentage <= 10 else "❌ 조정 필요"
            
        message.append(
            f"• {ticker}: {holding['value']:,.0f} KRW ({holding['ratio']:.1f}%)"
            f"\n  목표: {target['value']:,.0f} KRW ({target['ratio']:.1f}%)"
            f"\n  차이: {diff:,.0f} KRW - {status} \n"
        )

    # 리밸런싱 필요 여부 및 주문
    if not needs_rebalancing:
        message.append("\n✨ 모든 자산이 목표 ±10% 내에 있어 리밸런싱 불필요.")
    else:
        if sell_orders:
            message.append("\n=== 매도 주문 ===")
            for order in sell_orders:
                message.append(f"• {order['ticker']}: {order['amount']:,.0f} KRW 매도")
        if buy_orders:
            message.append("\n=== 매수 주문 ===")
            for order in buy_orders:
                message.append(f"• {order['ticker']}: {order['amount']:,.0f} KRW 매수")

    return "\n".join(message)

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
            f"  24시간 변동률: {coin['change_rate']:+.2f}%",
            f"  평단가수정: {'Yes' if coin['avg_buy_price_modified'] else 'No'}\n"
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
    sell_orders, buy_orders, needs_rebalancing = generate_orders(current_holdings, target_amounts)

    plan_message = format_detailed_plan(
        total_asset_value, available_cash, current_holdings, target_amounts, sell_orders, buy_orders, needs_rebalancing
    )

    # 디스코드로 전송
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
    sell_orders, buy_orders, needs_rebalancing = generate_orders(current_holdings, target_amounts)

    if not needs_rebalancing:
        message = ["✨ 모든 자산이 목표 ±10% 내에 있어 리밸런싱 불필요."]
        
        if webhook_url:
            await send_to_discord(webhook_url, "\n".join(message))
        return "\n".join(message)

    current_prices = {f"KRW-{coin['ticker']}": float(coin['current_price']) for coin in upbit_balance}
    results = []

    # Execute sell orders
    for order in sell_orders:
        price = current_prices.get(order['market'])
        if not price:
            print(f"Skipping {order['market']}: No price data.")
            continue
        volume = format_volume(order['amount'] / price, DECIMAL_PLACES.get(order['market'].split('-')[1], 8))
        if float(volume) * price < MIN_ORDER_AMOUNT:
            continue
        result = await place_upbit_order(order['market'], 'ask', volume, None, 'market')
        results.append(result)

    # Refresh cash balance after sells
    _, upbit_krw = await fetch_balance_data()
    remaining_cash = float(upbit_krw.get('available', 0)) if upbit_krw else 0

    # Execute buy orders
    for order in buy_orders:
        if remaining_cash < MIN_ORDER_AMOUNT:
            break
        buy_amount = min(order['amount'], remaining_cash * (1 - FEE_RATE))
        if buy_amount < MIN_ORDER_AMOUNT:
            continue
        result = await place_upbit_order(order['market'], 'bid', None, buy_amount, 'price')
        if not result.get('error'):
            remaining_cash -= buy_amount * (1 + FEE_RATE)
        results.append(result)

    execution_message = "\n".join([str(r) for r in results]) if results else "No orders executed."
    if webhook_url:
        await send_to_discord(webhook_url, execution_message)
    return execution_message

# 테스트 코드
async def test():
    """Test the rebalancing functionality"""
    print("\n=== 업비트 리밸런싱 테스트 시작 ===")
    
    print("\n1. 계좌 잔고 조회 테스트")
    balance_result = await print_balance()
    print(balance_result)
    
    print("\n2. 리밸런싱 계획 수립 테스트")
    plan_result = await plan_rebalancing()
    print(plan_result)
    
    print("\n=== 테스트 완료 ===")

if __name__ == "__main__":
    # 테스트 실행
    asyncio.run(test())

# python get_account_balance.py

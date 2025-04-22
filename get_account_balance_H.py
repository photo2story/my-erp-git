# //my-flask-app/get_account_balance_H.py
# 한국투자증권 계좌 관리 전용

import os
import asyncio
import pandas as pd
from dotenv import load_dotenv
from get_account_H_API import (
    get_kis_account_balance,
    calculate_buyable_balance,
    get_ticker_info,
    get_ticker_price
)
from get_account_util_API import send_to_discord
import config

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

async def get_balance_plan():
    """발란싱 플랜을 생성합니다."""
    load_dotenv()
    holdings = await get_kis_account_balance()
    
    if not holdings:
        return "보유 중인 자산이 없습니다."
        
    # 1. 총 자산을 N등분 (기본 배분액 설정)
    total_assets = sum(float(holding['total_value']) for holding in holdings)
    num_assets = len(holdings)
    base_amount = total_assets / num_assets
    
    messages = [
        "=== 한국투자증권 포트폴리오 발란싱 플랜 ===",
        f"총 자산: ${total_assets:,.2f}",
        f"기본 배분 금액: ${base_amount:,.2f} (자산당 {100/num_assets:.1f}%)",
        "\n보유 자산 현황:"
    ]
    
    rebalancing_needed = False
    for holding in holdings:
        ticker = holding['ticker']
        current_value = float(holding['total_value'])
        current_price = float(holding['current_price'])
        profit_rate = float(holding['profit_rate'])
        
        # 2. 백테스팅 결과 읽기 (시그널, MRI)
        cash_ratio, mri = get_signal_info(ticker)
        if cash_ratio is None:
            cash_ratio = 0  # 시그널 파일이 없으면 기본값 사용
        if mri is None:
            mri = 0
        
        # 3. 스탑로스 체크
        stop_loss = -10.0 if mri >= 35 else -5.0  # MRI에 따른 스탑로스 설정
        stop_loss_triggered = float(profit_rate) <= stop_loss
        
        # 4. 최종 목표금액 설정
        if stop_loss_triggered:
            # 스탑로스 발동 시 전량 매도 (현금 100%)
            target_amount = 0
            action = "전량 매도 (스탑로스)"
        else:
            # 시그널의 현금 비율에 따른 목표금액 조정
            stock_ratio = 100 - cash_ratio
            target_amount = base_amount * (stock_ratio / 100)
            action = f"주식 {stock_ratio}% 유지"
        
        # 5. 현재 상태와 비교
        difference = target_amount - current_value
        difference_ratio = abs((current_value - target_amount) / target_amount) * 100 if target_amount > 0 else 100
        
        # 10% 이상 차이나거나 스탑로스 발동 시 조정 필요
        action_needed = difference_ratio > 10 or stop_loss_triggered
        if action_needed:
            rebalancing_needed = True
        
        messages.extend([
            f"\n• {holding['ticker']} ({holding['name']}):",
            f"  현재 상태:",
            f"    보유수량: {float(holding['holding_quantity']):.4f}주",
            f"    현재가격: ${current_price:.2f}",
            f"    평가금액: ${current_value:,.2f}",
            f"    수익률: {profit_rate:.2f}%",
            f"  백테스트 정보:",
            f"    MRI: {mri:.2f}",
            f"    시그널 현금비율: {cash_ratio}%",
            f"    스탑로스 기준: {stop_loss}%",
            f"  조치사항:",
            f"    스탑로스: {'발동' if stop_loss_triggered else '미발동'}",
            f"    목표금액: ${target_amount:,.2f}",
            f"    필요조치: {action}",
            f"    조정금액: ${abs(difference):,.2f}"
        ])
    
    messages.append(f"\n전체 판단: {'리밸런싱 필요' if rebalancing_needed else '현재 상태 유지'}")
    return "\n".join(messages)

async def execute_balance():
    """발란싱을 실행합니다."""
    load_dotenv()
    holdings = await get_kis_account_balance()
    
    if not holdings:
        return "보유 중인 자산이 없습니다."
        
    # 1. 총 자산을 N등분 (기본 배분액 설정)
    total_assets = sum(float(holding['total_value']) for holding in holdings)
    num_assets = len(holdings)
    base_amount = total_assets / num_assets
    
    messages = [
        "=== 한국투자증권 포트폴리오 발란싱 실행 ===",
        f"총 자산: ${total_assets:,.2f}",
        "\n실행할 주문:"
    ]
    
    orders_executed = False
    for holding in holdings:
        ticker = holding['ticker']
        current_value = float(holding['total_value'])
        current_price = float(holding['current_price'])
        profit_rate = float(holding['profit_rate'])
        
        # 2. 백테스팅 결과 읽기 (시그널, MRI)
        cash_ratio, mri = get_signal_info(ticker)
        if cash_ratio is None:
            cash_ratio = 0
        if mri is None:
            mri = 0
        
        # 3. 스탑로스 체크
        stop_loss = -10.0 if mri >= 35 else -5.0
        stop_loss_triggered = float(profit_rate) <= stop_loss
        
        # 4. 최종 목표금액 설정
        if stop_loss_triggered:
            target_amount = 0
            action_reason = "스탑로스"
        else:
            stock_ratio = 100 - cash_ratio
            target_amount = base_amount * (stock_ratio / 100)
            action_reason = "발란싱"
        
        # 5. 주문 실행 여부 결정
        difference = target_amount - current_value
        difference_ratio = abs((current_value - target_amount) / target_amount) * 100 if target_amount > 0 else 100
        
        if difference_ratio > 10 or stop_loss_triggered:
            orders_executed = True
            action = "매수" if difference > 0 else "매도"
            amount = abs(difference)
            shares = amount / current_price
            
            messages.extend([
                f"\n• {holding['ticker']} ({holding['name']}):",
                f"  {action}: ${amount:,.2f} ({shares:.4f}주)",
                f"  현재가격: ${current_price:.2f}",
                f"  수익률: {profit_rate:.2f}%",
                f"  MRI: {mri:.2f}",
                f"  사유: {action_reason}"
            ])
            
            # TODO: 실제 주문 실행 로직 추가
            # order_result = await execute_order(ticker, action, amount)
            # messages.append(f"  주문 결과: {order_result}")
    
    if not orders_executed:
        messages.append("\n실행할 주문이 없습니다. 현재 상태가 적절합니다.")
    
    return "\n".join(messages)

async def print_balance():
    """Fetch and display Korea Investment account balance."""
    load_dotenv()
    balance = await get_kis_account_balance()
    
    if isinstance(balance, dict) and balance.get('error'):
        return f"Error: {balance['error']}"
    
    return balance  # get_kis_account_balance() already returns formatted message

def calculate_buyable_amount():
    """매수 가능 금액을 계산합니다."""
    load_dotenv()
    key = os.getenv('H_APIKEY')
    secret = os.getenv('H_SECRET')
    acc_no = os.getenv('H_ACCOUNT')
    
    if not all([key, secret, acc_no]):
        return "Error: API credentials not found"
        
    won_psbl_amt, us_psbl_amt, frst_bltn_exrt, buyable_balance = calculate_buyable_balance(key, secret, acc_no)
    
    message = [
        "=== 한국투자증권 매수 가능 금액 ===",
        f"원화 매수 가능 금액: {won_psbl_amt:,.0f} KRW",
        f"달러 매수 가능 금액: ${us_psbl_amt:,.2f}",
        f"환율: {frst_bltn_exrt:,.2f} KRW/USD",
        f"총 매수 가능 금액: ${buyable_balance:,.2f}"
    ]
    
    return "\n".join(message)

# 테스트 코드
async def test():
    """Test the Korea Investment account functionality"""
    print("\n=== 한국투자증권 계좌 테스트 시작 ===")
    
    print("\n1. 계좌 잔고 조회 테스트")
    balance_result = await print_balance()
    print(balance_result)
    
    print("\n2. 매수 가능 금액 조회 테스트")
    buyable_result = calculate_buyable_amount()
    print(buyable_result)
    
    print("\n3. 발란싱 플랜 테스트")
    plan_result = await get_balance_plan()
    print(plan_result)
    
    print("\n=== 테스트 완료 ===")

if __name__ == "__main__":
    # 테스트 실행
    asyncio.run(test())

# python get_account_balance_H.py 
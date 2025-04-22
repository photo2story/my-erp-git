# Strategy_sell.py


from datetime import datetime

def strategy_sell(shares, cash, sell_price, option_strategy, mri, hedge_50_active, current_date,
                  rsi_ta=None, account_balance=0, ticker=None, ppo_histogram=0):
    shares_to_sell = 0
    signal = ''
    sell_ratio = 0.0  # 매도 비율 초기화

    # ✅ default 전략은 매도 없음
    if option_strategy == 'default':
        return 0, '', hedge_50_active

    if shares <= 0 or account_balance == 0:
        return 0, '', hedge_50_active

    # 위험도 따라 목표 현금 비중 결정
    if mri >= 0.5 and (rsi_ta is None or rsi_ta > 45):
        target_cash_ratio = 1  # 90% 현금화
        condition = "Very Risky"
    elif mri >= 0.45 and (rsi_ta is None or rsi_ta > 40):
        target_cash_ratio = 0.8  # 80% 현금화
        condition = "Risky"
    elif mri >= 0.4 and (rsi_ta is None or rsi_ta > 35):
        target_cash_ratio = 0.5  # 50% 현금화
        condition = "Caution"
    else:
        target_cash_ratio = 0.0  # 홀딩
        condition = "Normal"

    current_cash_ratio = cash / account_balance

    if (
        target_cash_ratio > 0
        and ppo_histogram < -0.5
        # and (rsi_ta is None or rsi_ta > 20)
        and current_cash_ratio < target_cash_ratio
    ):
        # 목표 현금 비중을 달성하기 위한 매도 비율 계산
        sell_ratio = target_cash_ratio - current_cash_ratio
        shares_to_sell = int(shares * sell_ratio)
        
        if shares_to_sell > 0:
            hedge_50_active = True
            # 실제 매도 비율 계산 (소수점 2자리까지)
            actual_sell_ratio = (shares_to_sell / shares) * 100
            signal = f"Sell {actual_sell_ratio:.1f}% of holdings ({shares_to_sell} shares - {condition} - MRI:{mri:.2f})"
            print(f"[DEBUG] SELL - {signal}")

    return shares_to_sell, signal, hedge_50_active


if __name__ == "__main__":
    import pandas as pd
    shares = 1000
    cash = 5000
    sell_price = 100 * 0.995
    mri = 0.46
    hedge_50_active = False
    rsi_ta = 50
    ppo_histogram = -0.5
    account_balance = 20000
    current_date = pd.to_datetime("2025-06-01")

    for strategy in ['default', 'modified_mri_seasonal', 'hedge_50', 'hybrid', 'shannon', 'buy_and_hold']:
        shares_to_sell, signal, hedge = strategy_sell(
            shares, cash, sell_price, strategy, mri, hedge_50_active, current_date,
            rsi_ta=rsi_ta, account_balance=account_balance, ppo_histogram=ppo_histogram
        )
        print(f"Strategy: {strategy}")
        print(f"Shares to sell: {shares_to_sell}")
        print(f"Signal: {signal}")
        print(f"Hedge_50 Active: {hedge}")
        print()




# python Strategy_sell.py
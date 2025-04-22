# Strategy_buy.py

import pandas as pd

# strategy_buy.py

import pandas as pd

def strategy_buy(cash, price, option_strategy, mri, hedge_50_active, account_balance,
                 ticker, rsi_ta=None, ppo_histogram=0, portfolio_value=0):
    shares_to_buy = 0
    signal = ''
    buy_ratio = 0.0  # 매수 비율 초기화

    # ✅ default 전략은 무조건 매수
    if option_strategy == 'default':
        buy_ratio = 1.0  # 100% 매수
        shares_to_buy = int(cash / price)
        if shares_to_buy > 0:
            signal = f"Buy {buy_ratio*100:.0f}% of cash ({shares_to_buy} shares - Default Strategy)"
            print(f"[DEBUG] BUY - {signal}")
        return shares_to_buy, signal, hedge_50_active

    # 🔁 그 외 전략은 MRI 기반 매수
    if mri < 0.35:
        buy_ratio = 1.0  # 100%
        condition = "Very Safe"
    elif mri < 0.4:
        buy_ratio = 0.9  # 90%
        condition = "Safe"
    elif mri < 0.45:
        buy_ratio = 0.8  # 80%
        condition = "Caution"
    else:
        buy_ratio = 0.0  # 0%
        condition = "Very Risky"

    if (
        buy_ratio > 0
        and ppo_histogram > 0.5
        and (rsi_ta is None or rsi_ta < 85)
    ):
        invest = cash * buy_ratio
        shares_to_buy = int(invest / price)
        if shares_to_buy > 0:
            signal = f"Buy {buy_ratio*100:.0f}% of cash ({shares_to_buy} shares - {condition} - MRI:{mri:.2f})"
            print(f"[DEBUG] BUY - {signal}")

    return shares_to_buy, signal, hedge_50_active


if __name__ == "__main__":
    current_date = pd.to_datetime("2025-03-01")
    cash = 5000
    buy_price = 100
    mri = 0.28
    hedge_50_active = True
    for strategy in ['default', 'modified_mri_seasonal', 'hedge_50', 'hybrid', 'shannon']:
        shares, signal, hedge = strategy_buy(cash, buy_price, strategy, mri, hedge_50_active, cash, 'AAPL')
        print(f"Strategy: {strategy}")
        print(f"Shares to buy: {shares}")
        print(f"Signal: {signal}")
        print(f"Hedge_50 Active: {hedge}")
        print()


# python Strategy_buy.py
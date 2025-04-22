# //my-flask-app/get_account_util_API.py

import requests
import aiohttp

async def send_to_discord(webhook_url, message):
    """Send a message to Discord webhook with consistent profile."""
    data = {
        "username": "Captain Hook 🤖",
        "avatar_url": "https://i.imgur.com/4M34hi2.png",  # 원하는 이미지 URL로 변경하세요
        "content": message
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(webhook_url, json=data) as response:
            return response.status == 204  # Discord returns 204 on success

def format_balance_message(balance_list, total_value, title):
    """잔고 정보를 디스코드 메시지 형식으로 변환합니다."""
    message = [f"=== {title} ==="]
    message.append(f"전체 보유자산 평가금액: {total_value:,.2f} KRW")
    message.append(f"보유 종목 수: {len(balance_list)}개")
    message.append("")  # 빈 줄 추가
    
    for item in balance_list:
        message.append(f"• {item['name']} ({item['ticker']})")
        message.append(f"  수량: {item['holding_quantity']:.4f}")
        message.append(f"  현재가: {item['current_price']:.4f} KRW")
        message.append(f"  평가금액: {item['total_value']:.4f} KRW")
        message.append(f"  수익률: {item['profit_rate']:.2f}%")
        
        if 'change_rate_24h' in item:
            message.append(f"  24시간 변동률: {item['change_rate_24h']:.2f}%")
            
            if 'min_order_amount' in item:
                message.append(f"  최소 주문금액: {item['min_order_amount']:,.0f} KRW")
            message.append("")  # 각 종목 사이에 빈 줄 추가
        else:
            message.append("")
    
    return "\n".join(message)

def format_backtest_message(backtest_info, title="백테스트 결과"):
    """백테스트 결과를 디스코드 메시지 형식으로 변환합니다."""
    if backtest_info.get('error'):
        return f"=== {title} ===\n❌ 에러: {backtest_info['error']}"

    message = [f"=== {title} ===\n"]  # 제목 다음에 빈 줄 추가
    
    # 기본 정보
    message.append(f"📊 계좌 잔고: {backtest_info['account_balance']:,.0f}")
    message.append(f"💰 현금: {backtest_info['cash']:,.0f}")
    message.append(f"📈 포트폴리오 가치: {backtest_info['portfolio_value']:,.0f}")
    message.append(f"💵 현금 보유 비중: {backtest_info['cash_ratio']:.2f}%")
    message.append(f"📅 기준 일자: {backtest_info['date']}")
    message.append("")  # 기본 정보 다음에 빈 줄 추가
    
    # 최근 매매 신호
    if backtest_info['latest_signal'] != 'NONE':
        message.append(f"🔔 최근 매매 신호:")
        message.append(f"{backtest_info['latest_signal']}")
        message.append("")  # 매매 신호 다음에 빈 줄 추가
    
    # 디버깅 정보
    if 'debug' in backtest_info:
        debug = backtest_info['debug']
        message.append(f"📌 상세 정보:")
        message.append(f"• 마지막 신호 일자: {debug['last_signal_date']}")
        message.append(f"• 신호 이후 경과일: {debug['days_since_signal']} 일")
        message.append(f"• 전체 신호 수: {debug['total_signals']} 개")
        message.append("")  # 매매 신호 다음에 빈 줄 추가

    
    return "\n".join(message) 
# //my-flask-app/get_account_upbit_API.py

import jwt
import hashlib
import requests
import uuid
from urllib.parse import urlencode, unquote
from get_account_util_API import send_to_discord, format_balance_message
import config
import pandas as pd
import os
from datetime import datetime
import asyncio
from dotenv import load_dotenv
import aiohttp

def convert_upbit_to_yahoo_ticker(upbit_market):
    """
    Convert Upbit market code to Yahoo Finance ticker
    Example: 'KRW-BTC' -> 'BTC-USD'
    """
    if not upbit_market.startswith('KRW-'):
        return None
    
    # Extract the coin symbol
    coin_symbol = upbit_market.split('-')[1]
    
    # List of supported cryptocurrencies
    supported_coins = {
        'BTC': 'BTC-USD',
        'DOGE': 'DOGE-USD',
        'ETH': 'ETH-USD',
        'SOL': 'SOL-USD',
        'XRP': 'XRP-USD'
    }
    
    return supported_coins.get(coin_symbol)

async def check_crypto_backtest_signal(market):
    """
    Check trading signal from the latest backtest results for any cryptocurrency
    """
    try:
        # Convert Upbit market code to Yahoo Finance ticker
        yahoo_ticker = convert_upbit_to_yahoo_ticker(market)
        if not yahoo_ticker:
            return False, f"Unsupported market: {market}"
            
        csv_path = os.path.join(config.STATIC_IMAGES_PATH, f'result_VOO_{yahoo_ticker}.csv')
        if not os.path.exists(csv_path):
            return False, f"Backtest results not found for {yahoo_ticker}"
            
        # Read the last row of backtest results
        df = pd.read_csv(csv_path)
        if df.empty:
            return False, f"No backtest data available for {yahoo_ticker}"
            
        latest_data = df.iloc[-1]
        
        # Check MRI and signal conditions
        mri = latest_data.get('MRI', 0)
        ppo_histogram = latest_data.get('ppo_histogram', 0)
        rsi = latest_data.get('RSI_14', 50)
        
        if mri >= 0.5:
            return True, f"Very Risky Market (MRI: {mri:.2f})"
        elif mri >= 0.45 and ppo_histogram < 0:
            return True, f"Risky Market with Negative Momentum (MRI: {mri:.2f}, PPO: {ppo_histogram:.2f})"
        elif rsi > 70:
            return True, f"Overbought (RSI: {rsi:.2f})"
            
        return False, f"No sell signal from backtest for {yahoo_ticker}"
        
    except Exception as e:
        return False, f"Error checking backtest signal for {market}: {str(e)}"

async def get_valid_markets(server_url):
    """업비트에서 지원하는 유효한 마켓 코드 목록을 조회합니다."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(server_url + '/v1/market/all') as response:
                if response.status == 200:
                    markets = await response.json()
                    return {market['market'] for market in markets if market['market'].startswith('KRW-')}
                else:
                    print(f"마켓 목록 조회 실패: {response.status}")
                    return set()
    except Exception as e:
        print(f"마켓 목록 조회 중 오류: {e}")
        return set()

async def get_market_info(server_url, markets):
    """마켓별 최소 주문 금액 등의 정보를 조회합니다."""
    try:
        markets_param = ",".join(markets)
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{server_url}/v1/market/order-book", params={"markets": markets_param}) as response:
                if response.status == 200:
                    data = await response.json()
                    market_info = {}
                    for item in data:
                        market = item.get('market')
                        if market:
                            market_info[market] = {
                                'min_total': item.get('market', {}).get('ask', {}).get('min_total', 0),
                                'state': item.get('market', {}).get('state', 'unknown')
                            }
                    return market_info
    except Exception as e:
        print(f"마켓 정보 조회 중 오류 발생: {e}")
    return {}

async def get_upbit_krw_balance():
    """업비트 원화 잔고를 조회합니다."""
    try:
        # .env에서 API 키 로드
        access_key = os.getenv('UPBIT_ACCESS_KEY')
        secret_key = os.getenv('UPBIT_SECRET_KEY')
        server_url = os.getenv('UPBIT_SERVER_URL', 'https://api.upbit.com')

        # 계좌 잔고 조회를 위한 인증
        payload = {
            'access_key': access_key,
            'nonce': str(uuid.uuid4()),
        }

        jwt_token = jwt.encode(payload, secret_key)
        authorization = 'Bearer {}'.format(jwt_token)
        headers = {
            'Authorization': authorization,
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(server_url + "/v1/accounts", headers=headers) as response:
                if response.status == 200:
                    accounts = await response.json()
                    
                    # KRW 계좌 찾기
                    krw_account = next((acc for acc in accounts if acc['currency'] == 'KRW'), None)
                    
                    if krw_account:
                        return {
                            'available': krw_account['balance'],
                            'locked': krw_account['locked'],
                            'total': str(float(krw_account['balance']) + float(krw_account['locked']))
                        }
                    else:
                        return {'available': '0', 'locked': '0', 'total': '0'}
                else:
                    print(f"Error: {response.status}")
                    return None

    except Exception as e:
        print(f"Error fetching KRW balance: {str(e)}")
        return None

async def get_upbit_balance():
    """업비트 계좌 잔고를 조회합니다."""
    try:
        # .env에서 API 키 로드
        access_key = os.getenv('UPBIT_ACCESS_KEY')
        secret_key = os.getenv('UPBIT_SECRET_KEY')
        server_url = os.getenv('UPBIT_SERVER_URL', 'https://api.upbit.com')

        # 계좌 잔고 조회를 위한 인증
        payload = {
            'access_key': access_key,
            'nonce': str(uuid.uuid4()),
        }

        jwt_token = jwt.encode(payload, secret_key)
        authorization = 'Bearer {}'.format(jwt_token)
        headers = {
            'Authorization': authorization,
        }

        async with aiohttp.ClientSession() as session:
            # 계좌 잔고 조회
            async with session.get(server_url + "/v1/accounts", headers=headers) as response:
                if response.status == 200:
                    accounts = await response.json()
                    
                    if not accounts:
                        return []

                    balance_list = []
                    
                    # 잔고 정보 구성
                    for account in accounts:
                        if float(account['balance']) > 0 and account['currency'] != 'KRW':
                            market = f"KRW-{account['currency']}"
                            
                            # 현재가 조회
                            current_price = None
                            try:
                                async with session.get(f"{server_url}/v1/ticker?markets={market}") as price_response:
                                    if price_response.status == 200:
                                        prices = await price_response.json()
                                        if prices and len(prices) > 0:
                                            current_price = float(prices[0]['trade_price'])
                                            change_rate = float(prices[0]['signed_change_rate']) * 100
                            except Exception as e:
                                print(f"현재가 조회 실패 - {market}: {str(e)}")
                                current_price = None
                                change_rate = 0
                            
                            # 현재가를 가져오지 못한 경우 평균매수가 사용
                            if not current_price:
                                current_price = float(account['avg_buy_price'])
                                change_rate = 0
                            
                            holding_quantity = float(account['balance'])
                            locked_quantity = float(account['locked'])
                            avg_buy_price = float(account['avg_buy_price'])
                            total_value = (holding_quantity + locked_quantity) * current_price
                            
                            # 수익률 계산
                            profit_rate = ((current_price - avg_buy_price) / avg_buy_price * 100) if avg_buy_price > 0 else 0
                            
                            balance_list.append({
                                'ticker': account['currency'],
                                'holding_quantity': holding_quantity,
                                'locked_quantity': locked_quantity,
                                'avg_buy_price': avg_buy_price,
                                'avg_buy_price_modified': account['avg_buy_price_modified'],
                                'current_price': current_price,
                                'total_value': total_value,
                                'profit_rate': profit_rate,
                                'change_rate': change_rate,
                                'unit_currency': account['unit_currency']
                            })
                    return balance_list
                else:
                    print(f"Error: {response.status}")
                    return None

    except Exception as e:
        print(f"Error fetching balance: {str(e)}")
        return None

def format_order_message(result, current_price):
    """주문 결과를 디스코드 메시지 형식으로 변환합니다."""
    order_type = '매수' if result.get('side') == 'bid' else '매도'
    ord_type = result.get('ord_type', '')
    
    if ord_type == 'price':  # 시장가 매수
        price = float(result.get('price', 0))
        message = [
            f"=== 업비트 {order_type} 주문 체결 (시장가) ===",
            f"마켓: {result.get('market')}",
            f"주문 종류: {order_type}",
            f"주문 금액: {price:,.2f} KRW",
            f"예상 체결가: {current_price:,.2f} KRW",
            f"예상 체결 수량: {(price / current_price) if current_price else 0:.8f}",
            f"주문 UUID: {result.get('uuid')}",
            f"주문 시각: {result.get('created_at')}"
        ]
    elif ord_type == 'market':  # 시장가 매도
        volume = float(result.get('volume', 0))
        message = [
            f"=== 업비트 {order_type} 주문 체결 (시장가) ===",
            f"마켓: {result.get('market')}",
            f"주문 종류: {order_type}",
            f"주문 수량: {volume}",
            f"예상 체결가: {current_price:,.2f} KRW",
            f"예상 체결 금액: {volume * current_price if current_price else 0:,.2f} KRW",
            f"주문 UUID: {result.get('uuid')}",
            f"주문 시각: {result.get('created_at')}"
        ]
    else:  # 지정가 주문
        volume = float(result.get('volume', 0))
        price = float(result.get('price', 0))
        message = [
            f"=== 업비트 {order_type} 주문 체결 (지정가) ===",
            f"마켓: {result.get('market')}",
            f"주문 종류: {order_type}",
            f"주문 수량: {volume}",
            f"주문 가격: {price:,.2f} KRW",
            f"예상 체결 금액: {volume * price:,.2f} KRW",
            f"주문 UUID: {result.get('uuid')}",
            f"주문 시각: {result.get('created_at')}"
        ]
    
    return "\n".join(message)

async def place_upbit_order(market, side, volume=None, price=None, ord_type='limit'):
    """업비트 주문을 실행합니다.
    
    Args:
        market (str): 마켓 코드 (예: KRW-BTC)
        side (str): 주문 방향 ('bid': 매수, 'ask': 매도)
        volume (float, optional): 주문 수량 (지정가, 시장가 매도 시 필수)
        price (float, optional): 주문 가격 또는 금액 (지정가, 시장가 매수 시 필수)
        ord_type (str): 주문 타입 ('limit': 지정가, 'price': 시장가 매수, 'market': 시장가 매도)
    
    Returns:
        dict: 주문 결과
    """
    try:
        # .env에서 API 키 로드
        access_key = os.getenv('UPBIT_ACCESS_KEY')
        secret_key = os.getenv('UPBIT_SECRET_KEY')
        server_url = os.getenv('UPBIT_SERVER_URL', 'https://api.upbit.com')

        # 매도 주문인 경우 백테스트 시그널 확인 (시그널 전략일 때만)
        if side == 'ask' and market.startswith('KRW-') and config.current_strategy == config.STRATEGY_MODE['SIGNAL']:
            should_sell, reason = await check_crypto_backtest_signal(market)
            if not should_sell:
                error_message = f"매도 시그널이 없습니다. 사유: {reason}"
                if hasattr(config, 'DISCORD_WEBHOOK_URL'):
                    await send_to_discord(config.DISCORD_WEBHOOK_URL, f"=== 업비트 매도 주문 거부 ===\n{error_message}")
                return {'error': {'message': error_message}}

        # 현재가 조회
        current_price = None
        if ord_type in ['market', 'price']:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{server_url}/v1/ticker", params={"markets": market}) as price_res:
                        if price_res.status == 200:
                            price_data = await price_res.json()
                            if price_data:
                                current_price = price_data[0]['trade_price']
            except Exception as e:
                print(f"현재가 조회 중 오류: {e}")

        params = {
            'market': market,
            'side': side,
        }
        
        # 시장가 매수 (price 파라미터만 사용)
        if side == 'bid' and ord_type == 'price':
            params['ord_type'] = 'price'
            params['price'] = str(int(price))
        # 시장가 매도 (volume 파라미터만 사용)
        elif side == 'ask' and ord_type == 'market':
            params['ord_type'] = 'market'
            params['volume'] = str(volume)
        # 지정가 주문 (volume과 price 모두 사용)
        else:
            params['ord_type'] = 'limit'
            params['volume'] = str(volume)
            params['price'] = str(int(price))
        
        query_string = unquote(urlencode(params, doseq=True)).encode("utf-8")
        
        m = hashlib.sha512()
        m.update(query_string)
        query_hash = m.hexdigest()
        
        payload = {
            'access_key': access_key,
            'nonce': str(uuid.uuid4()),
            'query_hash': query_hash,
            'query_hash_alg': 'SHA512',
        }
        
        jwt_token = jwt.encode(payload, secret_key)
        authorization = 'Bearer {}'.format(jwt_token)
        headers = {
            'Authorization': authorization,
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(server_url + '/v1/orders?' + query_string.decode('utf-8'), headers=headers) as response:
                if response.status == 201:  # 성공
                    result = await response.json()
                    
                    # 디스코드로 주문 결과 전송
                    if hasattr(config, 'DISCORD_WEBHOOK_URL'):
                        message = format_order_message(result, current_price)
                        await send_to_discord(config.DISCORD_WEBHOOK_URL, message)
                    
                    return result
                else:
                    error_text = await response.text()
                    error_message = f"주문 실패: {response.status}\n에러 메시지: {error_text}"
                    print(error_message)
                    
                    # 실패 메시지도 디스코드로 전송
                    if hasattr(config, 'DISCORD_WEBHOOK_URL'):
                        error_discord = f"=== 업비트 주문 실패 ===\n마켓: {market}\n실패 사유: {error_text}"
                        await send_to_discord(config.DISCORD_WEBHOOK_URL, error_discord)
                    
                    return {'error': {'message': error_message}}
            
    except Exception as e:
        error_message = f"주문 중 오류 발생: {str(e)}"
        print(error_message)
        
        # 오류 메시지도 디스코드로 전송
        if hasattr(config, 'DISCORD_WEBHOOK_URL'):
            error_discord = f"=== 업비트 주문 오류 ===\n마켓: {market}\n오류 내용: {str(e)}"
            await send_to_discord(config.DISCORD_WEBHOOK_URL, error_discord)
        
        return {'error': {'message': error_message}}


# 테스트 코드
if __name__ == "__main__":
    load_dotenv()
    
    async def test():
        print("\n=== 업비트 계좌 테스트 ===")
        try:
            result = await get_upbit_balance()
            
            if isinstance(result, dict) and result.get('error'):
                print(f"❌ 오류 발생: {result['error']}")
            elif not result:
                print("❌ 계좌 조회 실패")
            else:
                print("✅ 계좌 조회 성공")
                print(result)
                
        except Exception as e:
            print(f"❌ 테스트 중 오류 발생: {str(e)}")
    
    asyncio.run(test())

# python get_account_upbit_API.py 
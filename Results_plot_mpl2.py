## Results_plot_mpl.py

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mplchart.chart import Chart
from mplchart.primitives import Candlesticks, Volume, Price, LinePlot
from mplchart.indicators import SMA, PPO, RSI
import pandas as pd
import requests
import os, sys
from dotenv import load_dotenv
import asyncio
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import platform

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from git_operations import move_files_to_images_folder
from get_ticker import get_ticker_name
from Get_data import get_post_market_prices

# 한글 폰트 설정
def set_korean_font():
    if platform.system() == 'Windows':
        font_path = 'C:/Windows/Fonts/malgun.ttf'  # 윈도우의 경우
    elif platform.system() == 'Darwin':  # macOS
        font_path = '/System/Library/Fonts/AppleGothic.ttf'
    else:  # Linux
        font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

def save_figure(fig, file_path):
    """파일 경로를 처리하여 그림을 저장하고 닫습니다."""
    file_path = os.path.join(config.STATIC_IMAGES_PATH, file_path.replace('/', '-'))
    fig.savefig(file_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Image saved at: {file_path}")

async def fetch_csv_from_remote(ticker):
    """원격 저장소에서 간략화된 CSV 파일을 다운로드합니다."""
    url = f"{config.STATIC_IMAGES_PATH}/result_VOO_{ticker}.csv"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.text), parse_dates=['Date'], index_col='Date')
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch data for {ticker} from remote: {e}")
        return None

async def plot_results_mpl(ticker, start_date, end_date, prices):
    """주어진 티커와 기간에 대한 데이터를 사용하여 차트를 생성하고, 결과를 Discord로 전송합니다."""

    try:
        # 데이터 프레임이 None이거나 비어있는지 확인
        if prices is None or prices.empty:
            print(f"No data available for {ticker}")
            return

        # 인덱스를 날짜 형식으로 변환
        if not isinstance(prices.index, pd.DatetimeIndex):
            prices['Date'] = pd.to_datetime(prices['Date'])
            prices.set_index('Date', inplace=True)

        # 모든 컬럼 이름을 소문자로 변환
        prices.columns = prices.columns.str.lower()

        # 필요한 컬럼들이 있는지 확인
        required_columns = ['sma05_ta', 'sma20_ta', 'sma60_ta', 'ppo_histogram', 'rsi_ta', 'close']
        missing_columns = [col for col in required_columns if col not in prices.columns]
        if missing_columns:
            print(f"Missing required columns for {ticker}: {missing_columns}")
            return

        # 데이터의 실제 마지막 날짜 사용
        actual_end_date = prices.index.max()
        start_date_6m = actual_end_date - pd.DateOffset(months=6)
        filtered_prices = prices[prices.index >= start_date_6m]
        
        if filtered_prices.empty:
            print(f"No data available for {ticker} in the specified date range")
            print(f"Data range: {prices.index.min()} to {prices.index.max()}")
            return
        
        # 최신 값들 추출
        try:
            latest_rsi = filtered_prices['rsi_ta'].iloc[-1]
            latest_ppo = filtered_prices['ppo_histogram'].iloc[-1]
            latest_close = filtered_prices['close'].iloc[-1]
            latest_sma5 = filtered_prices['sma05_ta'].iloc[-1]
            latest_sma20 = filtered_prices['sma20_ta'].iloc[-1]
            latest_sma60 = filtered_prices['sma60_ta'].iloc[-1]
        except (IndexError, KeyError) as e:
            print(f"Error extracting latest values for {ticker}: {str(e)}")
            return

        # 포스트 마켓 가격 먼저 가져오기
        post_market_price = await get_post_market_prices(ticker)
        
        print("=== Chart Rendering Start ===")
        
        # 차트 생성
        indicators = [
            Candlesticks(), 
            SMA(5) | LinePlot(style="dashed", color="#C71585", alpha=0.8, width=3),
            SMA(20) | LinePlot(style="solid", color="red", alpha=0.5, width=3),
            SMA(60) | LinePlot(style="solid", color="blue", alpha=0.5, width=3),
            Volume(),
            RSI(), 
            PPO()
        ]
        
        # 한글이 포함된 문자열을 미리 UTF-8로 인코딩
        name = get_ticker_name(ticker)
        chart_title = (
            f'{ticker} ({name})\n'
            f'Close: {latest_close:,.2f} (last_price: {post_market_price})\n'
            f'MA5: {latest_sma5:,.2f}, MA20: {latest_sma20:,.2f}, MA60: {latest_sma60:,.2f}\n'
            f'RSI: {latest_rsi:.2f}, PPO: {latest_ppo:.2f}'
        ).encode('utf-8').decode('utf-8')
        
        chart = Chart(title=chart_title, max_bars=80)
        chart.plot(prices, indicators)
        fig = chart.figure

        # 이미지 파일로 저장
        image_filename = f'result_mpl_{ticker}.png'
        save_figure(fig, image_filename)

        print("=== Chart Rendering Complete ===")

        # 메시지 작성
        message = (f"Stock: {ticker} ({name})\n"
                   f"Close: {filtered_prices['close'].iloc[-1]:,.2f} (last_price: {post_market_price})\n"
                   f"SMA 5: {filtered_prices['sma05_ta'].iloc[-1]:,.2f}\n"
                   f"SMA 20: {filtered_prices['sma20_ta'].iloc[-1]:,.2f}\n"
                   f"SMA 60: {filtered_prices['sma60_ta'].iloc[-1]:,.2f}\n"
                   f"RSI: {filtered_prices['rsi_ta'].iloc[-1]:,.2f}\n"
                   f"PPO Histogram: {filtered_prices['ppo_histogram'].iloc[-1]:,.2f}\n")

        # Discord로 메시지 전송
        response = requests.post(config.DISCORD_WEBHOOK_URL, data={'content': message})
        if response.status_code != 204:
            print('Discord 메시지 전송 실패')
            print(f"Response: {response.status_code} {response.text}")
        else:
            print('Discord 메시지 전송 성공')

        # Discord로 이미지 전송
        try:
            full_image_path = os.path.join(config.STATIC_IMAGES_PATH, image_filename)
            with open(full_image_path, 'rb') as image_file:
                response = requests.post(config.DISCORD_WEBHOOK_URL, files={'file': image_file})
                if response.status_code in [200, 204]:
                    print(f'Graph 전송 성공: {ticker}')
                else:
                    print(f'Graph 전송 실패: {ticker}')
                    print(f"Response: {response.status_code} {response.text}")

            # Upload the image to GitHub
            await move_files_to_images_folder(full_image_path)
            
        except Exception as e:
            print(f"Error occurred while sending image: {e}")

    except Exception as e:
        print(f"Error in plot_results_mpl: {str(e)}")
        return

if __name__ == "__main__":
    print("Starting test for plotting results.")
    ticker = "GPT-USD"
    start_date = config.START_DATE
    end_date = config.END_DATE
    print(f"Plotting results for {ticker} from {start_date} to {end_date}")

    try:
        # 로컬 파일로부터 prices 데이터 로드
        prices_file_path = os.path.join(config.STATIC_IMAGES_PATH, f'result_VOO_{ticker}.csv')
        
        if os.path.exists(prices_file_path):
            prices = pd.read_csv(prices_file_path, parse_dates=['Date'], index_col='Date')
        else:
            print(f"Data file for {ticker} not found at {prices_file_path}. Please ensure data is generated first.")
            exit()
        
        # plot_results_mpl 호출 시 prices 인자 전달
        asyncio.run(plot_results_mpl(ticker, start_date, end_date, prices))
        print("Plotting completed successfully.")
    except Exception as e:
        print(f"Error occurred while plotting results: {e}")




r"""
cd my-flask-app
python Results_plot_mpl.py

"""
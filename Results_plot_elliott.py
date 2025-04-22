# Results_plot_elliott.py

import matplotlib.dates as mdates  # 기존의 dates 모듈을 mdates로 변경합니다.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import requests
import asyncio
from dotenv import load_dotenv
from io import StringIO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config  # config 모듈을 불러옵니다.
from git_operations import move_files_to_images_folder
from get_ticker import get_ticker_name, is_valid_stock  

import matplotlib.font_manager as fm

# Noto Sans KR 폰트 설정 부분은 그대로 유지
font_path = os.path.join(config.PROJECT_ROOT, 'Noto_Sans_KR', 'static', 'NotoSansKR-Regular.ttf')
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
else:
    print("Font file not found.")

async def plot_comparison_results(ticker, start_date, end_date):
    stock2 = 'VOO'
    fig, ax1 = plt.subplots(figsize=(12, 6))

    full_path= os.path.join(config.STATIC_IMAGES_PATH, f"result_VOO_{ticker}.csv")
    df_graph = pd.read_csv(full_path, parse_dates=['Date'], index_col='Date')
    simplified_df_path = os.path.join(config.STATIC_IMAGES_PATH, f"result_{ticker}.csv")

    try:
        df = pd.read_csv(simplified_df_path, parse_dates=['Date'], index_col='Date')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise

    if start_date is None:
        start_date = df_graph.index.min()
    if end_date is None:
        end_date = df_graph.index.max()

    df_graph = df_graph.loc[start_date:end_date]

    # 7일 및 20일 평균 값 대신 실제 누적 수익률로 변경
    ax1.plot(df_graph.index, df_graph['rate'], label=f'{ticker} Cumulative Return', color='purple')
    ax1.plot(df_graph.index, df_graph['rate_vs'], label='S&P 500 Cumulative Return', color='green')

    # 엘리엇 팩터를 점선 파란색으로 추가
    if 'Elliott_Factor' in df.columns:
        ax2 = ax1.twinx()
        ax2.plot(df.index, df['Elliott_Factor'], label='Elliott Factor', color='blue', linestyle='--', alpha=0.7)
        ax2.set_ylabel('Elliott Factor')
        ax2.legend(loc='upper right')

    plt.ylabel('Total return (%)')
    ax1.legend(loc='upper left')

    plt.title(f"{ticker} ({get_ticker_name(ticker)}) vs {stock2}\n" +
              f"Total Rate: {df_graph['rate'].iloc[-1]:.2f}% (VOO: {df_graph['rate_vs'].iloc[-1]:.2f}%)",
              pad=10)

    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    save_path = os.path.join(config.STATIC_IMAGES_PATH, f'comparison_{ticker}_VOO.png')
    plt.subplots_adjust(top=0.8)
    fig.savefig(save_path)
    plt.cla()
    plt.clf()
    plt.close(fig)

    message = f"Stock: {ticker} ({get_ticker_name(ticker)}) vs VOO\n" \
              f"Total Rate: {df_graph['rate'].iloc[-1]:.2f}% (VOO: {df_graph['rate_vs'].iloc[-1]:.2f}%)"
    response = requests.post(config.DISCORD_WEBHOOK_URL, data={'content': message})

    if response.status_code != 204:
        print('Discord 메시지 전송 실패')
    else:
        print('Discord 메시지 전송 성공')

    try:
        with open(save_path, 'rb') as image:
            response = requests.post(
                config.DISCORD_WEBHOOK_URL,
                files={'file': image}
            )
            if response.status_code in [200, 204]:
                print(f'Graph 전송 성공: {ticker}')
            else:
                print(f'Graph 전송 실패: {ticker}')
                
        await move_files_to_images_folder(save_path)
    except Exception as e:
        print(f"Error occurred while sending image: {e}")


if __name__ == "__main__":
    print("Starting test for plotting results.")
    
    # 기본 테스트 대상 설정
    ticker = "AAPL"
    
    # config에서 시작일과 종료일 가져오기
    start_date = getattr(config, 'START_DATE', None)
    end_date = getattr(config, 'END_DATE', None)
    
    if start_date is None or end_date is None:
        print("Error: START_DATE or END_DATE is not defined in config. Please set these values.")
        sys.exit(1)
    
    # 유효한 티커 심볼인지 확인
    if not is_valid_stock(ticker):
        print(f"Error: '{ticker}' is not a valid stock ticker.")
        sys.exit(1)
    
    print(f"Plotting results for {ticker} from {start_date} to {end_date}")
    
    try:
        # asyncio.run을 통해 비동기 함수 실행
        asyncio.run(plot_comparison_results(ticker, start_date, end_date))
        print(f"Plotting completed successfully for {ticker} from {start_date} to {end_date}")
        
        # 이미지 파일 경로 확인
        save_path = os.path.join(config.STATIC_IMAGES_PATH, f'comparison_{ticker}_VOO.png')
        if os.path.exists(save_path):
            print(f"Graph saved successfully at {save_path}")
        else:
            print(f"Error: Graph image file was not saved at {save_path}")
    
    except FileNotFoundError as fnf_error:
        print(f"FileNotFoundError occurred: {fnf_error}")
        print("Please check if the necessary CSV files are present in the specified path.")
    
    except ValueError as ve:
        print(f"ValueError occurred: {ve}")
        print("Please verify that all input values are correct.")
    
    except Exception as e:
        print(f"An unexpected error occurred while plotting results: {e}")
    
    finally:
        print("Test completed.")


        
# python Results_plot_elliott.py


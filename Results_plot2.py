# Results_plot.py

import matplotlib.dates as dates
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import requests
import asyncio
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from git_operations import move_files_to_images_folder
from get_ticker import get_ticker_name, is_valid_stock
import matplotlib.font_manager as fm

# 폰트 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
font_path = os.path.join(config.PROJECT_ROOT, 'Noto_Sans_KR', 'static', 'NotoSansKR-Regular.ttf')
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = font_prop.get_name()

def save_figure(fig, file_path):
    fig.savefig(file_path)
    plt.close(fig)

async def fetch_csv_from_remote(ticker):
    """
    원격 저장소에서 간략화된 CSV 파일을 다운로드합니다.
    """
    url = f"{config.STATIC_IMAGES_PATH}/result_VOO_{ticker}.csv"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.text), parse_dates=['Date'], index_col='Date')
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch data for {ticker} from remote: {e}")
        return None
    
async def plot_comparison_results(ticker, start_date, end_date, combined_df=None):
    stock2 = 'VOO'
    fig, ax2 = plt.subplots(figsize=(8, 6))
    
    # combined_df가 없는 경우에만 CSV 파일에서 로드
    if combined_df is None:
        csv_path = os.path.join(config.STATIC_IMAGES_PATH, f'result_VOO_{ticker}.csv')
        try:
            df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
            print(f"Loaded data from {csv_path}")
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
            raise
    else:
        df = combined_df
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        print("Using provided combined DataFrame")


    # 날짜 필터링
    if start_date is None:
        start_date = df.index.min()
    if end_date is None:
        end_date = df.index.max()
    
    df = df.loc[start_date:end_date]

    # 컬럼 이름 확인 및 플로팅
    ticker_col = f'rate_{ticker}_5D'
    voo_col = 'rate_VOO_20D'
    
    ax2.plot(df.index, df[ticker_col], label=f'{ticker} Monthly Return')
    ax2.plot(df.index, df[voo_col], label='VOO Monthly Return')

    plt.ylabel('Total Return (%)')
    plt.legend(loc='upper left')
    
    # 오른쪽 y축 추가
    ax_right = ax2.twinx()
    ax_right.set_ylim(ax2.get_ylim())

    # results_relative_divergence.csv 파일에서 추가 데이터 로드
    results_df_path = os.path.join(config.STATIC_IMAGES_PATH, "results_relative_divergence.csv")
    try:
        results_df = pd.read_csv(results_df_path)
        results_row = results_df[results_df['Ticker'] == ticker].iloc[0]
        
        # 필요한 값들 추출
        max_divergence = results_row['Max_Divergence']
        min_divergence = results_row['Min_Divergence']
        current_divergence = results_row['Divergence']
        relative_divergence = results_row['Relative_Divergence']
        expected_return = results_row['Expected_Return']
        recent_signal = results_row['Delta_Previous_Relative_Divergence']
        
        # 최종 수익률 계산
        total_rate = df[ticker_col].iloc[-1] if not df.empty else 0
        voo_rate = df[voo_col].iloc[-1] if not df.empty else 0

        # result_VOO_{ticker}.csv에서 MRI 데이터 로드
        # Hybrid Signal: 마지막 유효한 'cash_' 시그널 찾기
        try:
            if combined_df is not None:
                df_voo = combined_df
                print("Using combined_df for hybrid signal.")
            else:
                voo_csv_path = os.path.join(config.STATIC_IMAGES_PATH, f'result_VOO_{ticker}.csv')
                df_voo = pd.read_csv(voo_csv_path, parse_dates=['Date'])
                print(f"Loaded fallback data from {voo_csv_path} for hybrid signal.")

            # 최근 지표 추출
            mri = df_voo.iloc[-1].get('MRI', 0.0)
            rsi = df_voo.iloc[-1].get('rsi_ta', 0.0)
            ppo = df_voo.iloc[-1].get('ppo_histogram', 0.0)

            # 마지막 유효한 cash_ 시그널
            cash_signal_row = df_voo[df_voo['signal'].astype(str).str.startswith('cash_')].iloc[-1] \
                if not df_voo[df_voo['signal'].astype(str).str.startswith('cash_')].empty else None

            if cash_signal_row is not None:
                raw_signal = cash_signal_row['signal']
                parts = raw_signal.split('_', 2)
                hybrid_signal = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else "N/A"
            else:
                hybrid_signal = "N/A"

        except Exception as e:
            print(f"Error processing hybrid signal from combined_df or CSV: {e}")
            mri = 0.0
            hybrid_signal = "N/A"

        # 그래프 제목 설정
        plt.title(
            f"{ticker} ({get_ticker_name(ticker)}) vs {stock2}\n" +
            f"Total Rate: {total_rate:.2f}% (VOO: {voo_rate:.2f}%), Relative Divergence: {relative_divergence:.2f}%\n" +
            f"Current Divergence: {current_divergence:.2f}% (max: {max_divergence:.2f}, min: {min_divergence:.2f})\n" +
            f"Expected Return: {expected_return:.2f}%, Recent Signal: {recent_signal:.2f}%\n" +
            f"MRI: {mri:.4f}, Hybrid Signal: {hybrid_signal}",
            pad=10
        )
        
    except Exception as e:
        print(f"Error processing results_relative_divergence.csv: {e}")
        plt.title(f"{ticker} vs {stock2}")

    ax2.xaxis.set_major_locator(dates.YearLocator())
    plt.subplots_adjust(top=0.8)

    # 그래프 저장
    save_path = os.path.join(config.STATIC_IMAGES_PATH, f'comparison_{ticker}_VOO.png')
    save_figure(fig, save_path)
    print(f"Graph saved at: {save_path}")

    # Discord로 메시지 및 이미지 전송
    try:
        message = (
            f"Stock: {ticker} ({get_ticker_name(ticker)}) vs VOO\n"
            f"Total Rate: {total_rate:.2f}% (VOO: {voo_rate:.2f}%), Relative Divergence: {relative_divergence:.2f}\n"
            f"Current Divergence: {current_divergence:.2f} (max: {max_divergence:.2f}, min: {min_divergence:.2f})\n"
            f"Expected Return: {expected_return:.2f}%, Recent Signal: {recent_signal:.2f}%\n"
            f"MRI: {mri:.4f}, Hybrid Signal: {hybrid_signal}"
        )
        response = requests.post(config.DISCORD_WEBHOOK_URL, data={'content': message})
        
        if response.status_code == 204:
            print('Discord 메시지 전송 성공')
            
            # 이미지 파일 전송
            with open(save_path, 'rb') as image:
                response = requests.post(config.DISCORD_WEBHOOK_URL, files={'file': image})
                if response.status_code in [200, 204]:
                    print(f'Graph 전송 성공: {ticker}')
                else:
                    print(f'Graph 전송 실패: {ticker}')
        
        # GitHub에 업로드
        await move_files_to_images_folder(save_path)
        print(f"Successfully uploaded {save_path} to GitHub.")
        
    except Exception as e:
        print(f"Error in Discord/GitHub operations: {e}")

if __name__ == "__main__":
    print("Starting test for plotting comparison results.")
    ticker = "BTC-USD"
    start_date = config.START_DATE
    end_date = config.END_DATE
    print(f"Plotting results for {ticker} from {start_date} to {end_date}")

    try:
        # f'result_VOO_{ticker}.csv' 파일이 있는지 확인
        csv_path = os.path.join(config.STATIC_IMAGES_PATH, f'result_VOO_{ticker}.csv')
        if not os.path.exists(csv_path):
            print(f"Data file not found: {csv_path}")
            print("Please ensure the monthly return data is generated first.")
            exit()
        
        # 그래프 생성 함수 호출
        asyncio.run(plot_comparison_results(ticker, start_date, end_date, None))  # df_graph 인자는 None으로 전달
        print("Plotting completed successfully.")
    except Exception as e:
        print(f"Error occurred while plotting results: {e}")

# python Results_plot.py
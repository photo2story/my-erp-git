# backtest_send.py
import requests
import os, sys
import pandas as pd
import matplotlib.pyplot as plt
from discord.ext import commands
import discord
import asyncio
import traceback  # 추가

# 사용자 정의 모듈 임포트
from Results_plot import plot_comparison_results
from Results_plot_mpl import plot_results_mpl
from get_compare_stock_data import save_simplified_csv
from git_operations import move_files_to_images_folder
from Get_data import get_stock_data  # 여기에서 stock 데이터를 가져옴
import My_strategy
from Data_export import export_csv
from get_ticker import is_valid_stock
from get_compare_alpha import process_alpha_beta

# Import configuration
import config

# 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

option_strategy = config.option_strategy  # 시뮬레이션 전략 설정

# VOO 캐시 파일만 삭제하는 함수 (result_VOO_VOO.csv는 유지)
async def clear_voo_cache():
    # VOO 캐시 파일 삭제
    if os.path.exists(config.VOO_CACHE_FILE):
        os.remove(config.VOO_CACHE_FILE)
        print("🧹 VOO 캐시 파일(cached_voo_data.csv) 삭제됨.")
    else:
        print("ℹ️ VOO 캐시 파일(cached_voo_data.csv)이 존재하지 않습니다.")
    
    # MRI 캐시 파일 삭제
    mri_cache_file = os.path.join(config.STATIC_DATA_PATH, 'market_risk_indicator.csv')
    if os.path.exists(mri_cache_file):
        os.remove(mri_cache_file)
        print("🧹 MRI 캐시 파일(market_risk_indicator.csv) 삭제됨.")
    else:
        print("ℹ️ MRI 캐시 파일(market_risk_indicator.csv)이 존재하지 않습니다.")

# VOO 데이터를 가져오거나 캐시된 데이터를 사용하는 함수
async def get_voo_data(option_strategy, first_date, last_date, ctx):
    # 캐시된 VOO 파일이 있는지 확인
    if os.path.exists(config.VOO_CACHE_FILE):
        cached_voo_data = pd.read_csv(config.VOO_CACHE_FILE, parse_dates=['Date'])

        # 캐시된 VOO 데이터의 첫 날짜와 마지막 날짜를 확인
        cached_first_date = cached_voo_data['Date'].min()
        cached_last_date = cached_voo_data['Date'].max()

        # 캐시된 데이터가 유효한지 확인
        if cached_first_date <= pd.to_datetime(first_date) and cached_last_date >= pd.to_datetime(last_date):
            await ctx.send("Using cached VOO data.")
            
            # result_VOO_VOO.csv 파일이 없다면 캐시된 데이터로 생성
            result_file = os.path.join(config.STATIC_IMAGES_PATH, 'result_VOO_VOO.csv')
            if not os.path.exists(result_file):
                os.makedirs(os.path.dirname(result_file), exist_ok=True)
                cached_voo_data.to_csv(result_file, index=False)
                await ctx.send(f"Created result_VOO_VOO.csv from cached data")
            
            # 데이터 병합을 위해 임시로 컬럼명 변경한 복사본 생성
            voo_data_merged = cached_voo_data.copy()
            voo_data_merged = voo_data_merged.rename(columns={
                'rate': 'rate_vs',
                'Close': 'Close_vs',
                'shares': 'shares_vs'
            })
            
            # 필요한 컬럼이 모두 있는지 확인
            required_columns = ['Date', 'rate_vs', 'Close_vs', 'shares_vs']
            missing_columns = [col for col in required_columns if col not in voo_data_merged.columns]
            if missing_columns:
                await ctx.send(f"Error: Missing columns in cached VOO data: {missing_columns}")
                await ctx.send("Fetching new VOO data.")
                os.remove(config.VOO_CACHE_FILE)  # 캐시 파일 삭제
                return await get_voo_data(option_strategy, first_date, last_date, ctx)  # 새로운 데이터 가져오기
                
            return voo_data_merged
        else:
            await ctx.send("Cached VOO data is outdated. Fetching new data.")

    # 새 데이터를 다운로드하는 경우
    await ctx.send(f"Fetching new VOO data from {first_date} to {last_date}")
    voo_data, _, _ = await get_stock_data('VOO', first_date, last_date)
    voo_data_df = My_strategy.my_strategy(voo_data, option_strategy) # VOO 데이터 시뮬레이션
    
    # 필수 컬럼이 있는지 확인
    required_columns = ['Date', 'Close', 'rate', 'shares']
    missing_columns = [col for col in required_columns if col not in voo_data_df.columns]
    if missing_columns:
        await ctx.send(f"Error: Missing required columns in VOO data: {missing_columns}")
        return None
    
    # 데이터 저장 전에 이터 검증
    if voo_data_df.empty:
        await ctx.send("Error: VOO data is empty")
        return None
        
    try:
        # 원본 파일명으로 저장
        result_file = os.path.join(config.STATIC_IMAGES_PATH, 'result_VOO_VOO.csv')
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        voo_data_df.to_csv(result_file, index=False)
        await ctx.send(f"Successfully saved VOO data to {result_file}")
        
        # 캐시에도 원본 데이터 저장
        os.makedirs(os.path.dirname(config.VOO_CACHE_FILE), exist_ok=True)
        voo_data_df.to_csv(config.VOO_CACHE_FILE, index=False)
        await ctx.send("Successfully saved VOO data to cache")
        
        # 저장된 파일 확인
        if not os.path.exists(result_file):
            await ctx.send(f"Warning: Failed to verify saved file at {result_file}")
        else:
            await ctx.send(f"Verified: File exists at {result_file}")
            
        # 병합을 위해 임시로 컬럼명 변경한 복사본 반환
        voo_data_merged = voo_data_df.copy()
        voo_data_merged = voo_data_merged.rename(columns={
            'rate': 'rate_vs',
            'Close': 'Close_vs',
            'shares': 'shares_vs'
        })
        
        return voo_data_merged
        
    except Exception as e:
        await ctx.send(f"Error saving VOO data: {str(e)}")
        return None


# 백테스트를 수행하고 결과를 전송하는 함수
async def backtest_and_send(ctx, ticker, option_strategy, bot=None):
    if bot is None:
        raise ValueError("bot 파라미터는 None일 수 없습니다.")
    
    await ctx.send(f"Backtesting and sending command for: {ticker}")
    
    try:
        await ctx.send(f'get_data for {ticker}.')
        
        # 주식 데이터 가져오기
        stock_data, first_date, last_date = await get_stock_data(ticker, config.START_DATE, config.END_DATE)
        
        if stock_data.empty or first_date is None or last_date is None:
            await ctx.send(f"No stock data found for {ticker}.")
            return

        # 시뮬레이션 실행
        await ctx.send(f'Running strategy for {ticker}.')
        stock_result_df = My_strategy.my_strategy(stock_data, option_strategy)
        
        if stock_result_df.empty:
            await ctx.send(f"No strategy result data for {ticker}.")
            return
        
        # VOO 데이터 가져오기 (캐시된 데이터 사용 또는 새로 가져오기)
        voo_data_df = await get_voo_data(option_strategy, first_date, last_date, ctx)

        await ctx.send(f'Combining data for {ticker} with VOO data.')
        
        # 날짜 형식 통일
        stock_result_df['Date'] = pd.to_datetime(stock_result_df['Date'])
        voo_data_df['Date'] = pd.to_datetime(voo_data_df['Date'])

        # 데이터 범위 확인
        stock_start = stock_result_df['Date'].min()
        stock_end = stock_result_df['Date'].max()
        voo_start = voo_data_df['Date'].min()
        voo_end = voo_data_df['Date'].max()
        
        await ctx.send(f"Data ranges:\n"
                      f"{ticker}: {stock_start.strftime('%Y-%m-%d')} to {stock_end.strftime('%Y-%m-%d')}\n"
                      f"VOO: {voo_start.strftime('%Y-%m-%d')} to {voo_end.strftime('%Y-%m-%d')}")

        # first_date에 맞춰 VOO의 값들을 변환
        reset_date = pd.to_datetime(first_date)
        
        # reset_date가 voo_data_df에 있는지 확인
        if not any(voo_data_df['Date'] == reset_date):
            await ctx.send(f"Error: No VOO data available for the start date {first_date}.")
            return
        
        # rate_vs 초기화
        reset_value = voo_data_df.loc[voo_data_df['Date'] == reset_date, 'rate_vs'].values[0]
        voo_data_df['rate_vs'] = voo_data_df['rate_vs'] - reset_value
        
        # shares_vs 초기화
        reset_value = voo_data_df.loc[voo_data_df['Date'] == reset_date, 'shares_vs'].values[0]
        voo_data_df['shares_vs'] = voo_data_df['shares_vs'] - reset_value
        
        # Close_vs는 초기화하지 않음 (원래 값 유지)

        # stock_result_df와 voo_data_df 병합
        combined_df = pd.merge(
            stock_result_df,
            voo_data_df[['Date', 'rate_vs', 'shares_vs', 'Close_vs']],
            on='Date',
            how='inner'
        )
        
        if combined_df.empty:
            await ctx.send(f"No combined data for {ticker}.")
            return
        
        # 병합 후 결측치 채우기
        combined_df.fillna(0, inplace=True)

        # 유효하지 않은 끝부분 제거: 'Close' 가 0인 행 제거
        combined_df = combined_df[combined_df['Close'] != 0]

        # 중복된 날짜 제거
        combined_df.drop_duplicates(subset='Date', keep='first', inplace=True)
        
        # CSV 파일로 내보내기
        safe_ticker = ticker.replace('/', '-')
        file_path = os.path.join(config.STATIC_IMAGES_PATH, f'result_VOO_{safe_ticker}.csv')
        combined_df.to_csv(file_path, index=False)
        await move_files_to_images_folder(file_path)
        
        # CSV 파일 간소화 및 간소화된 데이터프레임을 반환
        simplified_df = await save_simplified_csv(ticker)
        
        # 알파와 베타 산출
        if not isinstance(combined_df, pd.DataFrame):
            combined_df = pd.DataFrame(combined_df)

        if not isinstance(voo_data_df, pd.DataFrame):
            voo_data_df = pd.DataFrame(voo_data_df)

        # 알파와 베타 계산
        await process_alpha_beta(ticker, combined_df, voo_data_df)
        
        # simplified_df가 비어 있는지 확인
        if combined_df.empty or combined_df[['rate', 'rate_vs']].isnull().all().any():
            await ctx.send(f"{ticker}에 대한 유효한 데이터가 습니다. 데이터를 다시 확인하세요.")
            print(f"Error: {ticker}에 대한 유효한 데이터가 없니다. rate와 rate_vs 데이터를 확인하세요.")
            return

        # 그래프 1: plot_comparison_results
        # await plot_comparison_results(ticker, config.START_DATE, config.END_DATE, combined_df)
        await plot_comparison_results(ticker, config.START_DATE, config.END_DATE, simplified_df, combined_df=combined_df)


        await ctx.send(f'plot_comparison_Results for {ticker} displayed successfully.\n\n')

        # 그래프 2: plot_results_mpl
        await plot_results_mpl(ticker, config.START_DATE, config.END_DATE, combined_df)
        await ctx.send(f'plot_results_mpl for {ticker} displayed successfully.\n\n')

        await ctx.send(f"Backtest and send process completed successfully for {ticker}.")

        # 백테스팅이 완료되었는지 확인하고 캐시 삭제
        await clear_voo_cache()
        print("🧹 VOO 캐시 삭제 완료.")
                    
    except Exception as e:
        error_message = f"An error occurred while processing {ticker}: {e}"
        error_trace = traceback.format_exc()
        await ctx.send(error_message)
        await ctx.send(f"Traceback: {error_trace}")
        print(error_message)
        print(error_trace)

# 메인 실행부 및 테스트는 동일



# 테스트 코드 추가
async def test_backtest_and_send():
    class MockContext:
        async def send(self, message):
            # 중요한 메시지만 출력
            if "Backtesting and sending command for" in message or \
               "Running strategy for" in message or \
               "Combining data for" in message or \
               "Error" in message:  # 에러 메시지는 항상 출력
                print(f"MockContext.send: {message}")

    class MockBot:
        async def change_presence(self, status=None, activity=None):
            pass

    ctx = MockContext()
    bot = MockBot()
    
    try:
        # VOO 데이터 백테스팅 먼저 실행
        await backtest_and_send(ctx, 'VOO', option_strategy='default', bot=bot)
        print("VOO backtesting completed successfully.")
        
        # 나머지 티커 실행
        tickers = [
            'DOGE-USD',  # Rank 1
            # 'ENPH',      # Rank 2
            'BTC-USD',   # Rank 3
            'AAPL',       # Rank 4
            # 'NVDA',      # Rank 5
            'ETH-USD',   # Rank 7
            # 'SOXL',      # Rank 8
            # 'TSLA',      # Rank 9
            # 'SCHD',       # Rank 10
            # '457480.KS'       # Rank 6
        ]
            
        if config.is_cache_valid(config.VOO_CACHE_FILE, config.START_DATE, config.END_DATE):
            print(f"Using cached VOO data for testing.")
        else:
            print(f"VOO cache is not valid or missing. New data will be fetched.")

        for ticker in tickers:
            await backtest_and_send(ctx, ticker, option_strategy='hybrid', bot=bot)
            print(f"Backtesting completed successfully for {ticker}.")
            
        print("\nAll backtesting completed successfully.")
    except Exception as e:
        print(f"Error occurred while backtesting: {e}")
        
    # 모든 백테스팅이 완료되었는지 확인하고 캐시 삭제
    await clear_voo_cache()
    print("🧹 VOO 캐시 삭제 완료.")


# 메인 실행부
if __name__ == "__main__":
    print("Starting test for back-testing.")
    asyncio.run(test_backtest_and_send())


    # python backtest_send.py        

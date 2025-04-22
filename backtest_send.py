# backtest_send.py
import asyncio
import os
import sys
import logging
import traceback
import requests
import pandas as pd
import matplotlib.pyplot as plt
from discord.ext import commands
import discord
from dotenv import load_dotenv

# 사용자 정의 모듈 임포트
from Results_plot import plot_project_status
from Results_plot_mpl import plot_yearly_comparison

# Import configuration
import config

# 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 환경 변수 로드
load_dotenv()

class MockContext:
    async def send(self, message):
        # 중요한 메시지만 출력
        if "Starting analysis" in message or \
           "Completed analysis" in message or \
           "Error" in message:
            logging.info(f"MockContext.send: {message}")

class MockBot:
    async def change_presence(self, status=None, activity=None):
        pass

async def analyze_project(ctx, proj_code):
    """개별 프로젝트 분석을 수행합니다."""
    try:
        # 1. Results_plot.py 실행 - 프로젝트 상태 분석
        logging.info(f"\n=== Starting Project Status Analysis for {proj_code} ===")
        await plot_project_status(proj_code)
        logging.info(f"✅ Project status analysis completed for {proj_code}")

        # 2. Results_plot_mpl.py 실행 - 연도별 비교 분석
        logging.info(f"\n=== Starting Yearly Comparison Analysis for {proj_code} ===")
        await plot_yearly_comparison(proj_code)
        logging.info(f"✅ Yearly comparison analysis completed for {proj_code}")

        return True
    except Exception as e:
        logging.error(f"Error processing project {proj_code}: {e}")
        logging.error(traceback.format_exc())
        return False

async def run_project_analysis():
    """프로젝트 분석을 실행합니다."""
    try:
        ctx = MockContext()
        bot = MockBot()
        
        logging.info("\n=== Starting Project Analysis ===")
        
        success_count = 0
        error_count = 0
        
        for proj_code in test_projects:
            if await analyze_project(ctx, proj_code):
                success_count += 1
                logging.info(f"✅ Successfully analyzed project: {proj_code}")
            else:
                error_count += 1
                logging.error(f"❌ Failed to analyze project: {proj_code}")
        
        logging.info(f"\n=== Analysis Complete ===")
        logging.info(f"Successfully analyzed: {success_count} projects")
        if error_count > 0:
            logging.warning(f"Failed to analyze: {error_count} projects")

    except Exception as e:
        logging.error(f"Error in project analysis: {e}")
        logging.error(traceback.format_exc())

# 테스트 실행
if __name__ == "__main__":
    # 테스트용 프로젝트 코드 목록
    test_projects = [
        'C20240160',  # 몽골 울란바토르 대용량 대중교통 메트로사업 PMC
        'C20170001',  # 미얀마 우정의 교량(Dala) 건설사업 설계 및 감리
        'C20230239',  # pgn
        # 필요한 경우 더 많은 프로젝트 코드 추가
    ]
    
    print("\nStarting project analysis...")
    asyncio.run(run_project_analysis())

    # python backtest_send.py

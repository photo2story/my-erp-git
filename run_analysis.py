import asyncio
import os
import sys
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 사용자 정의 모듈 임포트
from Results_plot import plot_project_status
from Results_plot_mpl import plot_department_breakdown, plot_yearly_comparison
from get_compare_erp_data import test_department_costs

class MockContext:
    async def send(self, message):
        logging.info(f"MockContext.send: {message}")

async def run_analysis():
    try:
        ctx = MockContext()
        
        # 1. ERP 데이터 분석 실행
        logging.info("Starting ERP data analysis...")
        await test_department_costs()
        logging.info("ERP data analysis completed.")

        # 테스트용 프로젝트 코드 목록
        test_projects = [
            'C20240160',  # 예시 프로젝트 코드
            # 필요한 경우 더 많은 프로젝트 코드 추가
        ]

        # 2. 각 프로젝트에 대해 Results_plot.py 실행
        for proj_code in test_projects:
            logging.info(f"\nAnalyzing project {proj_code} with Results_plot...")
            await plot_project_status(proj_code)
            logging.info(f"Results_plot analysis completed for {proj_code}")

            # 3. Results_plot_mpl.py 실행
            logging.info(f"\nAnalyzing project {proj_code} with Results_plot_mpl...")
            await plot_department_breakdown(proj_code)
            await plot_yearly_comparison(proj_code)
            logging.info(f"Results_plot_mpl analysis completed for {proj_code}")

        logging.info("\nAll analyses completed successfully!")

    except Exception as e:
        logging.error(f"Error occurred during analysis: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    print("Starting comprehensive analysis...")
    asyncio.run(run_analysis())

    # python run_analysis.py 
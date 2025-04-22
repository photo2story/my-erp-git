# 파일 이름: /my-flask-app/get_compare_erp_outsourcing.py
# 설명: ERP 데이터를 로드하고 부서별 외주비를 예측하는 프로그램
# 작성자: 박준호
# 작성일: 2025-04-21

import os
import pandas as pd
import asyncio
import logging
from git_operations import move_files_to_github
import numpy as np
import re
import requests
from dotenv import load_dotenv
from datetime import datetime
from typing import Tuple

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 환경 변수 로드
load_dotenv()
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')

# 상수 정의 수정 - 재원조달방식과 국내외구분에 따른 외주율
PROJECT_TYPES = {
    'private': {
        'domestic': {'outsourcing_rate': 0.40},    # 국내 민간 수금대비 외주율 40%
        'overseas': {'outsourcing_rate': 0.40}     # 해외 민간 수금대비 외주율 40%
    },
    'public': {
        'domestic': {'outsourcing_rate': 0.25},    # 국내 공공 수금대비 외주율 25%
        'overseas': {'outsourcing_rate': 0.40}     # 해외 공공 수금대비 외주율 40%
    }
}


def is_public_client(row):
    """재원조달방식을 기준으로 공공/민간을 구분합니다."""
    if not isinstance(row, pd.Series):
        return False
    
    # 재원조달방식 컬럼이 있는 경우
    if '재원조달방식' in row.index and not pd.isna(row['재원조달방식']):
        funding_source = str(row['재원조달방식']).strip()
        return '공공' in funding_source
    
    # 발주처 기준으로 폴백
    if '발주처' in row.index and not pd.isna(row['발주처']):
        client = str(row['발주처']).strip()
        return any(keyword in client for keyword in ['공공', '정부', '공사', '공단']) or client in PUBLIC_CLIENTS
    
    return False

def find_latest_file(folder: str, prefix: str, ext: str = '.csv') -> str:
    """최신 파일을 찾습니다."""
    files = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(ext)]
    def extract_date(fname):
        match = re.search(r'(\d{6,8})', fname)
        return match.group(1) if match else '000000'
    files.sort(key=lambda x: extract_date(x), reverse=True)
    return os.path.join(folder, files[0]) if files else None

def detect_encoding(file_path):
    """파일의 인코딩을 감지합니다."""
    with open(file_path, 'rb') as f:
        raw = f.read(4)
        if raw.startswith(b'\xff\xfe'):
            return 'utf-16le'
        elif raw.startswith(b'\xfe\xff'):
            return 'utf-16be'
        elif raw.startswith(b'\xef\xbb\xbf'):
            return 'utf-8-sig'
        
        f.seek(0)
        content = f.read()
        if all(content[i] == 0 for i in range(1, len(content), 2)):
            return 'utf-16be'
        elif all(content[i] == 0 for i in range(0, len(content), 2)):
            return 'utf-16le'
        
        return 'cp949'

def load_merged_data():
    """병합된 데이터를 로드합니다."""
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'data', 'merged_data.csv')
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        logging.error(f"병합 데이터 로드 중 오류 발생: {e}")
        return None

def get_project_rate(row):
    """프로젝트 유형에 따른 외주율을 반환합니다."""
    project_type = 'public' if is_public_client(row) else 'private'
    location_type = 'overseas' if row.get('국내외구분', '') == '해외' else 'domestic'
    return PROJECT_TYPES[project_type][location_type]['outsourcing_rate']

def calculate_project_metrics(project):
    """프로젝트별 지표를 계산합니다."""
    # 기본값 설정
    total_outsourcing = project.get('전체 외주비', 0)  # 기집행 외주비
    total_collection = project.get('전체 수금', 0)
    current_year_collection_plan = project.get('당년 수금계획', 0)
    total_outsourcing_execution = project.get('전체 외주_실행', 0)  # 외주비 실행예산
    contract_amount = project.get('원화공급가액(천원)', 0)
    
    # 1. 잔여 외주비 계산 (외주비 실행예산 - 기집행 외주비)
    remaining_outsourcing = total_outsourcing_execution - total_outsourcing
    
    # 2. 예상 수금율 계산
    expected_collection_rate = (total_collection + current_year_collection_plan) / contract_amount if contract_amount > 0 else 0
    
    # 3. 지급 예상 외주비 계산 (잔여 외주비 * 예상 수금율)
    expected_outsourcing_payment = remaining_outsourcing * expected_collection_rate
    
    # 4. 당년 예상 외주비 계산 (지급 예상 외주비)
    current_year_expected_outsourcing = expected_outsourcing_payment
    
    # 로깅 추가
    logging.debug(f"""프로젝트 지표 계산:
    - 외주비 실행예산: {format_currency(total_outsourcing_execution)}천원
    - 기집행 외주비: {format_currency(total_outsourcing)}천원
    - 잔여 외주비: {format_currency(remaining_outsourcing)}천원
    - 예상 수금율: {expected_collection_rate:.1%}
    - 당년 지급예상 외주비: {format_currency(current_year_expected_outsourcing)}천원""")
    
    return {
        '잔여외주비': remaining_outsourcing,
        '예상수금율': expected_collection_rate,
        '지급예상외주비': expected_outsourcing_payment,
        '당년예상외주비': current_year_expected_outsourcing,
        '외주실행예산': total_outsourcing_execution,
        '기집행외주비': total_outsourcing
    }

def calculate_existing_project_outsourcing(project_df: pd.DataFrame) -> float:
    """기존 프로젝트의 당년 예상 외주비를 계산합니다."""
    total_outsourcing = 0
    
    for _, project in project_df.iterrows():
        # 기본 데이터 추출
        contract_amount = project['원화공급가액(천원)']
        total_collection = project['전체 수금']
        current_year_plan = project.get('당년 수금계획', 0)
        total_outsourcing_execution = project.get('전체 외주_실행', 0)
        total_outsourcing_cost = project.get('전체 외주비', 0)
        
        # 당년 예상 수금 누계 및 수금율 계산
        expected_collection = total_collection + current_year_plan
        expected_collection_rate = expected_collection / contract_amount if contract_amount > 0 else 0
        
        # Case A: 외주실행예산이 있는 경우
        if pd.notna(total_outsourcing_execution) and total_outsourcing_execution > 0:
            project_outsourcing = total_outsourcing_execution * expected_collection_rate - total_outsourcing_cost
        
        # Case B: 외주실행예산이 없는 경우
        else:
            outsourcing_rate = get_project_rate(project)
            project_outsourcing = expected_collection * outsourcing_rate - total_outsourcing_cost
        
        total_outsourcing += max(0, project_outsourcing)  # 음수 방지
        
        # 로깅 추가
        logging.debug(f"""
        프로젝트: {project['사업코드']} - {project['사업명']}
        계약금액: {contract_amount:,.0f}천원
        전체 수금: {total_collection:,.0f}천원
        당년 수금계획: {current_year_plan:,.0f}천원
        예상 수금율: {expected_collection_rate:.1%}
        예상 외주비: {project_outsourcing:,.0f}천원
        """)
    
    return total_outsourcing

def load_target_amounts():
    """부서별 수주목표 데이터를 로드합니다."""
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'data', 'target_amounts.csv')
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        logging.error(f"수주목표 데이터 로드 중 오류 발생: {e}")
        return None

def get_current_year_contracts(merged_df, current_ym):
    """현재 연도의 부서별 수주액을 계산합니다."""
    current_year = current_ym[:4]
    
    # 진행상태가 '진행' 또는 '중지'이고 계약금액이 있는 프로젝트만 필터링
    active_projects = merged_df[
        (merged_df['진행상태'].isin(['진행', '중지'])) & 
        (merged_df['원화공급가액(천원)'] > 0)
    ]
    logging.info("진행 또는 중지 상태이며 계약금액이 있는 프로젝트만 필터링")
    
    # B로 시작하는 사업코드 제외하고 2025년 사업만 포함
    current_year_projects = active_projects[
        (~active_projects['사업코드'].str.startswith('B', na=False)) &
        (active_projects['사업코드'].str.contains('2025', na=False))
    ]
    
    # 부서별 수주액 집계
    dept_contracts = current_year_projects.groupby('PM부서')['원화공급가액(천원)'].sum().reset_index()
    dept_contracts.columns = ['부서', '현재수주액']
    
    # 로깅 추가
    for _, row in dept_contracts.iterrows():
        logging.info(f"{row['부서']} 2025년 수주액: {format_currency(row['현재수주액'])}천원")
        
        # 해당 부서의 프로젝트 목록 로깅
        dept_projects = current_year_projects[current_year_projects['PM부서'] == row['부서']]
        for _, proj in dept_projects.iterrows():
            # 수금율과 외주기성율 계산
            collection_rate = (proj['전체 수금'] / proj['원화공급가액(천원)'] * 100) if proj['원화공급가액(천원)'] > 0 else 0
            outsourcing_rate = (proj['전체 외주비'] / proj['전체 외주_실행'] * 100) if proj['전체 외주_실행'] > 0 else 0
            
            logging.info(f"""
            [{proj['사업코드']}] {proj['사업명']}
            PM부서: {proj['PM부서']}
            진행상태: {proj['진행상태']}
            계약금액: {format_currency(proj['원화공급가액(천원)'])}천원
            전체 수금: {format_currency(proj['전체 수금'])}천원 (수금율 {collection_rate:.2f}%)
            전체 외주_실행: {format_currency(proj['전체 외주_실행'])}천원
            전체 외주비: {format_currency(proj['전체 외주비'])}천원(외주기성율 {outsourcing_rate:.2f}%)
            당년 수금계획: {format_currency(proj.get('당년 수금계획', 0))}천원
            당년 공정계획: {format_currency(proj.get('당년 공정계획', 0))}천원
            """)
    
    return dept_contracts

def calculate_new_project_outsourcing(current_contracts: float, target_amount: float) -> Tuple[float, float]:
    """신규 수주 및 추가 수주 프로젝트의 당년 예상 외주비를 계산합니다."""
    # 추가 수주 필요액 계산
    additional_target = max(0, target_amount - current_contracts)
    
    # 신규 수주(C2025) 프로젝트 예상 외주비
    # 공공사업 기준(25%) 적용, 예상 수금율 80% 적용
    new_project_outsourcing = current_contracts * 0.25 * 0.8
    
    # 추가 수주 프로젝트 예상 외주비
    additional_outsourcing = additional_target * 0.25 * 0.8
    
    return new_project_outsourcing, additional_outsourcing

def format_currency(value):
    """천 원 단위의 숫자를 포맷팅합니다."""
    return f"{value:,.0f}"

def analyze_outsourcing_prediction(df: pd.DataFrame, department: str) -> dict:
    """부서의 수주 및 외주비 현황을 분석합니다."""
    # 1. 데이터 필터링
    dept_df = df[
        (df['구분'] == department) &
        (df['진행상태'].isin(['진행', '중지'])) &
        (~df['사업코드'].str.startswith('B'))
    ].copy()
    
    # 임시로 PM부서 컬럼 추가
    dept_df['PM부서'] = dept_df['구분']
    
    # 2. 수주목표액 로드
    try:
        target_df = pd.read_csv(os.path.join('..', 'static', 'data', 'target_amounts.csv'))
        target_amount = target_df[target_df['구분'] == department]['수주목표(천원)'].iloc[0]
    except Exception as e:
        logging.error(f"수주목표 데이터 로드 실패: {e}")
        target_amount = 0
    
    # 3. 당년 수주금액 계산 (2025년 코드 프로젝트)
    current_contracts = dept_df[
        dept_df['사업코드'].str.contains('2025')
    ]['원화공급가액(천원)'].sum()
    
    # 4. 기존 프로젝트 예상 외주비
    existing_outsourcing = calculate_existing_project_outsourcing(dept_df)
    
    # 5. 신규 및 추가 수주 프로젝트 예상 외주비
    new_outsourcing, additional_outsourcing = calculate_new_project_outsourcing(
        current_contracts, target_amount
    )
    
    # 6. 총 예상 외주비
    total_outsourcing = existing_outsourcing + new_outsourcing + additional_outsourcing
    
    return {
        '수주목표액': target_amount,
        '당년 수주금액': current_contracts,
        '추가 수주 필요금액': max(0, target_amount - current_contracts),
        '기존 프로젝트 예상 외주비': existing_outsourcing,
        '신규 수주 프로젝트 예상 외주비': new_outsourcing,
        '추가 수주 프로젝트 예상 외주비': additional_outsourcing,
        '당년 외주비 예상 총액': total_outsourcing
    }

def save_prediction_details(df: pd.DataFrame, current_year_pattern: str) -> None:
    """예측 상세 내역을 CSV 파일로 저장합니다."""
    current_ym = datetime.now().strftime('%Y%m')
    
    # 1. 전체 예측 결과
    output_path = os.path.join('..', 'static', 'data', f'outsourcing_prediction_{current_ym}.csv')
    df.groupby('PM부서').agg({
        '원화공급가액(천원)': 'sum',
        '전체 수금': 'sum',
        '당년 수금계획': 'sum',
        '전체 외주_실행': 'sum',
        '전체 외주비': 'sum'
    }).to_csv(output_path, encoding='utf-8-sig')
    
    # 2. 프로젝트별 상세 내역
    details_path = os.path.join('..', 'static', 'data', f'outsourcing_prediction_details_{current_ym}.csv')
    df.to_csv(details_path, encoding='utf-8-sig', index=False)
    
    # GitHub 업로드
    asyncio.create_task(move_files_to_github(output_path))
    asyncio.create_task(move_files_to_github(details_path))

async def send_to_discord(text):
    """Discord로 메시지를 전송합니다."""
    try:
        for i in range(0, len(text), 2000):
            part = text[i:i+2000]
            response = requests.post(DISCORD_WEBHOOK_URL, json={'content': part})
            if response.status_code != 204:
                logging.error(f"Discord 메시지 전송 실패: {response.status_code}")
            await asyncio.sleep(1)
    except Exception as e:
        logging.error(f"Discord 전송 중 오류 발생: {e}")

def fetch_erp_data():
    """ERP 데이터를 로드하고 현재 연월을 반환합니다."""
    try:
        # 2025년 데이터 사용
        current_ym = '202503'
        
        # 병합 데이터 로드
        merged_df = load_merged_data()
        if merged_df is None:
            return None, None
            
        return merged_df, current_ym
        
    except Exception as e:
        logging.error(f"데이터 로드 중 오류 발생: {e}")
        return None, None

async def test_department_outsourcing(department_name='구조부'):
    """특정 부서의 외주비를 예측합니다."""
    df, current_ym = fetch_erp_data()
    if df is None or current_ym is None:
        logging.error("데이터 로드 실패")
        return
    
    logging.info(f"{department_name} 데이터 분석 시작")
    
    # 해당 부서 데이터만 필터링
    dept_df = df[df['PM부서'] == department_name].copy()
    if dept_df.empty:
        logging.error(f"{department_name} 데이터가 없습니다")
        return
    
    # 결과 분석 및 보고서 생성
    message = analyze_outsourcing_prediction(df, department_name)
    
    # Discord로 결과 전송
    await send_to_discord(message)
    
    logging.info(f"{department_name} 분석 완료")

async def analyze_department_summary(department_name):
    """부서의 수주 및 외주비 현황을 분석합니다."""
    try:
        df, current_ym = fetch_erp_data()
        if df is None:
            print("데이터 로드 실패")
            return 0
            
        # 1. 데이터 필터링
        dept_df = df[
            (df['PM부서'] == department_name) &
            (df['진행상태'].isin(['진행', '중지'])) &
            (~df['사업코드'].str.startswith('B'))
        ].copy()
        
        # 2. 수주목표액 로드
        target_df = pd.read_csv(os.path.join('..', 'static', 'data', 'target_amounts.csv'))
        target_amount = target_df[target_df['구분'] == department_name]['수주목표(천원)'].iloc[0]
        
        # 3. 당년 수주금액 계산 (2025년 코드 프로젝트)
        current_year_pattern = '2025'
        current_contracts = dept_df[
            dept_df['사업코드'].str.contains(current_year_pattern)
        ]['원화공급가액(천원)'].sum()
        
        # 4. 기존 프로젝트 예상 외주비
        existing_outsourcing = calculate_existing_project_outsourcing(dept_df)
        
        # 5. 신규 및 추가 수주 프로젝트 예상 외주비
        new_outsourcing, additional_outsourcing = calculate_new_project_outsourcing(
            current_contracts, target_amount
        )
        
        # 6. 총 예상 외주비
        total_outsourcing = existing_outsourcing + new_outsourcing + additional_outsourcing
        
        # 결과 메시지 생성
        message = f"\n=== {department_name} 수주 및 외주비 현황 ===\n"
        message += f"수주목표액: {format_currency(target_amount)} 천원\n"
        message += f"당년 수주금액: {format_currency(current_contracts)} 천원\n"
        message += f"추가 수주 필요금액: {format_currency(max(0, target_amount - current_contracts))} 천원\n"
        message += f"기존 프로젝트 예상 외주비: {format_currency(existing_outsourcing)} 천원\n"
        message += f"신규 수주 프로젝트 예상 외주비: {format_currency(new_outsourcing)} 천원\n"
        message += f"추가 수주 프로젝트 예상 외주비: {format_currency(additional_outsourcing)} 천원\n"
        message += f"당년 외주비 예상 총액: {format_currency(total_outsourcing)} 천원"
        
        # Discord로 결과 전송
        await send_to_discord(message)
        
        # 콘솔에도 출력
        print(message)
        
        return total_outsourcing
        
    except Exception as e:
        error_message = f"에러 발생: {str(e)}"
        print(error_message)
        await send_to_discord(error_message)
        return 0

async def analyze_project_detail(project_code):
    """특정 프로젝트의 상세 분석"""
    try:
        df, _ = fetch_erp_data()
        if df is None:
            print("데이터 로드 실패")
            return
            
        # 해당 프로젝트 데이터 필터링
        project_data = df[df['사업코드'] == project_code]
        
        if project_data.empty:
            print(f"프로젝트 코드 {project_code}에 대한 데이터가 없습니다.")
            return
            
        # 프로젝트 정보 추출
        project = project_data.iloc[0]
        contract_amount = project['원화공급가액(천원)']
        total_collection = project['전체 수금']
        current_year_plan = project.get('당년 수금계획', 0)
        total_outsourcing_execution = project.get('전체 외주_실행', 0)
        total_outsourcing_cost = project.get('전체 외주비', 0)
        
        # 수금율 계산
        collection_rate = (total_collection / contract_amount * 100) if contract_amount > 0 else 0
        outsourcing_rate = (total_outsourcing_cost / total_outsourcing_execution * 100) if pd.notna(total_outsourcing_execution) and total_outsourcing_execution > 0 else 0
        
        # 당년 예상 수금 누계 및 수금율 계산
        expected_collection = total_collection + current_year_plan
        expected_collection_rate = expected_collection / contract_amount if contract_amount > 0 else 0
        
        # 외주비 계산
        if pd.notna(total_outsourcing_execution) and total_outsourcing_execution > 0:
            remaining_outsourcing = total_outsourcing_execution - total_outsourcing_cost
            expected_outsourcing = max(0, total_outsourcing_execution * expected_collection_rate - total_outsourcing_cost)
            actual_outsourcing_rate = (total_outsourcing_execution / contract_amount * 100) if contract_amount > 0 else 0
            display_outsourcing_rate = actual_outsourcing_rate
        else:
            remaining_outsourcing = 0
            outsourcing_rate_applied = get_project_rate(project) * 100
            expected_outsourcing = max(0, expected_collection * (outsourcing_rate_applied/100) - total_outsourcing_cost)
            display_outsourcing_rate = outsourcing_rate_applied
        
        # 결과 메시지 생성
        message = f"\n#### [{project_code}] {project['사업명']}\n"
        message += f"재원조달방식: {project.get('재원조달방식', '정보없음')}\n"
        message += f"진행상태: {project['진행상태']}\n"
        message += f"계약금액: {format_currency(contract_amount)}천원\n"
        message += f"전체 수금: {format_currency(total_collection)}천원 (수금율 {collection_rate:.2f}%)\n"
        message += f"전체 외주실행: {format_currency(total_outsourcing_execution) if pd.notna(total_outsourcing_execution) else 'nan'}천원\n"
        message += f"전체 외주비: {format_currency(total_outsourcing_cost)}천원(외주기성율 {outsourcing_rate:.2f}%)\n"
        message += f"당년 수금계획: {format_currency(current_year_plan)}천원\n"
        message += f"당년 공정계획: {format_currency(project.get('당년 공정계획', 0))}천원\n"
        message += f"잔여 외주비: {format_currency(remaining_outsourcing)}천원\n"
        message += f"지급 예상 외주비: {format_currency(expected_outsourcing)}천원\n"
        message += f"당년 예상 외주비: {format_currency(expected_outsourcing)}천원\n"
        message += f"외주율: {display_outsourcing_rate:.1f}%"
        
        # Discord로 결과 전송
        await send_to_discord(message)
        
        # 콘솔에도 출력
        print(message)
            
    except Exception as e:
        error_message = f"에러 발생: {str(e)}"
        print(error_message)
        await send_to_discord(error_message)

if __name__ == "__main__":
    # 부서 전체 현황과 특정 프로젝트 상세 분석을 순차적으로 실행
    async def main():
        await analyze_department_summary('구조부')
        await analyze_project_detail('C20170001')
    
    asyncio.run(main())

# python get_compare_erp_outsourcing.py
# python get_compare_erp_costs.py

import os
import pandas as pd
import asyncio
import logging
from git_operations import move_files_to_github
import numpy as np
import re
import requests
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 환경 변수 로드
load_dotenv()
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')

# 상수 정의
COST_TYPES = ['인건비', '제경비', '외주비', '판관비', '수금', '수금계획', '공정계획', '총원가']
AGGREGATE_COSTS = {
    '인건비': ['직접인건', '간접인건'],
    '제경비': ['직접제경', '간접제경']
}

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
    # 바이너리 모드로 파일을 읽어서 BOM 확인
    with open(file_path, 'rb') as f:
        raw = f.read(4)
        if raw.startswith(b'\xff\xfe'):
            return 'utf-16le'
        elif raw.startswith(b'\xfe\xff'):
            return 'utf-16be'
        elif raw.startswith(b'\xef\xbb\xbf'):
            return 'utf-8-sig'
        
        # BOM이 없는 경우, 전체 내용을 읽어서 분석
        f.seek(0)
        content = f.read()
        # UTF-16 LE/BE 패턴 확인
        if all(content[i] == 0 for i in range(1, len(content), 2)):
            return 'utf-16be'
        elif all(content[i] == 0 for i in range(0, len(content), 2)):
            return 'utf-16le'
        
        # 기본값으로 cp949 반환
        return 'cp949'

def fetch_erp_data(proj_code=None):
    """ERP 데이터를 로드합니다."""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'data')
    file_path = find_latest_file(data_path, 'erp_data')
    
    if not file_path:
        logging.error("No ERP data files found.")
        return None, None
    
    current_ym = os.path.basename(file_path).replace('erp_data_', '').replace('.csv', '')
    
    try:
        # 파일 인코딩 감지 및 데이터 로드
        encoding = detect_encoding(file_path)
        df = pd.read_csv(file_path, encoding=encoding)
        
        # 필요한 컬럼만 선택
        needed_columns = ['구분']
        for year in range(2020, 2026):
            for cost_type in ['직접인건', '간접인건', '직접제경', '간접제경', '외주비', '판관비', '수금', '수금계획', '공정계획', '총원가']:
                col = f'{year}_{cost_type}'
                if col in df.columns:
                    needed_columns.append(col)
        
        # 필요한 컬럼만 유지
        df = df[needed_columns]
        
        # 구분 컬럼에서 부서명 추출
        df['부서'] = df['구분'].astype(str).apply(lambda x: x.split()[0] if pd.notna(x) and len(x.split()) > 0 else x)
        
        # 숫자 컬럼 변환
        numeric_columns = [col for col in df.columns if col != '구분' and col != '부서']
        for col in numeric_columns:
            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            df[col] = df[col].replace(['nan', 'NaN', 'NULL', '', 'None', 'inf', '-inf'], '0')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # 부서별로 그룹화하여 합계 계산
        df_grouped = df.groupby('부서')[numeric_columns].sum().reset_index()
        df_grouped = df_grouped.rename(columns={'부서': '구분'})
        
        if proj_code:
            df_grouped = df_grouped[df_grouped['사업코드'] == proj_code]
            if df_grouped.empty:
                logging.warning(f"No data for project code: {proj_code}")
                return None, None
        
        return df_grouped, current_ym
    except Exception as e:
        logging.error(f"Error processing data: {e}")
        return None, None

def get_year_columns(start_year, end_year, current_ym):
    """연도별 비용 컬럼을 생성합니다."""
    if not end_year:
        end_year = int(current_ym[:4]) if current_ym else 2024
    if not start_year:
        start_year = 2020
    year_columns = {}
    for year in range(start_year, end_year + 1):
        cols = {}
        for cost_type in AGGREGATE_COSTS:
            for sub_cost in AGGREGATE_COSTS[cost_type]:
                col = f'{year}_{sub_cost}'
                cols[sub_cost] = col
        for cost_type in ['외주비', '판관비', '수금', '수금계획', '공정계획', '총원가']:
            col = f'{year}_{cost_type}'
            cols[cost_type] = col
        year_columns[year] = cols
    return year_columns

async def predict_annual_costs(df, year_columns, current_ym, department_name=None):
    """현재 연도의 비용 예측"""
    current_year = int(current_ym[:4])
    current_month = int(current_ym[4:6])
    
    result_df = pd.DataFrame()
    result_df['구분'] = df['구분']
    
    # 2025년 데이터가 0인지 확인
    if df[f'{current_year}_인건비'].eq(0).all():
        logging.warning(f"No valid {current_year} data. Using {current_year-1} data for prediction.")
        result_df[f'{current_year}_예측_인건비'] = (df[f'{current_year-1}_인건비'] / 12 * 13)
        result_df[f'{current_year}_예측_제경비'] = (df[f'{current_year-1}_제경비'] / 12 * 12)
        result_df[f'{current_year}_예측_판관비'] = (df[f'{current_year-1}_판관비'] / 12 * 12)
    else:
        result_df[f'{current_year}_예측_인건비'] = (df[f'{current_year}_인건비'] / current_month) * 13
        result_df[f'{current_year}_예측_제경비'] = (df[f'{current_year}_제경비'] / current_month) * 12
        result_df[f'{current_year}_예측_판관비'] = (df[f'{current_year}_판관비'] / current_month) * 12
    
    # 예측 외주비는 아웃소싱 모듈에서 계산
    result_df[f'{current_year}_예측_외주비'] = 0  # 초기값 설정
    
    # department_name이 주어진 경우에만 해당 부서의 외주비를 계산
    if department_name:
        try:
            from get_compare_erp_outsourcing import analyze_department_summary
            outsourcing_cost = await analyze_department_summary(department_name)
            if outsourcing_cost is not None:
                # 해당 부서의 행에만 외주비 설정
                mask = result_df['구분'] == department_name
                result_df.loc[mask, f'{current_year}_예측_외주비'] = float(outsourcing_cost)
        except Exception as e:
            logging.error(f"Error getting outsourcing cost for {department_name}: {e}")
    
    result_df[f'{current_year}_예측_총원가'] = (
        result_df[f'{current_year}_예측_인건비'] +
        result_df[f'{current_year}_예측_제경비'] +
        result_df[f'{current_year}_예측_외주비'] +
        result_df[f'{current_year}_예측_판관비']
    )
    
    return result_df

async def calculate_department_costs(df, year_columns, current_ym, department_name=None):
    """부서별 비용 집계."""
    year = int(current_ym[:4])
    
    print(f"\nInitial data shape: {df.shape}")
    
    # 현재 연도의 총원가가 0이 아닌 부서만 필터링
    df_filtered = df[df[f'{year}_총원가'] > 0].copy()
    print(f"\n총원가 > 0 필터링 후 데이터 shape: {df_filtered.shape}")
    
    if df_filtered.empty:
        logging.warning("No departments with non-zero total cost")
        return pd.DataFrame()
    
    # 부서별로 그룹화하여 비용 집계
    unique_departments = df_filtered['구분'].unique()
    result_df = pd.DataFrame(index=unique_departments)
    result_df.index.name = '구분'
    print(f"\n총원가 > 0인 부서 목록: {unique_departments.tolist()}")
    
    # 각 연도별 비용 집계
    for y in range(2020, year + 1):
        # 인건비 (직접 + 간접)
        labor_cols = [f'{y}_직접인건', f'{y}_간접인건']
        labor_sum = df_filtered.groupby('구분')[labor_cols].sum()
        result_df[f'{y}_인건비'] = labor_sum.sum(axis=1)
        
        # 제경비 (직접 + 간접)
        expense_cols = [f'{y}_직접제경', f'{y}_간접제경']
        expense_sum = df_filtered.groupby('구분')[expense_cols].sum()
        result_df[f'{y}_제경비'] = expense_sum.sum(axis=1)
        
        # 기타 비용
        for cost in ['외주비', '판관비', '수금', '수금계획', '공정계획', '총원가']:
            col = f'{y}_{cost}'
            if col in df_filtered.columns:
                result_df[col] = df_filtered.groupby('구분')[col].sum()
    
    # 인덱스를 컬럼으로 변환
    result_df = result_df.reset_index()
    
    print("\n집계 전 데이터 샘플:")
    print(df_filtered[['구분', f'{year}_직접인건', f'{year}_간접인건', f'{year}_직접제경', f'{year}_간접제경']].head())
    
    print("\n집계 후 데이터 샘플:")
    print(result_df[['구분', f'{year}_인건비', f'{year}_제경비']].head())
    
    # 예측 데이터 계산
    predicted_df = await predict_annual_costs(result_df, year_columns, current_ym, department_name)
    result_df = pd.concat([result_df, predicted_df.drop('구분', axis=1)], axis=1)
    
    print(f"\nFinal data shape: {result_df.shape}")
    print("\n최종 데이터 샘플:")
    print(result_df.head())
    
    # 예측 총원가로 정렬
    result_df = result_df.sort_values(by=f'{year}_예측_총원가', ascending=False).reset_index(drop=True)
    
    return result_df

def format_currency(value):
    """천 원 단위의 숫자를 포맷팅합니다."""
    return f"{value:,.0f}"

def analyze_year_over_year_changes(df, current_ym):
    """연도별 원가 변화를 분석합니다."""
    try:
        current_year = int(current_ym[:4])
        prev_year = current_year - 1
        
        # 현재 연도와 전년도 데이터 비교
        analysis_text = f"# {prev_year}년 대비 {current_year}년 원가 예측 분석\n\n"
        
        for _, row in df.iterrows():
            dept = row['구분']
            
            # 전년도 데이터
            prev_labor = row[f'{prev_year}_인건비']
            prev_expense = row[f'{prev_year}_제경비']
            prev_outsource = row[f'{prev_year}_외주비']
            prev_total = row[f'{prev_year}_총원가']
            
            # 올해 예측 데이터
            curr_labor = row[f'{current_year}_예측_인건비']
            curr_expense = row[f'{current_year}_예측_제경비']
            curr_outsource = row[f'{current_year}_예측_외주비']
            curr_total = row[f'{current_year}_예측_총원가']
            
            # 변화율 계산
            def calc_change(curr, prev):
                return ((curr - prev) / prev * 100) if prev != 0 else float('inf')
            
            labor_change = calc_change(curr_labor, prev_labor)
            expense_change = calc_change(curr_expense, prev_expense)
            outsource_change = calc_change(curr_outsource, prev_outsource)
            total_change = calc_change(curr_total, prev_total)
            
            analysis_text += f"## {dept} 원가 분석\n\n"
            
            # 전년도 실적
            analysis_text += f"### {prev_year}년 실적\n"
            analysis_text += f"- 인건비: {format_currency(prev_labor)}천원\n"
            analysis_text += f"- 제경비: {format_currency(prev_expense)}천원\n"
            analysis_text += f"- 외주비: {format_currency(prev_outsource)}천원\n"
            analysis_text += f"- 총원가: {format_currency(prev_total)}천원\n\n"
            
            # 올해 예측
            analysis_text += f"### {current_year}년 예측\n"
            analysis_text += f"- 인건비: {format_currency(curr_labor)}천원 (변화율: {labor_change:.1f}%)\n"
            analysis_text += f"- 제경비: {format_currency(curr_expense)}천원 (변화율: {expense_change:.1f}%)\n"
            analysis_text += f"- 외주비: {format_currency(curr_outsource)}천원 (변화율: {outsource_change:.1f}%)\n"
            analysis_text += f"- 총원가: {format_currency(curr_total)}천원 (변화율: {total_change:.1f}%)\n\n"
            
            # 원가 구성비 분석
            analysis_text += "### 원가 구성비 분석\n"
            total_cost = curr_labor + curr_expense + curr_outsource
            if total_cost > 0:
                analysis_text += f"- 인건비 비중: {(curr_labor/total_cost*100):.1f}%\n"
                analysis_text += f"- 제경비 비중: {(curr_expense/total_cost*100):.1f}%\n"
                analysis_text += f"- 외주비 비중: {(curr_outsource/total_cost*100):.1f}%\n\n"
            
            # 주요 변화 사항 하이라이트
            analysis_text += "### 주요 변화 사항\n"
            highlights = []
            
            # 큰 변화가 있는 항목 강조 (20% 이상 변화)
            if abs(labor_change) >= 20:
                highlights.append(f"인건비가 전년 대비 {labor_change:+.1f}% 변화")
            if abs(expense_change) >= 20:
                highlights.append(f"제경비가 전년 대비 {expense_change:+.1f}% 변화")
            if abs(outsource_change) >= 20:
                highlights.append(f"외주비가 전년 대비 {outsource_change:+.1f}% 변화")
            
            # 원가 구성비 관련 주요 사항
            if curr_labor/total_cost*100 >= 40:
                highlights.append(f"인건비 비중이 높음 ({(curr_labor/total_cost*100):.1f}%)")
            if curr_outsource/total_cost*100 >= 30:
                highlights.append(f"외주비 비중이 높음 ({(curr_outsource/total_cost*100):.1f}%)")
            
            if highlights:
                for highlight in highlights:
                    analysis_text += f"- {highlight}\n"
            else:
                analysis_text += "- 특별한 변화 사항 없음\n"
            
            analysis_text += "\n---\n\n"  # 부서 구분선
        
        return analysis_text
    except Exception as e:
        logging.error(f"연도별 변화 분석 중 오류 발생: {e}")
        return None

async def send_to_discord(text):
    """Discord로 메시지를 전송합니다."""
    try:
        # 메시지를 2000자 단위로 분할하여 전송
        for i in range(0, len(text), 2000):
            part = text[i:i+2000]
            response = requests.post(DISCORD_WEBHOOK_URL, json={'content': part})
            if response.status_code != 204:
                logging.error(f"Discord 메시지 전송 실패: {response.status_code}")
            await asyncio.sleep(1)  # API 제한을 피하기 위한 지연
    except Exception as e:
        logging.error(f"Discord 전송 중 오류 발생: {e}")

async def test_department_costs(department_name=None):
    """부서별 데이터 확인. 특정 부서명이 주어지면 해당 부서만 분석합니다."""
    df, current_ym = fetch_erp_data()
    if df is None or current_ym is None:
        logging.error("Failed to fetch ERP data")
        return
    
    current_year = int(current_ym[:4])
    start_year = current_year - 5  # 현재 연도 기준 이전 5개년
    
    logging.info(f"데이터 분석 기간: {start_year}년 ~ {current_year}년")
    
    year_columns = get_year_columns(start_year, current_year, current_ym)
    result_df = await calculate_department_costs(df, year_columns, current_ym, department_name)
    
    if result_df.empty:
        logging.warning("부서별 원가 데이터가 없습니다")
        return
    
    # 특정 부서 데이터만 필터링
    if department_name:
        result_df = result_df[result_df['구분'] == department_name]
        if result_df.empty:
            logging.warning(f"'{department_name}' 부서의 데이터가 없습니다")
            return
        logging.info(f"'{department_name}' 부서 데이터 분석 시작")
    
    # Git 저장소 내부에 static/images 디렉토리 생성
    output_dir = os.path.join(os.path.dirname(__file__), 'static', 'images')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'results_depart_summary.csv')
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    try:
        await move_files_to_github(output_file)
        logging.info(f"부서별 원가 요약 저장 완료 ({start_year}년 ~ {current_year}년)")
        
        # 연도별 변화 분석 및 Discord 전송
        analysis_text = analyze_year_over_year_changes(result_df, current_ym)
        if analysis_text:
            # 분석 기간 정보 추가
            if department_name:
                header = f"# {department_name} 부서 원가 분석 ({start_year}년 ~ {current_year}년)\n\n"
            else:
                header = f"# {start_year}년 ~ {current_year}년 부서별 원가 분석\n\n"
            await send_to_discord(header + analysis_text)
            logging.info("Discord로 분석 결과 전송 완료")
        
    except Exception as e:
        logging.error(f"GitHub 업로드 또는 Discord 전송 중 오류 발생: {e}")

if __name__ == "__main__":
    # 테스트용 부서명
    test_department = "구조부"
    asyncio.run(test_department_costs(test_department))
# python get_compare_erp_costs.py
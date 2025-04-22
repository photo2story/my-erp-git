# -*- coding: utf-8 -*-

# gemini.py

import os
import sys
import pandas as pd
import requests
from dotenv import load_dotenv
import google.generativeai as genai
import asyncio
from datetime import datetime
import logging
import re

# 로깅 설정을 간단하게
logging.basicConfig(level=logging.WARNING)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from git_operations import move_files_to_github

# 프로젝트 루트 경로 설정
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# static/images 폴더 경로 설정 (프로젝트 루트 기준)
STATIC_IMAGES_PATH = os.path.join(PROJECT_ROOT, 'static', 'images')
STATIC_DATA_PATH = os.path.join(PROJECT_ROOT, 'static', 'data')

# 환경 변수 로드
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')

# Google Generative AI 설정
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def format_currency(value):
    """천 원 단위의 숫자를 포맷팅합니다."""
    return f"{value:,.0f}"

def get_latest_erp_file():
    """가장 최근 날짜의 ERP 데이터 파일을 찾습니다."""
    data_path = STATIC_DATA_PATH
    erp_files = [f for f in os.listdir(data_path) if f.startswith('erp_data_') and f.endswith('.csv')]
    
    if not erp_files:
        raise FileNotFoundError("ERP 데이터 파일을 찾을 수 없습니다.")
    
    # 파일명에서 날짜를 추출하여 가장 최근 파일을 찾음
    latest_file = max(erp_files, key=lambda x: x.replace('erp_data_', '').replace('.csv', ''))
    return os.path.join(data_path, latest_file)

def get_latest_contract_file():
    """가장 최근 날짜의 계약 데이터 파일을 찾습니다."""
    data_path = STATIC_DATA_PATH
    contract_files = [f for f in os.listdir(data_path) if f.startswith('contract_') and f.endswith('.csv')]
    
    if not contract_files:
        raise FileNotFoundError("계약 데이터 파일을 찾을 수 없습니다.")
    
    latest_file = max(contract_files, key=lambda x: x.replace('contract_', '').replace('.csv', ''))
    return os.path.join(data_path, latest_file)

def get_total_contract_amount(contract_data, base_proj_code):
    """기본 사업코드와 변경계약(알파벳 접미사)을 포함한 총 계약금액을 계산합니다."""
    try:
        related_projects = contract_data[contract_data['사업코드'].str.startswith(base_proj_code)]
        
        if related_projects.empty:
            print(f"[ERROR] {base_proj_code}에 대한 계약 데이터가 없습니다.")
            return 0

        total_amount = 0
        for _, row in related_projects.iterrows():
            proj_code = row['사업코드']
            amount = float(str(row['원화공급가액']).replace(',', ''))
            total_amount += amount

        return total_amount/1000  # 천원 단위로 변환
    except Exception as e:
        print(f"[ERROR] 총 계약금액 계산 중 오류 발생: {e}")
        return 0

def get_department_data(df, proj_code):
    """부서별 원가 데이터를 가져옵니다."""
    departments = df[df['사업코드'] == proj_code]['구분'].unique()
    dept_data = []
    
    for dept in departments:
        dept_df = df[(df['사업코드'] == proj_code) & (df['구분'] == dept)]
        
        # 부서별 데이터 계산
        collection = dept_df['전체 수금'].sum()
        labor_cost = dept_df['전체 직접인건'].sum() + dept_df['전체 간접인건'].sum()
        expense_cost = dept_df['전체 직접제경'].sum() + dept_df['전체 간접제경'].sum()
        outsourcing_cost = dept_df['전체 외주비'].sum()
        admin_cost = dept_df['전체 판관비'].sum()
        
        # 설계원가 = 인건비 + 제경비 + 외주비
        design_cost = labor_cost + expense_cost + outsourcing_cost
        
        dept_data.append({
            'department': dept,
            'collection': collection,
            'design_cost': design_cost,
            'labor_cost': labor_cost,
            'expense_cost': expense_cost,
            'outsourcing_cost': outsourcing_cost,
            'admin_cost': admin_cost
        })
    
    return dept_data

def get_yearly_data(df, proj_code):
    """연도별 원가 데이터를 가져옵니다."""
    yearly_data = []
    current_year = datetime.now().year
    
    for year in range(2000, current_year + 1):
        year_str = str(year)
        
        # 해당 연도의 컬럼들
        labor_direct = f'{year_str}_직접인건'
        labor_indirect = f'{year_str}_간접인건'
        expense_direct = f'{year_str}_직접제경'
        expense_indirect = f'{year_str}_간접제경'
        outsourcing = f'{year_str}_외주비'
        
        # 해당 연도의 컬럼이 모두 있는지 확인
        if all(col in df.columns for col in [labor_direct, labor_indirect, expense_direct, expense_indirect, outsourcing]):
            year_df = df[df['사업코드'] == proj_code]
            
            # 연도별 데이터 계산
            labor = year_df[labor_direct].sum() + year_df[labor_indirect].sum()
            expense = year_df[expense_direct].sum() + year_df[expense_indirect].sum()
            outsourcing_cost = year_df[outsourcing].sum()
            
            # 설계원가 계산
            design_cost = labor + expense + outsourcing_cost
            
            if design_cost > 0:
                yearly_data.append({
                    'year': year,
                    'design_cost': design_cost,
                    'labor_cost': labor,
                    'expense_cost': expense,
                    'outsourcing_cost': outsourcing_cost
                })
    
    return yearly_data

def get_current_year_data(df, proj_code):
    """현재 연도의 공정계획, 수금계획, 실제수금 데이터를 가져옵니다."""
    current_year = datetime.now().year
    year_str = str(current_year)
    
    # 해당 연도의 컬럼들
    progress_plan_col = f'{year_str}_공정계획'
    collection_plan_col = f'{year_str}_수금계획'
    collection_col = f'{year_str}_수금'
    
    proj_df = df[df['사업코드'] == proj_code]
    
    current_year_data = {
        'progress_plan': proj_df[progress_plan_col].sum() if progress_plan_col in df.columns else 0,
        'collection_plan': proj_df[collection_plan_col].sum() if collection_plan_col in df.columns else 0,
        'collection': proj_df[collection_col].sum() if collection_col in df.columns else 0
    }
    
    return current_year_data

def get_total_plan_data(df, proj_code):
    """시작부터 현재까지의 총 공정계획과 수금계획을 계산합니다."""
    current_year = datetime.now().year
    proj_df = df[df['사업코드'] == proj_code]
    
    total_progress_plan = 0
    total_collection_plan = 0
    
    # 2000년부터 현재 연도까지의 계획 합계 계산
    for year in range(2000, current_year + 1):
        year_str = str(year)
        progress_plan_col = f'{year_str}_공정계획'
        collection_plan_col = f'{year_str}_수금계획'
        
        if progress_plan_col in df.columns:
            total_progress_plan += proj_df[progress_plan_col].sum()
        if collection_plan_col in df.columns:
            total_collection_plan += proj_df[collection_plan_col].sum()
    
    return {
        'total_progress_plan': total_progress_plan,
        'total_collection_plan': total_collection_plan
    }

def get_prediction_data():
    """예측 데이터를 results_depart_summary.csv에서 읽어옵니다."""
    try:
        prediction_file = os.path.join(os.path.dirname(__file__), '..', 'static', 'images', 'results_depart_summary.csv')
        if not os.path.exists(prediction_file):
            return None
        
        df_pred = pd.read_csv(prediction_file)
        return df_pred
    except Exception as e:
        logging.error(f"예측 데이터 로드 중 오류 발생: {e}")
        return None

async def analyze_with_gemini_erp(proj_code):
    try:
        # 1. 최신 ERP 데이터 로드
        erp_path = get_latest_erp_file()
        print(f"[INFO] ERP 데이터 파일: {erp_path}")
        df_all = pd.read_csv(erp_path)
        
        # 2. 최신 계약 데이터 로드
        contract_path = get_latest_contract_file()
        print(f"[INFO] 계약 데이터 파일: {contract_path}")
        contract_data = pd.read_csv(contract_path)
        
        # 3. 기본 사업코드에서 알파벳 제거 (변경계약 코드를 기본 코드로 변환)
        base_proj_code = re.sub(r'[A-Z]+$', '', proj_code)
        
        # 4. 총 계약금액 계산 (변경계약 포함)
        contract_amount = get_total_contract_amount(contract_data, base_proj_code)
        if contract_amount == 0:
            raise ValueError(f"사업코드 {base_proj_code}에 대한 계약 데이터가 없습니다.")

        # 5. 특정 사업코드 필터링
        df = df_all[df_all['사업코드'].str.startswith(base_proj_code)].copy()
        if df.empty:
            raise ValueError(f"사업코드 {base_proj_code}에 대한 데이터가 없습니다.")

        # 6. 프로젝트 기본 정보
        project_name = df["사업명"].iloc[0] if not df.empty else ""
        execution_budget = df['전체 합계_실행예산'].sum()
        collection = df['전체 수금'].sum()
        
        # 7. 원가 정보 계산
        labor_cost = df['전체 직접인건'].sum() + df['전체 간접인건'].sum()
        expense_cost = df['전체 직접제경'].sum() + df['전체 간접제경'].sum()
        outsourcing_cost = df['전체 외주비'].sum()
        admin_cost = df['전체 판관비'].sum()
        
        # 설계원가 = 인건비 + 제경비 + 외주비
        design_cost = labor_cost + expense_cost + outsourcing_cost
        execution_rate = (design_cost/collection*100) if collection > 0 else 0
        
        # 8. 부서별 데이터 가져오기
        departments_data = get_department_data(df, base_proj_code)
        
        # 9. 연도별 데이터 가져오기
        yearly_data = get_yearly_data(df, base_proj_code)

        # 현재 연도 데이터 가져오기
        current_year_data = get_current_year_data(df, base_proj_code)
        current_year_progress_plan = current_year_data['progress_plan']
        current_year_collection_plan = current_year_data['collection_plan']
        current_year_collection = current_year_data['collection']

        # 누적 계획 데이터 가져오기
        total_plan_data = get_total_plan_data(df, base_proj_code)
        total_progress_plan = total_plan_data['total_progress_plan']
        total_collection_plan = total_plan_data['total_collection_plan']

        # 예측 데이터 로드 및 분석 추가
        df_pred = get_prediction_data()
        if df_pred is not None:
            prompt_erp = f"""
        주요 지표를 먼저 보여주고 상세 분석을 제공하는 보고서를 작성해주세요.

        1) 사업 개요:
           사업코드: {base_proj_code}
           사업명: {project_name}
           계약금액: {format_currency(contract_amount)}천원
           실행예산: {format_currency(execution_budget)}천원 ({(execution_budget/contract_amount*100):.2f}%)
           수금: {format_currency(collection)}천원 (수금율: {(collection/contract_amount*100):.2f}%)

        2) 누적 공정 현황:
           총 공정계획(시작~현재): {format_currency(total_progress_plan)}천원
           총 수금계획(시작~현재): {format_currency(total_collection_plan)}천원
           총 실제수금(시작~현재): {format_currency(collection)}천원
           - 누적공정계획 대비 수금율: {(collection/total_progress_plan*100 if total_progress_plan > 0 else 0):.1f}%
           - 누적수금계획 대비 수금율: {(collection/total_collection_plan*100 if total_collection_plan > 0 else 0):.1f}%

        3) 금년도 계획 대비 실적:
           금년 공정계획: {format_currency(current_year_progress_plan)}천원
           금년 수금계획: {format_currency(current_year_collection_plan)}천원
           금년 실제수금: {format_currency(current_year_collection)}천원
           - 공정계획 대비 수금율: {(current_year_collection/current_year_progress_plan*100 if current_year_progress_plan > 0 else 0):.1f}%
           - 수금계획 대비 수금율: {(current_year_collection/current_year_collection_plan*100 if current_year_collection_plan > 0 else 0):.1f}%
        4) 원가 분석:
           설계원가: {format_currency(design_cost)}천원 (실행율: {execution_rate:.2f}%)
           - 인건비: {format_currency(labor_cost)}천원 ({(labor_cost/design_cost*100):.1f}%)
           - 제경비: {format_currency(expense_cost)}천원 ({(expense_cost/design_cost*100):.1f}%)
           - 외주비: {format_currency(outsourcing_cost)}천원 ({(outsourcing_cost/design_cost*100):.1f}%)

           다음 사항들을 중점적으로 분석해주세요:(간단하게 팩트만 보고서 작성)
           1. 계약 금액대비  실행예산이 90%를 초과하는지, 초과시 리스크관리
           1. 수금 대비 설계원가의 실행율이 80%를 초과하는지, 초과시 리스크관리
           2. 누적 공정 대비 수금율이 시작부터 현재까지 수금율이 50% 미만인지 리스크관리
        5) 부서별 분석:
        """
        
        # 부서별 데이터 추가
        for dept in departments_data:
            dept_execution_rate = (dept['design_cost']/dept['collection']*100) if dept['collection'] > 0 else 0
            prompt_erp += f"""
           {dept['department']}:
           - 수금: {format_currency(dept['collection'])}천원
           - 설계원가: {format_currency(dept['design_cost'])}천원 (실행율: {dept_execution_rate:.2f}%)
           - 인건비: {format_currency(dept['labor_cost'])}천원 ({(dept['labor_cost']/dept['design_cost']*100 if dept['design_cost'] > 0 else 0):.1f}%)
           - 제경비: {format_currency(dept['expense_cost'])}천원 ({(dept['expense_cost']/dept['design_cost']*100 if dept['design_cost'] > 0 else 0):.1f}%)
           - 외주비: {format_currency(dept['outsourcing_cost'])}천원 ({(dept['outsourcing_cost']/dept['design_cost']*100 if dept['design_cost'] > 0 else 0):.1f}%)
            """

        prompt_erp += """
           1. 실행율이 80%를 초과하는 부서만 숫자로 나열하고 실행초과에 대해서만 간략히 언급
              (타부서 0% 등 언급하지 않음, 그 부서와 연관이 별로 없어서 협업을 안하는 거임)
              (- 교통부 122%)
              (- 환경부 115%)
        """

        prompt_erp += "\n        6) 연도별 진행 분석:\n"
        
        # 연도별 데이터 추가
        for year_data in yearly_data:
            prompt_erp += f"""
           {year_data['year']}년:
           - 설계원가: {format_currency(year_data['design_cost'])}천원
           - 인건비: {format_currency(year_data['labor_cost'])}천원 ({(year_data['labor_cost']/year_data['design_cost']*100):.1f}%)
           - 제경비: {format_currency(year_data['expense_cost'])}천원 ({(year_data['expense_cost']/year_data['design_cost']*100):.1f}%)
           - 외주비: {format_currency(year_data['outsourcing_cost'])}천원 ({(year_data['outsourcing_cost']/year_data['design_cost']*100):.1f}%)
            """

        prompt_erp += """
           1. 연도별 변동성이 급격한 경우만 업급 혹은 감소 언급 
        """

        # 예측 데이터 로드 및 분석 추가
        df_pred = get_prediction_data()
        if df_pred is not None:
            prompt_erp += "\n        7) 2025년 원가 예측 분석:\n"
            
            total_pred_labor = df_pred['2025_예측_인건비'].sum()
            total_pred_expense = df_pred['2025_예측_제경비'].sum()
            total_pred_outsourcing = df_pred['2025_예측_외주비'].sum()
            total_pred_admin = df_pred['2025_예측_판관비'].sum()
            total_pred_cost = df_pred['2025_예측_총원가'].sum()
            
            prompt_erp += f"""
           전체 예측:
           - 예측 총원가: {format_currency(total_pred_cost)}천원
           - 예측 인건비: {format_currency(total_pred_labor)}천원 ({(total_pred_labor/total_pred_cost*100):.1f}%)
           - 예측 제경비: {format_currency(total_pred_expense)}천원 ({(total_pred_expense/total_pred_cost*100):.1f}%)
           - 예측 외주비: {format_currency(total_pred_outsourcing)}천원 ({(total_pred_outsourcing/total_pred_cost*100):.1f}%)
           - 예측 판관비: {format_currency(total_pred_admin)}천원 ({(total_pred_admin/total_pred_cost*100):.1f}%)

           부서별 예측:"""
            
            # 예측 총원가 기준으로 정렬하여 상위 5개 부서만 표시
            top_departments = df_pred.nlargest(5, '2025_예측_총원가')
            for _, dept in top_departments.iterrows():
                dept_pred_cost = dept['2025_예측_총원가']
                if dept_pred_cost > 0:
                    prompt_erp += f"""
           {dept['구분']}:
           - 예측 총원가: {format_currency(dept['2025_예측_총원가'])}천원
           - 예측 인건비: {format_currency(dept['2025_예측_인건비'])}천원 ({(dept['2025_예측_인건비']/dept_pred_cost*100):.1f}%)
           - 예측 제경비: {format_currency(dept['2025_예측_제경비'])}천원 ({(dept['2025_예측_제경비']/dept_pred_cost*100):.1f}%)
           - 예측 외주비: {format_currency(dept['2025_예측_외주비'])}천원 ({(dept['2025_예측_외주비']/dept_pred_cost*100):.1f}%)
           - 예측 판관비: {format_currency(dept['2025_예측_판관비'])}천원 ({(dept['2025_예측_판관비']/dept_pred_cost*100):.1f}%)"""

            prompt_erp += """
           
           분석 포인트:
           1. 2024년 대비 2025년 예측 총원가의 증감률이 20% 이상인 부서
           2. 인건비 비중이 40% 이상으로 예측되는 부서
           3. 외주비 비중이 30% 이상으로 예측되는 부서
           """

        # Gemini API를 사용하여 분석 텍스트 생성
        response = model.generate_content(prompt_erp)
        
        # 분석 결과를 날짜와 함께 전체 report_text로 구성
        report_text = f"{datetime.now().strftime('%Y-%m-%d')} - 사업 분석 보고서\n" + response.text

        # Discord로 메시지 전송 (2000자 단위로 분할)
        for i in range(0, len(report_text), 2000):
            part = report_text[i:i+2000]
            requests.post(DISCORD_WEBHOOK_URL, json={'content': part})
            
        # 사업코드에서 연도 추출 (예: C20170001 -> 2017)
        year = "20" + base_proj_code[1:3]
        
        # 연도별 폴더 경로 생성
        destination_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static', 'images', year))
        
        # 연도별 폴더가 없으면 생성
        os.makedirs(destination_dir, exist_ok=True)
        
        # 파일로 전체 report_text 저장
        report_file = f'report_{base_proj_code}.txt'
        report_file_path = os.path.join(destination_dir, report_file)

        # 파일 저장
        with open(report_file_path, 'w', encoding='utf-8') as file:
            file.write(report_text)

        # 파일을 images 폴더로 이동
        await move_files_to_github(report_file_path)

        return f'사업코드 {base_proj_code}에 대한 분석이 완료되어 Discord로 전송되었으며, {year}년 폴더에 텍스트 파일로 저장되었습니다.'

    except Exception as e:
        error_message = f"사업코드 {proj_code} 분석 중 오류 발생: {str(e)}"
        print(error_message)
        requests.post(DISCORD_WEBHOOK_URL, data={'content': f"```\n{error_message}\n```"})
        raise

# 메인 코드 시작 전에 추가
if __name__ == '__main__':
    print("\n[INFO] Starting ERP data analysis...")
    
    # 테스트할 사업코드 목록
    proj_codes = ['C20170001', 'C20110138']
    
    for proj_code in proj_codes:
        print(f"\n[INFO] Analyzing project {proj_code}...")
        try:
            # ERP 데이터 분석 실행
            result = asyncio.run(analyze_with_gemini_erp(proj_code))
            print(result)
            print(f"[SUCCESS] Analysis completed for {proj_code}")
            
        except Exception as e:
            print(f"[ERROR] Failed to analyze {proj_code}: {e}")
            continue
    
    print("\n[INFO] All analyses completed.")

# python gemini.py
 
 
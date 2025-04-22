# Results_plot.py

import matplotlib
matplotlib.use('Agg')  # GUI 없이 이미지 생성을 위한 백엔드 설정
import pandas as pd
import matplotlib.pyplot as plt
import os
import requests
import config
import matplotlib.font_manager as fm
from dotenv import load_dotenv
import asyncio
from git_operations import move_files_to_github
import numpy as np
from datetime import datetime
import re

# 환경 변수 로드
load_dotenv()

# 폰트 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
font_path = os.path.join(config.PROJECT_ROOT, 'Noto_Sans_KR', 'static', 'NotoSansKR-Regular.ttf')
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = font_prop.get_name()

def save_figure(fig, file_path):
    """그래프를 파일로 저장하고 메모리를 정리합니다."""
    try:
        fig.savefig(file_path)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"[ERROR] 그래프 저장 중 오류 발생: {e}")
        return False

def format_currency(value):
    """백만 원 단위의 숫자를 포맷팅합니다."""
    return f"{value/1000:,.1f}"  # 천원 단위를 백만원 단위로 변환하고 소수점 1자리까지 표시

def get_latest_erp_file():
    """가장 최근 날짜의 ERP 데이터 파일을 찾습니다."""
    data_path = config.STATIC_DATA_PATH
    erp_files = [f for f in os.listdir(data_path) if f.startswith('erp_data_') and f.endswith('.csv')]
    
    if not erp_files:
        raise FileNotFoundError("ERP 데이터 파일을 찾을 수 없습니다.")
    
    # 파일명에서 날짜를 추출하여 가장 최근 파일을 찾음
    latest_file = max(erp_files, key=lambda x: x.replace('erp_data_', '').replace('.csv', ''))
    return os.path.join(data_path, latest_file)

def get_latest_contract_file():
    """가장 최근 날짜의 계약 데이터 파일을 찾습니다."""
    data_path = config.STATIC_DATA_PATH
    contract_files = [f for f in os.listdir(data_path) if f.startswith('contract_') and f.endswith('.csv')]
    
    if not contract_files:
        raise FileNotFoundError("계약 데이터 파일을 찾을 수 없습니다.")
    
    # 파일명에서 날짜를 추출하여 가장 최근 파일을 찾음
    latest_file = max(contract_files, key=lambda x: x.replace('contract_', '').replace('.csv', ''))
    return os.path.join(data_path, latest_file)

def get_project_start_year(df, proj_code):
    """사업의 실제 시작 연도를 찾습니다 (총원가가 0이 아닌 첫 해)."""
    try:
        # 해당 사업코드의 데이터만 필터링
        proj_data = df[df['사업코드'] == proj_code]
        if proj_data.empty:
            return 2000  # 기본값

        # 2000년부터 현재까지의 연도별 총원가 컬럼 확인
        current_year = datetime.now().year
        for year in range(2000, current_year + 1):
            col_name = f'{year}_총원가'
            if col_name in proj_data.columns and proj_data[col_name].sum() > 0:
                print(f"[INFO] 사업 시작 연도 탐지: {year} (컬럼: {col_name}, 총원가: {proj_data[col_name].sum():,.0f})")
                return year
        
        print(f"[WARNING] {proj_code}: 총원가가 있는 연도를 찾지 못했습니다. 2000년부터 시작합니다.")
        return 2000  # 데이터가 없는 경우 기본값
    except Exception as e:
        print(f"[WARNING] 시작 연도 탐지 중 오류 발생: {e}")
        return 2000

def get_total_contract_amount(contract_data, base_proj_code):
    """기본 사업코드와 변경계약(알파벳 접미사)을 포함한 총 계약금액을 계산합니다."""
    try:
        # 기본 사업코드로 시작하는 모든 프로젝트 코드 찾기
        related_projects = contract_data[contract_data['사업코드'].str.startswith(base_proj_code)]
        
        if related_projects.empty:
            print(f"[ERROR] {base_proj_code}에 대한 계약 데이터가 없습니다.")
            return 0

        # 각 프로젝트의 원화공급가액 합산
        total_amount = 0
        for _, row in related_projects.iterrows():
            proj_code = row['사업코드']
            amount = float(str(row['원화공급가액']).replace(',', ''))
            total_amount += amount
            # 변경계약 내역 출력
            if proj_code != base_proj_code:
                print(f"[INFO] 변경계약 {proj_code}: {format_currency(amount/1000)}천원")

        print(f"[INFO] {base_proj_code} 총 계약금액: {format_currency(total_amount/1000)}천원 (변경계약 {len(related_projects)-1}건 포함)")
        return total_amount/1000  # 천원 단위로 변환

    except Exception as e:
        print(f"[ERROR] 총 계약금액 계산 중 오류 발생: {e}")
        return 0

def get_last_active_year(df, base_proj_code):
    """마지막으로 데이터가 있는 연도를 찾습니다."""
    try:
        # 해당 사업코드의 데이터만 필터링
        proj_data = df[df['사업코드'].str.startswith(base_proj_code)]
        if proj_data.empty:
            return datetime.now().year

        current_year = datetime.now().year
        last_active_year = 2000

        # 2000년부터 현재까지 연도별로 검사
        for year in range(2000, current_year + 1):
            # 해당 연도의 수금, 원가 데이터 확인
            labor_direct_col = f'{year}_직접인건'
            labor_indirect_col = f'{year}_간접인건'
            expense_direct_col = f'{year}_직접제경'
            expense_indirect_col = f'{year}_간접제경'
            outsourcing_col = f'{year}_외주비'

            # 해당 연도의 컬럼이 존재하고 데이터가 있는지 확인
            if all(col in proj_data.columns for col in [labor_direct_col, labor_indirect_col, 
                                                      expense_direct_col, expense_indirect_col, 
                                                      outsourcing_col]):
                # 원가 데이터 합산
                year_total = (
                    proj_data[labor_direct_col].sum() +
                    proj_data[labor_indirect_col].sum() +
                    proj_data[expense_direct_col].sum() +
                    proj_data[expense_indirect_col].sum() +
                    proj_data[outsourcing_col].sum()
                )
                
                if year_total > 0:
                    last_active_year = year
                    print(f"[DEBUG] {year}년 원가 데이터 발견: {format_currency(year_total)}천원")

        print(f"[INFO] 마지막 데이터 발생 연도: {last_active_year}")
        return last_active_year

    except Exception as e:
        print(f"[WARNING] 마지막 활성 연도 탐지 중 오류 발생: {e}")
        return datetime.now().year

async def plot_project_status(proj_code):
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

        # 5. 특정 사업코드 필터링 (기본 코드로 시작하는 모든 데이터)
        df = df_all[df_all['사업코드'].str.startswith(base_proj_code)].copy()
        if df.empty:
            raise ValueError(f"사업코드 {base_proj_code} 에 대한 데이터가 없습니다.")

        # 6. 연도 범위 설정 (실제 시작 연도부터 마지막 데이터가 있는 연도까지)
        start_year = get_project_start_year(df_all, base_proj_code)
        end_year = get_last_active_year(df_all, base_proj_code)
        print(f"[INFO] 그래프 표시 기간: {start_year}년 ~ {end_year}년")
        years = list(range(start_year, end_year + 1))
        
        # 7. 부서별, 연도별 원가 데이터 추출
        departments = df['구분'].unique()
        cost_types = ['인건비', '제경비', '외주비']
        
        # 누적 데이터를 저장할 딕셔너리
        cumulative_data = {dept: {cost: [] for cost in cost_types} for dept in departments}
        
        # 연도별 데이터 수집 및 누적
        for dept in departments:
            dept_data = df[df['구분'] == dept]
            
            # 각 원가 유형별로 연도별 데이터 수집
            for cost_type in cost_types:
                yearly_values = []
                cumulative = 0
                
                for year in years:  # end_year까지만 반복
                    year_str = str(year)
                    if cost_type == '인건비':
                        value = (
                            dept_data[f'{year_str}_직접인건'].sum() +
                            dept_data[f'{year_str}_간접인건'].sum()
                        )
                    elif cost_type == '제경비':
                        value = (
                            dept_data[f'{year_str}_직접제경'].sum() +
                            dept_data[f'{year_str}_간접제경'].sum()
                        )
                    else:  # 외주비
                        value = dept_data[f'{year_str}_외주비'].sum()
                    
                    cumulative += value
                    yearly_values.append(cumulative)
                
                cumulative_data[dept][cost_type] = yearly_values

        # 8. 그래프 생성
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # 부서별로 다른 색상 사용
        colors = plt.cm.Set3(np.linspace(0, 1, len(departments)))
        
        # 부서별 누적 데이터를 저장할 딕셔너리
        department_handles = {}  # 부서별 범례 핸들 저장용
        
        # 누적 영역 그래프 생성
        for dept_idx, dept in enumerate(departments):
            bottom = np.zeros(len(years))
            dept_short = dept.replace('부', '')  # '부'만 제거하고 전체 이름 유지
            
            for cost_type in cost_types:
                values = cumulative_data[dept][cost_type]
                handle = ax.fill_between(years, bottom, bottom + values, 
                              alpha=0.7,
                              label='_nolegend_',  # 범례에 표시하지 않음
                              color=colors[dept_idx])
                bottom += values
                
                # 각 부서의 마지막 cost_type일 때 범례 핸들 저장
                if cost_type == cost_types[-1]:
                    department_handles[dept_short] = handle

        # 범례 설정 - 부서만 표시
        handles = list(department_handles.values())
        labels = list(department_handles.keys())
        plt.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)

        # 주요 금액 가로선 추가
        # 계약금액 선
        plt.axhline(y=contract_amount, color='red', linestyle='--', alpha=0.7)
        plt.text(years[0], contract_amount, f'계약금액: {format_currency(contract_amount)}백만원', 
                verticalalignment='bottom', color='red', fontsize=12)

        # 수금액 선
        plt.axhline(y=df["전체 수금"].sum(), color='blue', linestyle='--', alpha=0.7)
        plt.text(years[0], df["전체 수금"].sum(), f"수금: {format_currency(df['전체 수금'].sum())}백만원", 
                verticalalignment='bottom', color='blue', fontsize=12)

        # 실행예산 선
        plt.axhline(y=df["전체 합계_실행예산"].sum(), color='green', linestyle='--', alpha=0.7)
        plt.text(years[0], df["전체 합계_실행예산"].sum(), f"실행예산: {format_currency(df['전체 합계_실행예산'].sum())}백만원", 
                verticalalignment='bottom', color='green', fontsize=12)

        # 그래프 제목에 사업 정보 추가
        project_name = df["사업명"].iloc[0] if not df.empty else ""
        title = f"[{base_proj_code}] {project_name}\n"
        title += f"계약금액: {format_currency(contract_amount)}백만원, 수금: {format_currency(df['전체 수금'].sum())}백만원\n"
        title += '부서별 원가 누적 추이'
        plt.title(title, fontsize=14, pad=20)
        
        plt.xlabel('연도', fontsize=12)
        plt.ylabel('누적 금액 (백만 원)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # x축 연도 표시 설정
        plt.xticks(years, rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        
        plt.tight_layout()

        # 9. 저장 - 연도별 폴더 생성 및 저장
        year = base_proj_code[1:5]  # 사업코드에서 연도 추출 (예: C20170001 -> 2017)
        year_folder = os.path.join(config.STATIC_IMAGES_PATH, year)
        os.makedirs(year_folder, exist_ok=True)  # 연도 폴더가 없으면 생성
        
        save_path = os.path.join(year_folder, f'comparison_{base_proj_code}.png')
        if not save_figure(fig, save_path):
            return

        print(f"[INFO] 그래프 저장 완료: {save_path}")

        # 10. 디스코드 전송
        webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        if webhook_url:  # webhook URL이 있을 때만 Discord로 전송
            # 최종 누적 금액 계산
            final_totals = {dept: {
                cost: cumulative_data[dept][cost][-1]
                for cost in cost_types
            } for dept in departments}
            
            # 전체 총괄 계산
            total_by_cost = {cost: sum(final_totals[dept][cost] for dept in departments) 
                            for cost in cost_types}
            grand_total = sum(total_by_cost.values())
            
            msg = f"[{base_proj_code} 부서별 원가 누적 분석]\n"
            
            # 계약 정보 가져오기 (변경계약 포함)
            contract_amount = get_total_contract_amount(contract_data, base_proj_code)
            if contract_amount == 0:
                raise ValueError(f"사업코드 {base_proj_code}에 대한 계약 데이터가 없습니다.")

            # ERP 데이터에서 실행예산과 수금 정보 가져오기
            execution_budget = df['전체 합계_실행예산'].sum()
            collection = df['전체 수금'].sum()
            collection_plan = df['전체 수금계획'].sum()
            
            # 원가 정보 계산
            labor_cost = df['전체 직접인건'].sum() + df['전체 간접인건'].sum()
            expense_cost = df['전체 직접제경'].sum() + df['전체 간접제경'].sum()
            outsourcing_cost = df['전체 외주비'].sum()
            admin_cost = df['전체 판관비'].sum()
            
            # 설계원가 = 인건비 + 제경비 + 외주비
            design_cost = labor_cost + expense_cost + outsourcing_cost
            # 총원가 = 설계원가 + 판관비
            total_cost = design_cost + admin_cost

            # 전체 총괄 정보를 먼저 표시
            msg += f"\n[{base_proj_code} 총괄]\n"
            msg += f"사업명: {project_name}\n\n"
            msg += f"계약금액: {format_currency(contract_amount)}백만원\n"
            msg += f"실행예산: {format_currency(execution_budget)}백만원 ({(execution_budget/contract_amount*100):.2f}%, 실행예산/계약금액)\n"
            msg += f"수금: {format_currency(collection)}백만원 (수금율 {(collection/contract_amount*100):.2f}%, 수금/계약금액)\n"
            execution_rate = (design_cost/collection*100) if collection > 0 else 0
            msg += f"설계원가: {format_currency(design_cost)}백만원 (실행율 {execution_rate:.2f}%, 설계원가/수금)\n"
            msg += f"인건비: {format_currency(labor_cost)}백만원 ({(labor_cost/design_cost*100):.1f}%)\n"
            msg += f"제경비: {format_currency(expense_cost)}백만원 ({(expense_cost/design_cost*100):.1f}%)\n"
            msg += f"외주비: {format_currency(outsourcing_cost)}백만원 ({(outsourcing_cost/design_cost*100):.1f}%)\n"

            # 부서별 상세 정보
            for dept in departments:
                dept_data = df[df['구분'] == dept]
                dept_collection = dept_data['전체 수금'].sum()
                dept_labor = dept_data['전체 직접인건'].sum() + dept_data['전체 간접인건'].sum()
                dept_expense = dept_data['전체 직접제경'].sum() + dept_data['전체 간접제경'].sum()
                dept_outsourcing = dept_data['전체 외주비'].sum()
                dept_admin = dept_data['전체 판관비'].sum()
                
                # 부서별 설계원가 = 인건비 + 제경비 + 외주비
                dept_design_cost = dept_labor + dept_expense + dept_outsourcing
                # 부서별 총원가 = 설계원가 + 판관비
                dept_total_cost = dept_design_cost + dept_admin
                
                msg += f"\n{dept} 누적 원가 구성:\n"
                msg += f"수금: {format_currency(dept_collection)}백만원\n"
                dept_execution_rate = (dept_design_cost/dept_collection*100) if dept_collection > 0 else 0
                msg += f"설계원가: {format_currency(dept_design_cost)}백만원(실행율 {dept_execution_rate:.2f}%, 설계원가/수금)\n"
                msg += f"인건비: {format_currency(dept_labor)}백만원 ({(dept_labor/dept_design_cost*100 if dept_design_cost > 0 else 0):.1f}%)\n"
                msg += f"제경비: {format_currency(dept_expense)}백만원 ({(dept_expense/dept_design_cost*100 if dept_design_cost > 0 else 0):.1f}%)\n"
                msg += f"외주비: {format_currency(dept_outsourcing)}백만원 ({(dept_outsourcing/dept_design_cost*100 if dept_design_cost > 0 else 0):.1f}%)\n"

            response = requests.post(webhook_url, data={"content": msg})
            if response.status_code not in [200, 204]:
                print(f"[ERROR] Discord 메시지 전송 실패: {response.status_code}")
                return

            with open(save_path, 'rb') as f:
                response = requests.post(webhook_url, files={"file": f})
                if response.status_code not in [200, 204]:
                    print(f"[ERROR] Discord 이미지 전송 실패: {response.status_code}")
                    return
            
            print(f"[SUCCESS] Discord로 {base_proj_code} 그래프 전송 완료!")
        else:
            print("[INFO] Discord webhook URL이 설정되지 않아 Discord 전송을 건너뜁니다.")

        # 11. GitHub 저장소로 파일 이동
        await move_files_to_github(save_path)
        print(f"[SUCCESS] GitHub 저장소로 {os.path.basename(save_path)} 업로드 완료!")

    except FileNotFoundError as e:
        print(f"[ERROR] 파일 오류: {e}")
    except ValueError as e:
        print(f"[ERROR] 데이터 오류: {e}")
    except Exception as e:
        print(f"[ERROR] 예상치 못한 오류 발생: {e}")

def print_department_yearly_costs(df, proj_code, department):
    """특정 부서의 연도별 원가를 출력합니다."""
    try:
        # 해당 사업코드와 부서의 데이터만 필터링
        dept_data = df[(df['사업코드'] == proj_code) & (df['구분'] == department)]
        if dept_data.empty:
            print(f"[ERROR] {proj_code}의 {department} 데이터가 없습니다.")
            return

        print(f"\n{proj_code} {department}의 연도별 원가 상세:")
        print("연도\t설계원가\t\t인건비(직접+간접)\t제경비(직접+간접)\t외주비")
        print("-" * 100)
        
        # 2000년부터 현재까지 검색
        total = {'설계원가': 0, '인건비': 0, '제경비': 0, '외주비': 0}
        current_year = datetime.now().year
        for year in range(2000, current_year + 1):
            # 각 비용 항목 컬럼
            labor_direct_col = f'{year}_직접인건'
            labor_indirect_col = f'{year}_간접인건'
            expense_direct_col = f'{year}_직접제경'
            expense_indirect_col = f'{year}_간접제경'
            outsourcing_col = f'{year}_외주비'
            
            # 해당 연도의 데이터가 있는 경우만 처리
            if all(col in dept_data.columns for col in [labor_direct_col, labor_indirect_col, 
                                                      expense_direct_col, expense_indirect_col, outsourcing_col]):
                labor = dept_data[labor_direct_col].sum() + dept_data[labor_indirect_col].sum()
                expense = dept_data[expense_direct_col].sum() + dept_data[expense_indirect_col].sum()
                outsourcing = dept_data[outsourcing_col].sum()
                design_cost = labor + expense + outsourcing
                
                if design_cost > 0:
                    print(f"{year}\t{format_currency(design_cost)}\t{format_currency(labor)}\t\t{format_currency(expense)}\t\t{format_currency(outsourcing)}")
                    total['설계원가'] += design_cost
                    total['인건비'] += labor
                    total['제경비'] += expense
                    total['외주비'] += outsourcing

        print("-" * 100)
        print(f"총합:\t{format_currency(total['설계원가'])}\t{format_currency(total['인건비'])}\t\t{format_currency(total['제경비'])}\t\t{format_currency(total['외주비'])}")
        
        # 실제 설계원가 (비용 합계) 계산
        actual_design_cost = total['인건비'] + total['제경비'] + total['외주비']
        print(f"\n[비교] 연도별 설계원가 합계: {format_currency(total['설계원가'])}백만원")
        print(f"[비교] 실제 설계원가 (인건비+제경비+외주비): {format_currency(actual_design_cost)}백만원")
        if total['설계원가'] != actual_design_cost:
            print(f"[비교] 차이: {format_currency(total['설계원가'] - actual_design_cost)}백만원")
        
    except Exception as e:
        print(f"[ERROR] 연도별 원가 출력 중 오류 발생: {e}")

if __name__ == "__main__":
    proj_code = "C20170001"
    print("\n[INFO] 테스트 실행 시작")
    print(f"[INFO] 대상 사업코드: {proj_code}")
    
    try:
        # ERP 데이터 로드
        erp_path = get_latest_erp_file()
        df_all = pd.read_csv(erp_path)
        
        # 건설사업관리부 연도별 총원가 출력
        print_department_yearly_costs(df_all, proj_code, "건설사업관리부")
        
        # 그래프 생성 실행
        print("\n[INFO] 그래프 생성 시작...")
        asyncio.run(plot_project_status(proj_code))
        print(f"[SUCCESS] 테스트 성공! 프로젝트 {proj_code}의 원가 누적 분석 이미지가 생성되었습니다.")
        print(f"[INFO] 이미지 경로: {os.path.join(config.STATIC_IMAGES_PATH, f'comparison_{proj_code}.png')}")
        
    except Exception as e:
        print(f"[ERROR] 테스트 실패: {e}")
        print(f"[DEBUG] 오류 상세: {str(e)}")

# python Results_plot.py
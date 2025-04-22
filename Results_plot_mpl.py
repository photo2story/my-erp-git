## Results_plot_mpl.py

# Results_plot_mpl.py

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import requests
import config
import matplotlib.font_manager as fm
from dotenv import load_dotenv
import numpy as np
from datetime import datetime
import asyncio
from git_operations import move_files_to_github

# .env 로드
load_dotenv()

# 폰트 설정 (Noto Sans)
font_path = os.path.join(config.PROJECT_ROOT, 'Noto_Sans_KR', 'static', 'NotoSansKR-Regular.ttf')
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = font_prop.get_name()

def format_currency(value):
    """백만 원 단위의 숫자를 포맷팅합니다."""
    return f"{value/1000:,.1f}"  # 천원 단위를 백만원 단위로 변환하고 소수점 1자리까지 표시

def save_figure(fig, save_path):
    """그래프를 파일로 저장하고 메모리를 정리합니다."""
    try:
        fig.savefig(save_path)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"[ERROR] 그래프 저장 실패: {e}")
        return False

def get_column_value(df, column):
    """데이터프레임에서 안전하게 값을 가져옵니다."""
    try:
        if column in df.columns:
            value = df[column].sum()  # 부서별 합계
            return 0 if pd.isna(value) else value
        return 0
    except Exception as e:
        print(f"[WARNING] {column} 값 추출 중 오류 발생: {e}")
        return 0

def get_latest_contract_file():
    """가장 최근 날짜의 contract 파일을 찾습니다."""
    data_path = config.STATIC_DATA_PATH
    contract_files = [f for f in os.listdir(data_path) if f.startswith('contract_') and f.endswith('.csv')]
    
    if not contract_files:
        raise FileNotFoundError("계약 데이터 파일을 찾을 수 없습니다.")
    
    # 파일명에서 날짜를 추출하여 가장 최근 파일을 찾음
    latest_file = max(contract_files, key=lambda x: x.replace('contract_', '').replace('.csv', ''))
    return os.path.join(data_path, latest_file)

async def plot_department_breakdown(proj_code):
    try:
        # 1. 원본 ERP 데이터 로드
        erp_path = os.path.join(config.STATIC_DATA_PATH, 'erp_data_202503.csv')
        df_all = pd.read_csv(erp_path)
        df = df_all[df_all['사업코드'] == proj_code].copy()

        if df.empty:
            raise ValueError(f"[ERROR] 프로젝트 코드 {proj_code} 에 대한 데이터가 없습니다.")

        # 2. 부서별 항목 집계
        grouped = df.groupby('구분').agg({
            '전체 수금': 'sum',
            '전체 직접인건': 'sum',
            '전체 간접인건': 'sum',
            '전체 직접제경': 'sum',
            '전체 간접제경': 'sum',
            '전체 외주비': 'sum',
            '전체 총원가': 'sum',
            '전체 수금_비용손익': 'sum'
        }).fillna(0).reset_index()

        # 인건비와 제경비 합산
        grouped['전체 인건비'] = grouped['전체 직접인건'] + grouped['전체 간접인건']
        grouped['전체 제경비'] = grouped['전체 직접제경'] + grouped['전체 간접제경']

        # 그래프에 표시할 컬럼 선택
        plot_cols = ['전체 수금', '전체 인건비', '전체 제경비',
                    '전체 외주비', '전체 총원가', '전체 수금_비용손익']

        # 3. 그래프 생성
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_data = grouped.set_index('구분')[plot_cols]
        
        # 막대 그래프 생성
        plot_data.plot(kind='bar', ax=ax)
        plt.title(f'[{proj_code}] 부서별 주요 항목 비교')
        plt.ylabel('금액 (천 원)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # 연도별 폴더 생성 및 저장
        year = proj_code[1:5]  # 사업코드에서 연도 추출 (예: C20240160 -> 2024)
        year_folder = os.path.join(config.STATIC_IMAGES_PATH, year)
        os.makedirs(year_folder, exist_ok=True)  # 연도 폴더가 없으면 생성
        
        save_path = os.path.join(year_folder, f'department_comparison_{proj_code}.png')
        if not save_figure(fig, save_path):
            return

        print(f"[INFO] 그래프 저장 완료: {save_path}")

        # 4. Discord 메시지 전송
        webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        if webhook_url:  # webhook URL이 있을 때만 Discord로 전송
            # 전체 합계 계산
            totals = grouped[plot_cols].sum()
            
            msg = (
                f"[알림] [{proj_code}] 부서별 원가 분석 요약\n"
                f"- 전체 수금: {format_currency(totals['전체 수금'])}천원\n"
                f"- 전체 인건비: {format_currency(totals['전체 인건비'])}천원\n"
                f"- 전체 제경비: {format_currency(totals['전체 제경비'])}천원\n"
                f"- 전체 외주비: {format_currency(totals['전체 외주비'])}천원\n"
                f"- 총원가: {format_currency(totals['전체 총원가'])}천원\n"
                f"- 손익: {format_currency(totals['전체 수금_비용손익'])}천원"
            )
            
            response = requests.post(webhook_url, data={"content": msg})
            if response.status_code not in [200, 204]:
                print(f"[ERROR] Discord 텍스트 메시지 실패: {response.status_code}")

            with open(save_path, 'rb') as f:
                response = requests.post(webhook_url, files={"file": f})
                if response.status_code not in [200, 204]:
                    print(f"[ERROR] Discord 이미지 전송 실패: {response.status_code}")
                else:
                    print(f"[SUCCESS] Discord로 부서 그래프 전송 완료! [{proj_code}]")
        else:
            print("[INFO] Discord webhook URL이 설정되지 않아 Discord 전송을 건너뜁니다.")

        # GitHub 저장소로 파일 이동
        await move_files_to_github(save_path)
        print(f"[SUCCESS] GitHub 저장소로 부서별 비교 그래프 업로드 완료! [{proj_code}]")

    except FileNotFoundError as e:
        print(f"[ERROR] 파일 없음: {e}")
    except ValueError as e:
        print(f"[ERROR] 데이터 오류: {e}")
    except Exception as e:
        print(f"[ERROR] 처리 중 예외 발생: {e}")

async def plot_yearly_comparison(proj_code):
    try:
        # 1. 원본 ERP 데이터 로드
        erp_path = os.path.join(config.STATIC_DATA_PATH, 'erp_data_202503.csv')
        df_all = pd.read_csv(erp_path)
        df = df_all[df_all['사업코드'] == proj_code].copy()

        if df.empty:
            raise ValueError(f"[ERROR] 프로젝트 코드 {proj_code} 에 대한 데이터가 없습니다.")

        # 계약 데이터 로드 (최신 파일 사용)
        try:
            contract_path = get_latest_contract_file()
            print(f"[INFO] 계약 데이터 파일: {contract_path}")
            df_contract = pd.read_csv(contract_path)
            
            # 계약금액 컬럼 찾기
            possible_columns = ['원화공급가액', '원화공급금액', '원화공급가액(천원)', '계약금액', '계약금액(천원)', '공급가액']
            contract_column = None
            for col in possible_columns:
                if col in df_contract.columns:
                    contract_column = col
                    break
            
            if contract_column is None:
                print("[WARNING] 계약금액 컬럼을 찾을 수 없습니다.")
                print(f"[DEBUG] 사용 가능한 컬럼: {', '.join(df_contract.columns)}")
                contract_amount = 0
            else:
                contract_row = df_contract[df_contract['사업코드'] == proj_code]
                if contract_row.empty:
                    print(f"[WARNING] 프로젝트 코드 {proj_code}의 계약 데이터가 없습니다.")
                    contract_amount = 0
                else:
                    contract_amount = contract_row[contract_column].iloc[0]
                    # 문자열인 경우 숫자로 변환 (쉼표 제거 후 변환)
                    if isinstance(contract_amount, str):
                        contract_amount = float(contract_amount.replace(',', ''))
                    # 원 단위를 천원 단위로 변환
                    contract_amount = contract_amount / 1000
                    print(f"[INFO] 계약금액 컬럼 '{contract_column}' 사용")
        except Exception as e:
            print(f"[WARNING] 계약 데이터 로드 중 오류 발생: {e}")
            contract_amount = 0

        # 2. 데이터 집계
        # 전체 실행예산 가져오기
        total_budget = get_column_value(df, '전체 합계_실행예산')
        
        # 수금계획 데이터 가져오기
        prev_year_revenue_plan = get_column_value(df, '전체 수금계획') - get_column_value(df, '2025_수금계획')
        current_year_revenue_plan = get_column_value(df, '2025_수금계획')
        total_revenue_plan = get_column_value(df, '전체 수금계획')

        # 전체 데이터
        total_data = {
            '공정계획': get_column_value(df, '전체 공정계획'),
            '수금': get_column_value(df, '전체 수금'),
            '총원가': get_column_value(df, '전체 총원가'),
            '손익': get_column_value(df, '전체 수금_비용손익')
        }

        # 금년 누계 (2025년)
        current_year_data = {
            '공정계획': get_column_value(df, '2025_공정계획'),
            '수금': get_column_value(df, '2025_수금'),
            '총원가': get_column_value(df, '2025_총원가'),
            '손익': get_column_value(df, '2025_수금_비용손익')
        }

        # 전년도까지의 누계 (전체 - 금년)
        prev_years_data = {
            '공정계획': total_data['공정계획'] - current_year_data['공정계획'],
            '수금': total_data['수금'] - current_year_data['수금'],
            '총원가': total_data['총원가'] - current_year_data['총원가'],
            '손익': total_data['손익'] - current_year_data['손익']
        }

        # 3. 그래프 생성
        fig, ax = plt.subplots(figsize=(16, 8))  # 가로 크기를 더 늘림
        
        # y축 범위 설정을 위해 최대값 계산
        max_value = max([
            max(prev_years_data.values()),
            max(current_year_data.values()),
            max(total_data.values()),
            contract_amount,
            total_budget,
            total_revenue_plan
        ])
        # y축 범위 설정 (최대값의 130%로 설정하여 위쪽 여백 확보)
        plt.ylim(min(-2000000, min(prev_years_data['손익'], current_year_data['손익'], total_data['손익'])), max_value * 1.3)

        # 막대 그래프 생성
        bar_width = 0.15  # 막대 폭을 더 좁게 조정
        index = np.arange(3)  # 전년 누계, 금년 누계, 총계

        # 각 항목별 막대 그래프 (순서: 공정계획, 수금, 총원가, 손익)
        bars1 = ax.bar(index - bar_width*2, [prev_years_data['공정계획'], current_year_data['공정계획'], total_data['공정계획']], 
                      bar_width, label='공정계획', color='#9C27B0')
        bars2 = ax.bar(index - bar_width*0.5, [prev_years_data['수금'], current_year_data['수금'], total_data['수금']], 
                      bar_width, label='수금', color='#2196F3')
        bars3 = ax.bar(index + bar_width*0.5, [prev_years_data['총원가'], current_year_data['총원가'], total_data['총원가']], 
                      bar_width, label='총원가', color='#4CAF50')
        bars4 = ax.bar(index + bar_width*2, [prev_years_data['손익'], current_year_data['손익'], total_data['손익']], 
                      bar_width, label='손익', color='#FFC107')

        # 그래프 꾸미기
        plt.title(f'[{proj_code}] {df["사업명"].iloc[0] if not df.empty else "사업명 없음"}', pad=20, fontsize=14)
        plt.ylabel('금액 (백만 원)', fontsize=12)
        plt.xticks(index, ['전년 누계', '금년 누계', '총계'], fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)

        # 막대 위에 값 표시 - 위치 조정
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                # 값이 0인 경우 표시하지 않음
                if height == 0:
                    continue
                # 음수인 경우 아래에 표시, 양수인 경우 위에 표시
                va = 'bottom' if height >= 0 else 'top'
                y_offset = 200000 if height >= 0 else -200000  # 값 표시 위치 조정 (더 멀리)
                ax.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                       format_currency(height),
                       ha='center', va=va, rotation=0, fontsize=12)  # 폰트 크기 증가

        autolabel(bars1)
        autolabel(bars2)
        autolabel(bars3)
        autolabel(bars4)

        # 계약금액 가로선 추가 (총계 위치의 공정계획 막대 위치에만)
        progress_bar_center = index[2] - bar_width*2  # 총계 위치의 공정계획 막대 중앙
        line_x_start = progress_bar_center - bar_width  # 공정계획 막대의 시작점
        line_x_end = progress_bar_center + bar_width    # 공정계획 막대의 끝점
        plt.hlines(y=contract_amount, xmin=line_x_start, xmax=line_x_end, 
                  colors='purple', linestyles='dashed', linewidth=2)
        
        # 계약금액 값과 텍스트 표시 (공정계획 막대 위에)
        plt.text(progress_bar_center, contract_amount * 1.08, 
                f'계약금액\n{format_currency(contract_amount)}',
                ha='center', va='bottom', color='purple', fontsize=12)

        # 전체 실행예산 가로선 추가 (총계 위치의 총원가 막대 위치에만)
        cost_bar_center = index[2] + bar_width*0.5  # 총계 위치의 총원가 막대 중앙
        line_x_start = cost_bar_center - bar_width  # 총원가 막대의 시작점
        line_x_end = cost_bar_center + bar_width    # 총원가 막대의 끝점
        plt.hlines(y=total_budget, xmin=line_x_start, xmax=line_x_end, 
                  colors='red', linestyles='dashed', linewidth=2)
        
        # 실행예산 값과 텍스트 표시 (총원가 막대 위에)
        plt.text(cost_bar_center, total_budget * 1.08, 
                f'실행예산\n{format_currency(total_budget)}',
                ha='center', va='bottom', color='red', fontsize=12)

        # 수금계획 가로선 추가 (각 구간의 수금 막대 위치에)
        revenue_positions = [
            (index[0] - bar_width*0.5, prev_year_revenue_plan, '전년'),  # 전년 누계
            (index[1] - bar_width*0.5, current_year_revenue_plan, '금년'),  # 금년 누계
            (index[2] - bar_width*0.5, total_revenue_plan, '총계')  # 총계
        ]
        
        for x_center, plan_value, period in revenue_positions:
            # 수금 막대 위치에 맞춰 선 그리기
            line_x_start = x_center - bar_width
            line_x_end = x_center + bar_width
            plt.hlines(y=plan_value, xmin=line_x_start, xmax=line_x_end,
                      colors='blue', linestyles='dashed', linewidth=2)
            
            # 수금계획 값과 텍스트 표시
            plt.text(x_center, plan_value * 1.08,
                    f'수금계획\n{format_currency(plan_value)}',
                    ha='center', va='bottom', color='blue', fontsize=12)

        # 여백 조정
        plt.subplots_adjust(right=0.85, top=0.88)  # 범례와 제목을 위한 여백 확보
        plt.tight_layout()

        # 5. 저장 - 연도별 폴더 생성 및 저장
        year = proj_code[1:5]  # 사업코드에서 연도 추출 (예: C20240160 -> 2024)
        year_folder = os.path.join(config.STATIC_IMAGES_PATH, year)
        os.makedirs(year_folder, exist_ok=True)  # 연도 폴더가 없으면 생성
        
        save_path = os.path.join(year_folder, f'yearly_comparison_{proj_code}.png')
        if not save_figure(fig, save_path):
            return

        print(f"[INFO] 그래프 저장 완료: {save_path}")

        # 6. Discord 메시지 전송
        webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        if webhook_url:  # webhook URL이 있을 때만 Discord로 전송
            # 사업명 가져오기
            project_name = df['사업명'].iloc[0] if not df.empty else "사업명 없음"
            
            # 원가율, 수금율 계산 함수
            def calculate_rate(value, total):
                if total == 0:
                    return 0
                return (value / total) * 100

            # 수금율 계산 (수금/계약금액)
            total_revenue_rate = calculate_rate(total_data['수금'], contract_amount)
            current_year_revenue_rate = calculate_rate(current_year_data['수금'], contract_amount)
            
            # 원가율 계산 (원가/계약금액)
            total_cost_rate = calculate_rate(total_data['총원가'], contract_amount)
            current_year_cost_rate = calculate_rate(current_year_data['총원가'], contract_amount)
            
            msg = (
                f"[알림] 사업 실적 분석\n"
                f"- 사업코드: {proj_code}\n"
                f"- 사업명: {project_name}\n"
                f"- 계약금액: {format_currency(contract_amount)}천원\n"
                f"- 실행예산: {format_currency(total_budget)}천원\n"
                f"\n"
                f"[전년도까지 누계]\n"
                f"- 공정계획: {format_currency(prev_years_data['공정계획'])}천원\n"
                f"- 수금계획: {format_currency(prev_year_revenue_plan)}천원\n"
                f"- 수금: {format_currency(prev_years_data['수금'])}천원\n"
                f"- 총원가: {format_currency(prev_years_data['총원가'])}천원\n"
                f"- 손익: {format_currency(prev_years_data['손익'])}천원\n\n"
                f"[금년 누계]\n"
                f"- 공정계획: {format_currency(current_year_data['공정계획'])}천원\n"
                f"- 수금계획: {format_currency(current_year_revenue_plan)}천원\n"
                f"- 수금: {format_currency(current_year_data['수금'])}천원(수금율 {current_year_revenue_rate:.2f}%)\n"
                f"- 총원가: {format_currency(current_year_data['총원가'])}천원(원가율 {current_year_cost_rate:.2f}%)\n"
                f"- 손익: {format_currency(current_year_data['손익'])}천원\n\n"
                f"[총계]\n"
                f"- 공정계획: {format_currency(total_data['공정계획'])}천원\n"
                f"- 수금계획: {format_currency(total_revenue_plan)}천원\n"
                f"- 수금: {format_currency(total_data['수금'])}천원(수금율 {total_revenue_rate:.2f}%)\n"
                f"- 총원가: {format_currency(total_data['총원가'])}천원(원가율 {total_cost_rate:.2f}%)\n"
                f"- 손익: {format_currency(total_data['손익'])}천원"
            )
            
            response = requests.post(webhook_url, data={"content": msg})
            if response.status_code not in [200, 204]:
                print(f"[ERROR] Discord 텍스트 메시지 실패: {response.status_code}")

            with open(save_path, 'rb') as f:
                response = requests.post(webhook_url, files={"file": f})
                if response.status_code not in [200, 204]:
                    print(f"[ERROR] Discord 이미지 전송 실패: {response.status_code}")
                else:
                    print(f"[SUCCESS] Discord로 연도별 비교 그래프 전송 완료! [{proj_code}]")
        else:
            print("[INFO] Discord webhook URL이 설정되지 않아 Discord 전송을 건너뜁니다.")

        # GitHub 저장소로 파일 이동
        await move_files_to_github(save_path)
        print(f"[SUCCESS] GitHub 저장소로 연도별 비교 그래프 업로드 완료! [{proj_code}]")

    except FileNotFoundError as e:
        print(f"[ERROR] 파일 없음: {e}")
    except ValueError as e:
        print(f"[ERROR] 데이터 오류: {e}")
    except Exception as e:
        print(f"[ERROR] 처리 중 예외 발생: {e}")

if __name__ == "__main__":
    proj_code = "C20230239"
    print("\n[INFO] 테스트 실행 시작")
    print(f"[INFO] ERP 데이터 파일: {os.path.join(config.STATIC_DATA_PATH, 'erp_data_202503.csv')}")
    print(f"[INFO] 대상 사업코드: {proj_code}")
    
    try:
        # 데이터 로드 및 검증
        erp_path = os.path.join(config.STATIC_DATA_PATH, 'erp_data_202503.csv')
        if not os.path.exists(erp_path):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {erp_path}")
            
        df_all = pd.read_csv(erp_path)
        df = df_all[df_all['사업코드'] == proj_code].copy()
        
        if df.empty:
            raise ValueError(f"사업코드 {proj_code} 에 대한 데이터가 없습니다.")
        
        # 데이터 검증
        print("\n[DEBUG] 2025년 데이터 검증:")
        print(f"- 수금: {format_currency(get_column_value(df, '2025_수금'))}천원")
        print(f"- 총원가: {format_currency(get_column_value(df, '2025_총원가'))}천원")
        print(f"- 손익: {format_currency(get_column_value(df, '2025_수금_비용손익'))}천원")
        
        # 그래프 생성 실행
        print("\n[INFO] 그래프 생성 시작...")
        asyncio.run(plot_yearly_comparison(proj_code))
        print(f"[SUCCESS] 테스트 성공! 프로젝트 {proj_code}의 연도별 비교 이미지가 생성되었습니다.")
        print(f"[INFO] 이미지 경로: {os.path.join(config.STATIC_IMAGES_PATH, f'yearly_comparison_{proj_code}.png')}")
        
    except FileNotFoundError as e:
        print(f"[ERROR] 파일 오류: {e}")
    except ValueError as e:
        print(f"[ERROR] 데이터 오류: {e}")
    except Exception as e:
        print(f"[ERROR] 테스트 실패: {e}")
        print(f"[DEBUG] 오류 상세: {str(e)}")

# python Results_plot_mpl.py
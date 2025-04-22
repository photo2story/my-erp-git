# /my-flask-app/Get_data.py

import pandas as pd
import os
import re
from typing import Tuple, List, Dict, Set

# 최신 파일 탐색 및 연도 추출 함수
def find_latest_file(folder: str, prefix: str, ext: str = '.csv') -> Tuple[str, str]:
    """최신 파일을 찾고 연도를 추출합니다."""
    try:
        files = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(ext)]
        print(f"🔍 찾은 파일들: {files}")  # 디버그: 찾은 파일들 출력
        
        if not files:
            print(f"⚠️ {prefix}로 시작하는 {ext} 파일을 찾을 수 없습니다")
            return None, None
            
        def extract_date(fname):
            match = re.search(r'(\d{6,8})', fname)
            return match.group(1) if match else '000000'
            
        files.sort(key=lambda x: extract_date(x), reverse=True)
        latest_file = files[0]
        print(f"📌 선택된 최신 파일: {latest_file}")  # 디버그: 선택된 파일 출력
        
        # 연도 추출 (YYYYMM 형식에서)
        date_match = re.search(r'(\d{4})\d{2}', latest_file)
        current_year = date_match.group(1) if date_match else None
        print(f"📅 추출된 연도: {current_year}")  # 디버그: 추출된 연도 출력
        
        return os.path.join(folder, latest_file), current_year
    except Exception as e:
        print(f"❌ 파일 검색 중 오류 발생: {e}")
        return None, None

def get_column_pairs() -> List[Tuple[str, str]]:
    """전체/당년 컬럼 쌍을 반환합니다."""
    base_columns = [
        '수금', '계약금액', '수금계획', '직접인건', '간접인건',
        '직접제경', '간접제경', '외주비', '판관비', '총원가',
        '수금_비용손익', '공정계획', '외주_실행', '합계_실행예산'
    ]
    
    return [(f'전체 {col}', f'당년 {col}') for col in base_columns]

def create_funding_map(path: str) -> Dict[str, str]:
    """입찰 데이터에서 발주처별 재원조달방식 매핑을 생성합니다."""
    try:
        df = pd.read_csv(path)
        funding_map = df.set_index('발주처')['재원조달방식'].to_dict()
        print(f"✅ 발주처-재원조달방식 매핑 생성 완료: {len(funding_map)}개")
        return funding_map
    except Exception as e:
        print(f"❌ 발주처-재원조달방식 매핑 생성 실패: {e}")
        return {}

def load_erp_data(path: str) -> pd.DataFrame:
    """ERP 데이터를 로드하고 필요한 컬럼을 확인합니다."""
    try:
        if path is None:
            print("❌ ERP 파일 경로가 없습니다")
            return pd.DataFrame()
            
        df = pd.read_csv(path, encoding='utf-8-sig')
        print(f"✅ ERP 데이터 로드 완료: {df.shape}")
        
        # 전체/당년 컬럼 존재 확인 및 누락된 컬럼 생성
        column_pairs = get_column_pairs()
        for total_col, current_col in column_pairs:
            # 당년 컬럼의 경우 2025_로 시작하는 해당 컬럼 찾기
            if current_col.startswith('당년 '):
                metric = current_col.replace('당년 ', '')
                year_col = f'2025_{metric}'
                if year_col in df.columns:
                    df[current_col] = df[year_col]
                    continue
            
            if total_col not in df.columns:
                print(f"⚠️ 누락된 컬럼 생성: {total_col}")
                df[total_col] = 0
            if current_col not in df.columns:
                print(f"⚠️ 누락된 컬럼 생성: {current_col}")
                df[current_col] = 0
                
        return df
    except Exception as e:
        print(f"❌ ERP 데이터 로드 실패: {e}")
        return pd.DataFrame()

def load_contract_data(path: str) -> pd.DataFrame:
    """계약 데이터를 로드하고 금액을 천원 단위로 변환합니다."""
    try:
        if path is None:
            print("❌ 계약 파일 경로가 없습니다")
            return pd.DataFrame()
            
        df = pd.read_csv(path, encoding='utf-8-sig')
        
        # 원화공급가액 변환 (숫자가 아닌 문자 제거 후 변환)
        df['원화공급가액'] = df['원화공급가액'].astype(str).str.replace(r'[^\d.-]', '', regex=True)
        df['원화공급가액(천원)'] = pd.to_numeric(df['원화공급가액'], errors='coerce') / 1000
        
        print(f"✅ 계약현황 로드 완료: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ 계약현황 로드 실패: {e}")
        return pd.DataFrame()

def load_public_orgs(path: str) -> Set[str]:
    """공공기관 목록을 로드합니다."""
    try:
        df = pd.read_csv(path)
        # 첫 번째 컬럼의 값들을 집합으로 변환
        orgs = set(df.iloc[:, 0].values)
        print(f"✅ 공공기관 목록 로드 완료: {len(orgs)}개")
        return orgs
    except Exception as e:
        print(f"❌ 공공기관 목록 로드 실패: {e}")
        return set()

def merge_erp_contract(erp_df: pd.DataFrame, contract_df: pd.DataFrame, public_orgs: Set[str]) -> pd.DataFrame:
    # ERP: 전체/당년 항목 집계
    column_pairs = get_column_pairs()
    erp_cols = [col for pair in column_pairs for col in pair]  # 사업코드 제외
    
    erp_grouped = erp_df.groupby('사업코드')[erp_cols].sum(min_count=1).reset_index()

    # 계약: 대표 정보만 추출
    contract_main_cols = [
        '사업명', '사업구분', '국내외구분', '발주처', 'PM부서',
        '발주방법', '진행상태', 'PM', '공동도급사', '수주일자', 
        '원화공급가액(천원)'
    ]
    contract_grouped = contract_df.groupby('사업코드')[contract_main_cols].first().reset_index()

    # 발주처 기반 재원조달방식 설정
    contract_grouped['재원조달방식'] = contract_grouped['발주처'].apply(
        lambda x: '공공' if x in public_orgs else '민간'
    )
    
    # ERP 데이터 병합
    merged = pd.merge(contract_grouped, erp_grouped, on='사업코드', how='left')
    
    # 새로운 필드 계산
    # 1. 당년 예상 수금율 = (전체 수금 + 당년 수금계획) / 원화공급가액
    merged['당년_예상_수금율'] = (merged['전체 수금'] + merged['당년 수금계획']) / merged['원화공급가액(천원)']
    
    # 2. 당년 예상 외주비 = 전체 외주_실행 * 수금율 - 전체 외주비
    merged['당년_예상_외주비'] = merged['전체 외주_실행'] * merged['당년_예상_수금율'] - merged['전체 외주비']
    
    print(f"🔗 병합 완료: {merged.shape}")
    
    # 컬럼 순서 정리
    # 1. 기본 정보 컬럼
    fixed_cols = ['사업코드', '사업명', '사업구분', '국내외구분', '발주처', 
                 'PM부서', '발주방법', '진행상태', 'PM', '공동도급사', 
                 '수주일자', '원화공급가액(천원)', '재원조달방식']
    
    # 2. 당년 데이터 컬럼
    current_year_cols = [col for col in merged.columns if col.startswith('당년')] + ['당년_예상_수금율', '당년_예상_외주비']
    
    # 3. 전체 데이터 컬럼
    total_cols = [col for col in merged.columns if col.startswith('전체')]
    
    # 4. 나머지 컬럼 (있다면)
    other_cols = [col for col in merged.columns 
                 if col not in fixed_cols 
                 and col not in current_year_cols 
                 and col not in total_cols]
    
    # 컬럼 순서대로 데이터프레임 재구성
    merged = merged[fixed_cols + current_year_cols + total_cols + other_cols]
    
    return merged

def save_merged_data(df: pd.DataFrame, output_path: str):
    try:
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"📦 저장 완료: {output_path}")
    except Exception as e:
        print(f"❌ 저장 실패: {e}")

# 실행
if __name__ == "__main__":
    try:
        BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        DATA_PATH = os.path.join(BASE_PATH, 'static', 'data')
        
        # 디렉토리 존재 확인
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)
            print(f"📁 데이터 디렉토리 생성: {DATA_PATH}")
        
        OUTPUT_PATH = os.path.join(DATA_PATH, 'merged_data.csv')
        PUBLIC_ORGS_PATH = os.path.join(DATA_PATH, '공공.csv')

        # 파일 찾기
        erp_path, current_year = find_latest_file(DATA_PATH, 'erp_data')
        contract_path, _ = find_latest_file(DATA_PATH, 'contract')

        if current_year:
            print(f"📅 현재 연도: {current_year}")
        else:
            print("⚠️ 연도를 파일명에서 추출할 수 없습니다")

        print(f"📂 ERP 파일: {erp_path}")
        print(f"📂 계약 파일: {contract_path}")
        print(f"📂 공공기관 목록: {PUBLIC_ORGS_PATH}")

        # 데이터 로드
        erp_df = load_erp_data(erp_path)
        if erp_df.empty:
            raise ValueError("ERP 데이터를 로드할 수 없습니다")
            
        contract_df = load_contract_data(contract_path)
        if contract_df.empty:
            raise ValueError("계약 데이터를 로드할 수 없습니다")
            
        public_orgs = load_public_orgs(PUBLIC_ORGS_PATH)
        
        print(f"📋 공공기관 수: {len(public_orgs)}개")
        
        # 데이터 병합 및 저장
        merged_df = merge_erp_contract(erp_df, contract_df, public_orgs)
        if not merged_df.empty:
            save_merged_data(merged_df, OUTPUT_PATH)
        else:
            print("❌ 데이터 병합 실패")
            
    except Exception as e:
        print(f"❌ 프로그램 실행 중 오류 발생: {e}")

# python Get_data.py

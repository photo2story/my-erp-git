import pandas as pd
import os

# 데이터 파일 경로
BASE_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_PATH, 'static', 'data')
MERGED_FILE = os.path.join(DATA_PATH, 'merged_data.csv')

# 데이터 로드
df = pd.read_csv(MERGED_FILE)

# C20230239 사업 데이터 확인
project_code = 'C20230239'
project = df[df['사업코드'] == project_code].iloc[0]

print(f"\n#### [{project_code}] {project['사업명']}")
print(f"재원조달방식: {project['재원조달방식']}")
print(f"진행상태: {project['진행상태']}")
print(f"계약금액: {project['원화공급가액(천원)']:,.0f}천원")
print(f"전체 수금: {project['전체 수금']:,.0f}천원 (수금율 {project['전체 수금']/project['원화공급가액(천원)']*100:.2f}%)")
print(f"전체 외주실행: {project['전체 외주_실행']:,.0f}천원")
print(f"전체 외주비: {project['전체 외주비']:,.0f}천원(외주기성율 {project['전체 외주비']/project['전체 외주_실행']*100:.2f}%)")
print(f"당년 수금계획: {project['당년 수금계획']:,.0f}천원")
print(f"당년 공정계획: {project['당년 공정계획']:,.0f}천원")
print(f"잔여 외주비: {(project['전체 외주_실행'] - project['전체 외주비']):,.0f}천원")
print(f"지급 예상 외주비: {project['당년_예상_외주비']:,.0f}천원")
print(f"당년 예상 외주비: {project['당년_예상_외주비']:,.0f}천원")
print(f"외주율: {project['전체 외주_실행']/project['원화공급가액(천원)']*100:.1f}%") 
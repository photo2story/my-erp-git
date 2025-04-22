# get_heatmap.py

import pandas as pd
import squarify
import matplotlib.pyplot as plt
import os
import config

# 파일 경로 설정
input_file_path = os.path.join(config.STATIC_IMAGES_PATH, 'results_relative_divergence.csv')
output_file_path = os.path.join(config.STATIC_IMAGES_PATH, 'filtered_results_relative_divergence.csv')

# 데이터 읽기
df = pd.read_csv(input_file_path)

# Expected_Return이 0 이상인 항목만 필터링
filtered_df = df[df['Expected_Return'] > 0]

# 필터링된 데이터를 별도의 파일에 저장
filtered_df.to_csv(output_file_path, index=False)
print(f"Filtered data saved to {output_file_path}")

# 히트맵 생성
def draw_heatmap(dataframe):
    # 히트맵에 표시할 값 설정 (면적은 Expected_Return, 색상은 Relative_Divergence로 설정)
    sizes = dataframe['Expected_Return']
    labels = dataframe['Ticker']
    colors = dataframe['Relative_Divergence']

    # 색상을 정규화하여 플롯에 적용
    norm = plt.Normalize(vmin=min(colors), vmax=max(colors))
    colors_normalized = plt.cm.RdYlGn(norm(colors))

    fig, ax = plt.subplots(figsize=(16, 10))
    # 패딩 값을 줄여 아이템 간격을 최소화
    squarify.plot(sizes=sizes, label=labels, color=colors_normalized, pad=0.3, ax=ax)
    ax.axis('off')
    ax.set_title("Heatmap of Expected Return by Ticker", fontsize=20)

    # 컬러바 추가
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="RdYlGn"), ax=ax, orientation='vertical')
    cbar.set_label("Relative Divergence")

    plt.show()

# 필터링된 데이터로 히트맵 그리기
draw_heatmap(filtered_df)



#  python get_heatmap.py
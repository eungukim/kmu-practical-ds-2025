import sweetviz as sv
import pandas as pd
import os

# 파일 경로 설정
input_path = "/Users/dante/workspace/dante-code/projects/kmu-practical-ds-2025/ch02/datasets/Tweets.csv"
output_path = "/Users/dante/workspace/dante-code/projects/kmu-practical-ds-2025/ch02/datasets/Tweets_sweetviz_report.html"

# 데이터 로드
print(f"파일 로드 중: {input_path}")
df = pd.read_csv(input_path)
print(f"데이터셋 크기: {df.shape[0]} 행, {df.shape[1]} 열")

# Sweetviz 분석 수행
print("Sweetviz 분석 시작...")
sweet_report = sv.analyze(df, pairwise_analysis="on")

# HTML 리포트 저장
print(f"HTML 리포트 저장 중: {output_path}")
sweet_report.show_html(output_path, open_browser=False, layout='widescreen')

print(f"분석 완료! 리포트가 저장되었습니다: {output_path}") 
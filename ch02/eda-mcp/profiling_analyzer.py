import pandas as pd
import os

# 파일 경로 설정
input_path = "/Users/dante/workspace/dante-code/projects/kmu-practical-ds-2025/ch02/datasets/Tweets.csv"
output_path = "/Users/dante/workspace/dante-code/projects/kmu-practical-ds-2025/ch02/datasets/Tweets_profile_report.html"

# 데이터 로드
print(f"파일 로드 중: {input_path}")
df = pd.read_csv(input_path)
print(f"데이터셋 크기: {df.shape[0]} 행, {df.shape[1]} 열")

# pandas-profiling 사용을 시도합니다
try:
    from pandas_profiling import ProfileReport
    profiling_name = "pandas-profiling"
except ImportError:
    try:
        from ydata_profiling import ProfileReport
        profiling_name = "ydata-profiling"
    except ImportError:
        print("pandas-profiling 또는 ydata-profiling이 설치되어 있지 않습니다.")
        print("설치하려면 다음 명령어를 실행하세요:")
        print("pip install pandas-profiling")
        print("또는")
        print("pip install ydata-profiling")
        exit(1)

# EDA 리포트 생성
print(f"{profiling_name}을 사용하여 분석 시작...")
profile = ProfileReport(df, title="트윗 데이터셋 분석 리포트", minimal=True)

# HTML 리포트 저장
print(f"HTML 리포트 저장 중: {output_path}")
profile.to_file(output_path)

print(f"분석 완료! 리포트가 저장되었습니다: {output_path}") 
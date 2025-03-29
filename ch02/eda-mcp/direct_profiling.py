import pandas as pd
import os
import argparse

def create_profile_report(input_path, output_path, title="데이터셋 프로파일 보고서", minimal=True):
    """pandas-profiling 또는 ydata-profiling을 사용하여 프로파일 보고서 생성"""
    # 프로파일링 라이브러리 임포트 시도
    profiling_available = False
    profiling_name = None
    
    try:
        # 먼저 ydata_profiling 시도 (새 버전)
        try:
            from ydata_profiling import ProfileReport
            profiling_name = "ydata-profiling"
            profiling_available = True
        except ImportError:
            # 그 다음 pandas_profiling 시도 (구 버전)
            try:
                from pandas_profiling import ProfileReport
                profiling_name = "pandas-profiling"
                profiling_available = True
            except ImportError:
                profiling_available = False
    except Exception:
        profiling_available = False
    
    if not profiling_available:
        print("pandas-profiling 또는 ydata-profiling이 설치되어 있지 않습니다.")
        print("설치하려면 다음 명령어를 실행하세요:")
        print("pip install ydata-profiling")
        return False
    
    # 데이터 로드
    print(f"데이터셋 로드 중: {input_path}")
    df = pd.read_csv(input_path)
    print(f"데이터셋 크기: {df.shape[0]} 행, {df.shape[1]} 열")
    
    # 확장자가 .html이 아니면 추가
    if not output_path.lower().endswith('.html'):
        output_path += '.html'
    
    try:
        # 프로파일 리포트 생성
        print(f"{profiling_name}을 사용하여 데이터 프로파일링 시작...")
        profile = ProfileReport(df, title=title, minimal=minimal)
        
        # 결과를 HTML 파일로 저장
        print(f"HTML 보고서 저장 중: {output_path}")
        profile.to_file(output_path)
        
        print(f"분석 완료! 보고서가 저장되었습니다: {output_path}")
        return True
    except Exception as e:
        print(f"보고서 생성 중 오류가 발생했습니다: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pandas-profiling을 사용한 데이터셋 프로파일링")
    parser.add_argument("input_path", help="입력 CSV 파일 경로")
    parser.add_argument("output_path", help="출력 HTML 파일 경로")
    parser.add_argument("--title", default="데이터셋 프로파일 보고서", help="보고서 제목")
    parser.add_argument("--minimal", action="store_true", help="최소 프로파일링 모드 사용")
    
    args = parser.parse_args()
    create_profile_report(args.input_path, args.output_path, args.title, args.minimal) 
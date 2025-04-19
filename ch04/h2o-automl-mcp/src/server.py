from mcp.server.fastmcp import FastMCP
import sys
import os
import json
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import io
from fastmcp import Image
import base64


# 환경 변수에서 설정 가져오기
host = os.getenv("MCP_HOST", "0.0.0.0")
port = int(os.getenv("MCP_PORT", "8000"))
transport = os.getenv("MCP_TRANSPORT", "stdio")

# H2O 서버 초기화
h2o.init(ip="127.0.0.1", port=54321, start_h2o=True, nthreads=-1, max_mem_size="1g")

# 모델 저장 경로
MODEL_DIR = os.getenv("MODEL_DIR", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

mcp = FastMCP(
    name="h2o-automl-server",
    instructions="H2O AutoML을 활용한 자동 머신러닝 서버",
    host=host,
    port=port,
    debug=True  # 디버그 모드 활성화
)

# 계산기 도구 추가
@mcp.tool()
def add(a: int, b: int) -> int:
    """두 숫자를 더합니다"""
    return a + b

# H2O AutoML 관련 도구들

@mcp.tool()
def load_dataset(file_path: str, header: bool = True) -> str:
    """
    CSV 파일을 H2O 데이터프레임으로 로드합니다
    
    Args:
        file_path: 로드할 CSV 파일 경로
        header: CSV 파일에 헤더가 있는지 여부
    
    Returns:
        데이터셋 정보 요약
    """
    try:
        data = h2o.import_file(file_path, header=header)
        return f"데이터셋 로드 완료: {data.shape[0]}행 x {data.shape[1]}열\n{data.head().as_data_frame().to_string()}"
    except Exception as e:
        return f"데이터 로드 오류: {str(e)}"

@mcp.tool()
def get_dataset_info(file_path: str) -> str:
    """
    데이터셋 정보를 가져옵니다
    
    Args:
        file_path: 데이터셋 파일 경로
    
    Returns:
        데이터셋 정보 요약
    """
    try:
        data = h2o.import_file(file_path)
        # 기본 통계 정보 추출
        summary = data.describe().as_data_frame()
        
        # 데이터 타입 정보
        types = pd.DataFrame({
            'Column': data.columns,
            'Type': [data[col].type for col in data.columns]
        })
        
        result = f"데이터셋 정보 ({data.shape[0]}행 x {data.shape[1]}열):\n"
        result += f"기본 통계:\n{summary.to_string()}\n\n"
        result += f"데이터 타입:\n{types.to_string()}"
        
        return result
    except Exception as e:
        return f"데이터셋 정보 조회 오류: {str(e)}"

@mcp.tool()
def train_automl(
    file_path: str, 
    target_column: str,
    max_models: int = 20,
    max_runtime_secs: int = 300,
    model_name: str = "automl_model"
) -> str:
    """
    H2O AutoML을 사용하여 자동으로 모델을 훈련합니다
    
    Args:
        file_path: 데이터셋 파일 경로
        target_column: 예측할 대상 컬럼명
        max_models: 최대 생성할 모델 수
        max_runtime_secs: 최대 훈련 시간(초)
        model_name: 저장할 모델 이름
    
    Returns:
        훈련 결과 요약
    """
    try:
        # 데이터 로드
        data = h2o.import_file(file_path)
        
        # 피처와 타겟 분리
        x = data.columns
        y = target_column
        x.remove(y)
        
        # 데이터 분할
        train, valid, test = data.split_frame(ratios=[0.7, 0.15], seed=1234)
        
        # AutoML 설정 및 훈련
        aml = H2OAutoML(
            max_models=max_models,
            max_runtime_secs=max_runtime_secs,
            seed=1234,
            sort_metric="AUTO"
        )
        
        aml.train(x=x, y=y, training_frame=train, validation_frame=valid)
        
        # 모델 저장
        model_path = h2o.save_model(model=aml.leader, path=MODEL_DIR, force=True)
        
        # 결과 반환
        result = f"AutoML 훈련 완료!\n"
        result += f"최고 성능 모델: {aml.leader.model_id}\n"
        result += f"모델 저장 경로: {model_path}\n"
        result += f"리더보드 상위 5개 모델:\n{aml.leaderboard.head(5).as_data_frame().to_string()}"
        
        return result
    
    except Exception as e:
        return f"모델 훈련 오류: {str(e)}"

@mcp.tool()
def evaluate_model(model_name: str, test_file_path: str, target_column: str) -> str:
    """
    저장된 모델을 평가합니다
    
    Args:
        model_name: 평가할 모델 이름
        test_file_path: 테스트 데이터셋 파일 경로
        target_column: 타겟 컬럼명
    
    Returns:
        모델 평가 결과
    """
    try:
        # 모델 로드
        model_path = os.path.join(MODEL_DIR, model_name)
        model = h2o.load_model(model_path)
        
        # 테스트 데이터 로드
        test_data = h2o.import_file(test_file_path)
        
        # 평가
        perf = model.model_performance(test_data)
        
        # 결과 반환
        if model.model_category == "Binomial":
            result = f"모델 평가 결과 (이진 분류):\n"
            result += f"AUC: {perf.auc()}\n"
            result += f"로그 손실: {perf.logloss()}\n"
            result += f"정확도: {perf.accuracy()[0][1]}\n"
            result += f"정밀도: {perf.precision()[0][1]}\n"
            result += f"재현율: {perf.recall()[0][1]}\n"
            result += f"F1 점수: {perf.F1()[0][1]}"
        elif model.model_category == "Multinomial":
            result = f"모델 평가 결과 (다중 분류):\n"
            result += f"정확도: {perf.accuracy()[0][1]}\n"
            result += f"로그 손실: {perf.logloss()}\n"
            result += f"평균 정밀도: {sum(perf.precision()[0][1:]) / len(perf.precision()[0][1:])}\n"
            result += f"평균 재현율: {sum(perf.recall()[0][1:]) / len(perf.recall()[0][1:])}\n"
            result += f"평균 F1 점수: {sum(perf.F1()[0][1:]) / len(perf.F1()[0][1:])}"
        elif model.model_category == "Regression":
            result = f"모델 평가 결과 (회귀):\n"
            result += f"MSE: {perf.mse()}\n"
            result += f"RMSE: {perf.rmse()}\n"
            result += f"MAE: {perf.mae()}\n"
            result += f"R-squared: {perf.r2()}"
        else:
            result = f"모델 평가 결과:\n"
            result += str(perf)
            
        return result
        
    except Exception as e:
        return f"모델 평가 오류: {str(e)}"

@mcp.tool()
def predict(model_name: str, file_path: str) -> str:
    """
    저장된 모델을 사용하여 예측을 수행합니다
    
    Args:
        model_name: 사용할 모델 이름
        file_path: 예측할 데이터셋 파일 경로
    
    Returns:
        예측 결과 요약
    """
    try:
        # 모델 로드
        model_path = os.path.join(MODEL_DIR, model_name)
        model = h2o.load_model(model_path)
        
        # 데이터 로드
        data = h2o.import_file(file_path)
        
        # 예측
        preds = model.predict(data)
        
        # 결과 데이터프레임 생성
        if model.model_category in ["Binomial", "Multinomial"]:
            pred_df = preds.as_data_frame()
            result = f"예측 결과 (처음 10개):\n{pred_df.head(10).to_string()}"
        else:
            pred_df = pd.DataFrame({
                "예측값": preds.as_data_frame().values.flatten()
            })
            result = f"예측 결과 (처음 10개):\n{pred_df.head(10).to_string()}"
            
        return result
        
    except Exception as e:
        return f"예측 오류: {str(e)}"

@mcp.tool()
def visualize_model_performance(model_name: str, test_file_path: str, target_column: str) -> str:
    """
    모델 성능을 시각화합니다
    
    Args:
        model_name: 시각화할 모델 이름
        test_file_path: 테스트 데이터셋 파일 경로
        target_column: 타겟 컬럼명
    
    Returns:
        성능 시각화 결과 (이미지로 변환되어 반환)
    """
    try:
        # 모델 로드
        model_path = os.path.join(MODEL_DIR, model_name)
        model = h2o.load_model(model_path)
        
        # 테스트 데이터 로드
        test_data = h2o.import_file(test_file_path)
        
        # 예측
        preds = model.predict(test_data)
        
        # matplotlib 설정
        plt.figure(figsize=(10, 6))
        
        # 모델 유형에 따른 시각화
        if model.model_category == "Binomial":
            # ROC 커브
            perf = model.model_performance(test_data)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(perf.fprs, perf.tprs)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve (AUC: {perf.auc():.4f})')
            plt.grid(True)
        
        elif model.model_category == "Regression":
            # 실제값 vs 예측값
            actual = test_data[target_column].as_data_frame().values.flatten()
            predicted = preds.as_data_frame().values.flatten()
            
            plt.scatter(actual, predicted, alpha=0.5)
            plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'k--')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title('Actual vs Predicted')
            plt.grid(True)
        
        else:
            # 다른 모델 유형에 대한 시각화
            plt.text(0.5, 0.5, f"시각화가 지원되지 않는 모델 유형: {model.model_category}", 
                    ha='center', va='center', fontsize=12)
            plt.axis('off')
        
        # 이미지를 바이트로 변환
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Base64로 인코딩
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return f"data:image/png;base64,{img_str}"
        
    except Exception as e:
        return f"시각화 오류: {str(e)}"

# 기존 리소스들 유지
@mcp.resource("echo://{message}")
def echo_resource(message: str) -> str:
    """메시지를 반향합니다"""
    return f"ECHO: {message}"

# 건강 상태 체크 엔드포인트
@mcp.resource("health://check")
def health_check() -> dict:
    """서버 상태 확인"""
    return {"status": "healthy", "version": "1.0.0"}

# H2O 상태 체크 엔드포인트 추가
@mcp.resource("h2o://status")
def h2o_status() -> dict:
    """H2O 클러스터 상태 확인"""
    try:
        status = h2o.cluster_status()
        return {
            "status": "healthy",
            "cluster_name": status["cluster_name"],
            "total_nodes": status["cloud_size"],
            "total_memory": status["total_bytes"] / (1024**3),  # GB
            "version": status["version"]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    print("H2O AutoML 서버 시작 중...")
    print(f"서버 주소: {host}:{port}, 전송 방식: {transport}")
    
    # transport만 설정
    mcp.run(transport=transport)
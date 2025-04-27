# %%
from openai.types.responses import (ResponseTextDeltaEvent, ResponseFunctionCallArgumentsDeltaEvent)
from agents import Agent, Runner
from agents import function_tool
from dotenv import load_dotenv
import uuid
load_dotenv()

# %%
data_storage = {}

# %%
@function_tool
def kaggle_list_datasets(keyword: str) -> str:
    
    """
    검색 쿼리에 맞는 Kaggle 데이터셋 목록을 조회합니다
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    try:
        # 데이터셋 검색
        api = KaggleApi()
        api.authenticate()
        datasets = api.dataset_list(search=keyword, sort_by="hottest")
        
        result = []
        count = 0
        for dataset in datasets:
            if count >= 10:
                break
            dataset_info = {
                "ref": dataset.ref,  # owner/dataset-name 형식
                "title": dataset.title,
            }
            
            # 안전하게 속성 추가
            if hasattr(dataset, 'size'):
                dataset_info["size"] = dataset.size
            if hasattr(dataset, 'lastUpdated'):
                dataset_info["last_updated"] = str(dataset.lastUpdated)
            if hasattr(dataset, 'downloadCount'):
                dataset_info["download_count"] = dataset.downloadCount
            if hasattr(dataset, 'voteCount'):
                dataset_info["vote_count"] = dataset.voteCount
            if hasattr(dataset, 'tags'):
                dataset_info["tags"] = dataset.tags
            if hasattr(dataset, 'usabilityRating'):
                dataset_info["usability_rating"] = dataset.usabilityRating
            
            result.append(dataset_info)
            count += 1
        
        return {
            "success": True,
            "count": len(result),
            "datasets": result
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"데이터셋 목록 조회 실패: {str(e)}"
        }

# %%
@function_tool
def kaggle_download_dataset(dataset_ref: str) -> str:
    """
    Kaggle 데이터셋을 다운로드합니다
    """
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    try:
        # Kaggle API로 데이터셋 다운로드
        api.dataset_download_files(dataset_ref, path=f"datasets/{dataset_ref}/", unzip=True)
        import os
        import pandas as pd
        
        # 다운로드 성공 시 처리
        dataset_path = f"datasets/{dataset_ref}/"
        
        # 디렉토리 내 파일 목록 가져오기
        files = os.listdir(dataset_path)
        
        # CSV 파일 찾기
        csv_files = [f for f in files if f.endswith('.csv')]
        
        if csv_files:
            # 첫 번째 CSV 파일 로드
            first_csv = os.path.join(dataset_path, csv_files[0])
            df = pd.read_csv(first_csv)
            
            # 데이터 저장소에 저장
            data_storage[dataset_ref] = df
            
            return {
                "success": True,
                "message": f"데이터셋 다운로드 및 로드 성공 : (저장소 key : {dataset_ref})",
                "file_loaded": csv_files[0],
                "rows": len(df),
                "columns": len(df.columns)
            }
        else:
            return {
                "success": True,
                "message": f"데이터셋 다운로드 성공했으나 CSV 파일이 없습니다: {dataset_ref}",
                "files_available": files
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"데이터셋 다운로드 실패: {str(e)}"
        }

# %%
@function_tool
def show_df_stats(key: str) -> str:
    """저장된 데이터프레임의 기본 통계 정보 출력
    저장된 데이터 형태  (key) : Pandas DataFrame
    """
    if key not in data_storage:
        return "해당 URL의 CSV 파일을 찾을 수 없음"
    
    df = data_storage[key]
    return f"""
    행 수: {len(df)}
    열 수: {len(df.columns)}
    결측치 수: {df.isnull().sum().sum()}
    컬럼 목록: {df.columns.tolist()}
    컬럼 타입: {df.dtypes.to_dict()}
    기초 통계: {df.describe().to_dict()}
    """

# %%
@function_tool
def random_sample(key: str, n: int) -> str:
    """저장된 데이터프레임에서 랜덤 샘플 추출
    저장된 데이터 형태  (key) : Pandas DataFrame
    """
    if key not in data_storage:
        return "해당 URL의 CSV 파일을 찾을 수 없음"
    new_key = str(uuid.uuid4())
    df = data_storage[key]
    random_sample_df = df.sample(n)
    data_storage[new_key] = random_sample_df
    return f"랜덤 샘플 추출 완료: (저장소 key : {new_key})"

# %%
@function_tool
def prep_df(key: str, target_column: str) -> str:
    """저장된 데이터프레임을 머신러닝 모델에 적합한 형태로 전처리
    저장된 데이터 형태  (key) : Pandas DataFrame
    새로 저장할 데이터 형태 (new_key) : (X, y) 형태의 튜플
    """
    new_key = str(uuid.uuid4())
    df = data_storage[key]
    if target_column not in df.columns:
        return f"해당 컬럼 {target_column}이 존재하지 않습니다."
    df = df[df[target_column].notna()]
    
    # 실수형 데이터만 남기기
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df = df[numeric_cols]
    
    if target_column not in df.columns:
        return f"타겟 컬럼 {target_column}이 실수형 데이터가 아니어서 제거되었습니다."
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    data_storage[new_key] = (X, y)
    return f"데이터프레임 전처리 완료 (실수형 데이터만 포함): (저장소 key : {new_key})"

# %%
@function_tool
def visualize_tsne(key: str) -> str:
    """
    저장된 데이터프레임을 t-SNE로 시각화
    저장된 데이터 형태 (key) : (X, y) 형태의 튜플
    새로 저장할 데이터 형태 (new_key) : Plotly 객체
    시각화 결과는 PNG 파일로 저장됩니다.
    """
    from utils import TSNEVisualizer
    import plotly.io as pio
    import os
    new_key = str(uuid.uuid4())
    
    # 파일 저장을 위한 경로 설정
    output_dir = "visualization_output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"tsne_{new_key}.png")
    
    (X, y) = data_storage[key]
    fig = TSNEVisualizer.visualize(X, y, perplexity=5)
    
    # PNG 파일로 저장
    fig.write_image(output_path)
    
    data_storage[new_key] = fig
    
    return f"t-SNE 시각화 완료: (저장소 key : {new_key}, 파일 경로: {output_path})"

# %%
@function_tool
def under_resample(key: str) -> str:
    """
    저장된 데이터프레임을 언더리샘플링
    저장된 데이터 형태  (key) : (X, y) 형태의 튜플
    새로 저장할 데이터 형태 (new_key) : (X, y) 형태의 튜플
    """
    from utils import ImbalancedDataAnalyzer
    new_key = str(uuid.uuid4())
    (X, y) = data_storage[key]
    analyzer = ImbalancedDataAnalyzer(X, y)
    X_resampled, y_resampled = analyzer.random_undersample()
    data_storage[new_key] = (X_resampled, y_resampled)
    return f"언더리샘플링 완료: (저장소 key : {new_key})"

# %%
@function_tool
def isolation_forest(key: str, contamination: float) -> str:
    """
    저장된 데이터프레임을 Isolation Forest로 분석
    저장된 데이터 형태  (key) : (X, y) 형태의 튜플
    새로 저장할 데이터 형태 (new_key) : (Plotly 객체, y, score) 형태의 튜플
    시각화 결과는 PNG 파일로 저장됩니다.
    """
    from sklearn.ensemble import IsolationForest
    import plotly.express as px
    import os
    new_key = str(uuid.uuid4())
    
    # 파일 저장을 위한 경로 설정
    output_dir = "visualization_output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"isolation_forest_{new_key}.png")
    
    (X, y) = data_storage[key]
    model = IsolationForest(contamination=contamination, max_samples='auto')
    model.fit(X)
    score = model.score_samples(X)
    score = -1 * score
    fig = px.histogram(x=score, nbins=100, labels={'x':'Score'}, title="평가")
    fig.update_layout(width=600, height=400)
    
    # PNG 파일로 저장
    fig.write_image(output_path)
    
    # 그래프와 함께 y와 score도 저장
    data_storage[new_key] = (fig, y, score)
    
    # 추가로 precision-recall 곡선을 위한 별도 키 저장
    pr_key = str(uuid.uuid4())
    data_storage[pr_key] = (y, score)
    
    return f"Isolation Forest 분석 완료: (저장소 key : {new_key}, PR 곡선용 key: {pr_key}, 파일 경로: {output_path})"

@function_tool
def prec_rec_f1_curve(key: str):
    """
    저장된 데이터프레임을 Precision, Recall, F1 Score 곡선으로 시각화
    저장된 데이터 형태  (key) : (y, score) 형태의 튜플
    새로 저장할 데이터 형태 (new_key) : (precision, recall, f1, thresholds) 형태의 튜플
    """
    import numpy as np
    import plotly.graph_objects as go
    from sklearn.metrics import precision_recall_curve
    
    new_key = str(uuid.uuid4())
    y, score = data_storage[key]
    
    precision, recall, thresholds = precision_recall_curve(y, score, pos_label=1)
    f1 = 2 / (1/precision + 1/recall)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=np.delete(precision, -1), mode='lines', name='precision'))
    fig.add_trace(go.Scatter(x=thresholds, y=np.delete(recall, -1), mode='lines', name='recall'))
    fig.add_trace(go.Scatter(x=thresholds, y=np.delete(f1, -1), mode='lines', name='f1'))
    fig.update_layout(title='Anomaly Score에 따른 Precision, Recall, F1 Score',
                      xaxis_title='Anomaly Score',
                      yaxis_title='Score',
                      legend_title='Score Type')
    fig.show()
    
    data_storage[new_key] = (precision, recall, f1, thresholds)

    return precision, recall, f1, thresholds

# %%
@function_tool
def show_tool_list() -> str:
    """
    현재 사용 가능한 도구 목록을 출력
    """
    tool_names = [
        "show_tool_list",
        "kaggle_list_datasets",
        "kaggle_download_dataset",
        "show_df_stats",
        "random_sample",
        "prep_df",
        "visualize_tsne",
        "under_resample",
        "isolation_forest",
        "prec_rec_f1_curve",
    ]
    return f"현재 사용 가능한 도구 목록: {', '.join(tool_names)}"



# %%
agent = Agent(
    name="Assistant",
    instructions=
        "너는 데이터 분석 전문가야. "
        "가능한 한 항상 제공된 도구를 사용하는 것을 기억해야되. "
        "너가 사용가능한 도구 목록을 먼저 출력해놓고 시작해. show_tool_list 도구를 사용하면 됨. "
        "자신의 지식에 너무 의존하지 말고 대신 질문에 답하는 데 도움이 되는 도구를 사용하세요. "
        "python 코드를 직접구현하는 방식을 사용하지 말고, 도구에 의존하세요. "
        "t-SNE 시각화나 다른 시각화를 사용할 때 샘플 수가 적어 'perplexity must be less than n_samples' 등의 오류가 발생하면, "
        "해당 시각화 단계를 건너뛰고 사용자에게 샘플 수가 부족하여 시각화가 불가능하다고 알려주세요.",
    model="gpt-4o",
    tools=[
        show_tool_list,
        kaggle_list_datasets,
        kaggle_download_dataset,
        show_df_stats,
        random_sample,
        prep_df,
        visualize_tsne,
        under_resample,
        isolation_forest,
        prec_rec_f1_curve,
    ]
)

# %%
async def run_agent():
    # we do need to reinitialize our runner before re-executing
    response = Runner.run_streamed(
        starting_agent=agent,
        max_turns=20,
        input=(
            "1. 사기 이상탐지 관련 데이터셋을 캐글에서 조회한후,"
            "2. 첫번째 데이터셋을 다운로드해줘."
            "3. 다운로드 받은 데이터셋에서 3000개 레코드만 랜덤 샘플링해줘."
            "4. 3000개 레코드의 기초 통계량을 출력해줘."
            "5. 앞에서 처리한 랜덤 샘플링한 3000개 레코드를 머신러닝 모델에 적합한 형태로 전처리해줘."
            "6. 전처리한 데이터를 TSNE 시각화해줘."
            "7. 전처리한 데이터를 SMOTE 리샘플링을 해줘."
            "8. SMOTE 리샘플링한 데이터를 TSNE 시각화해줘."
            "9. SMOTE 리샘플링한 데이터를 Isolation Forest 분석해줘."
            "10. Isolation Forest 분석한 데이터를 Precision, Recall, F1 Score 곡선으로 시각화해줘." 
            "11. 이 내용을 바탕으로 이상탐지 보고서를 작성해줘. (도구 없이)"
        )
    )

    async for event in response.stream_events():
        if event.type == "raw_response_event":
            if isinstance(event.data, ResponseFunctionCallArgumentsDeltaEvent):
                # 도구 호출을 위한 스트리밍 매개변수
                print(event.data.delta, end="", flush=True)
            elif isinstance(event.data, ResponseTextDeltaEvent):
                # 스트리밍된 최종 응답 토큰
                print(event.data.delta, end="", flush=True)
        elif event.type == "agent_updated_stream_event":
            # 현재 사용 중인 에이전트 정보
            print(f"> 현재 에이전트: {event.new_agent.name}")
        elif event.type == "run_item_stream_event":
            # 사용자나 다운스트림 프로세스로 스트리밍할 정보를 포함하는 이벤트
            if event.name == "tool_called":
                # 모든 도구 토큰이 스트리밍된 후 전체 도구 호출 정보
                print()
                print(f"> 도구 호출됨, 이름: {event.item.raw_item.name}")
                print(f"> 도구 호출됨, 인자: {event.item.raw_item.arguments}")
            elif event.name == "tool_output":
                # 도구 실행 결과
                print(f"> 도구 출력: {event.item.raw_item['output']}")

# 이 함수를 실행하기 위한 코드
import asyncio
asyncio.run(run_agent())

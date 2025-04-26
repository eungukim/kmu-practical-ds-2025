import pandas as pd
import os
import tempfile
import json
from typing import List, Optional, Dict, Any
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import argparse
import io
import base64
import shutil

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="kaggle-mcp-server",
    instructions="Kaggle API를 활용한 데이터셋 조회, 다운로드 및 분석을 수행하는 MCP 서버",
    dependencies=[
        "pandas",
        "kaggle",
        "shutil",
        "tempfile",
        "json"
    ]
)

@mcp.tool('authenticate', "Kaggle API 인증")
async def authenticate(
    kaggle_username: str,
    kaggle_key: str
) -> dict:
    """Kaggle API 인증을 수행합니다"""
    # 환경 변수 설정
    os.environ['KAGGLE_USERNAME'] = kaggle_username
    os.environ['KAGGLE_KEY'] = kaggle_key
    
    try:
        # API 인증 시도
        api = KaggleApi()
        api.authenticate()
        return {
            "success": True,
            "message": "Kaggle API 인증에 성공했습니다."
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Kaggle API 인증 실패: {str(e)}"
        }

# 데이터셋 목록 조회 함수
@mcp.tool('list_datasets', "Kaggle 데이터셋 목록 조회")
async def list_datasets(
    search_query: str = "",
    max_results: int = 10,
    sort_by: str = "hottest"
) -> dict:
    """
    검색 쿼리에 맞는 Kaggle 데이터셋 목록을 조회합니다
    
    sort_by 매개변수는 다음 값 중 하나여야 합니다:
    - 'hottest': 인기순
    - 'votes': 투표순
    - 'updated': 업데이트순
    - 'active': 활동순
    - 'published': 출판순
    """
    try:
        # 유효한 정렬 옵션 확인
        valid_sort_options = ['hottest', 'votes', 'updated', 'active', 'published']
        if sort_by not in valid_sort_options:
            return {
                "success": False,
                "message": f"유효하지 않은 정렬 옵션입니다. 다음 중 하나를 선택하세요: {', '.join(valid_sort_options)}"
            }
        
        # Kaggle API 인증
        api = KaggleApi()
        api.authenticate()
        
        # 데이터셋 검색
        datasets = api.dataset_list(search=search_query, sort_by=sort_by)
        
        result = []
        count = 0
        for dataset in datasets:
            if count >= max_results:
                break
            dataset_info = {
                "ref": dataset.ref,  # owner/dataset-name 형식
                "title": dataset.title,
                "tags": dataset.tags if hasattr(dataset, 'tags') else []
            }
            
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

# 데이터셋 정보 조회 함수
@mcp.tool('dataset_info', "Kaggle 데이터셋 정보 조회")
async def dataset_info(
    dataset_ref: str
) -> dict:
    """데이터셋의 상세 정보를 조회합니다"""
    try:
        api = KaggleApi()
        api.authenticate()
        
        # owner/dataset-name 형식에서 분리
        owner, dataset_name = dataset_ref.split('/')
        
        # 데이터셋 정보 조회
        dataset = api.dataset_view(owner, dataset_name)
        
        # 파일 목록 조회
        files = api.dataset_list_files(dataset_ref).files
        file_list = [{"name": file.name} for file in files]
        
        return {
            "success": True,
            "dataset": {
                "ref": dataset_ref,
                "title": dataset.title,
                "files": file_list,
                "description": dataset.description,
                "tags": dataset.tags if hasattr(dataset, 'tags') else []
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"데이터셋 정보 조회 실패: {str(e)}"
        }

# 데이터셋 미리보기 함수
@mcp.tool('preview_dataset', "Kaggle 데이터셋 미리보기")
async def preview_dataset(
    dataset_ref: str,
    file_name: str,
    rows: int = 5
) -> dict:
    """데이터셋 파일의 내용을 미리 봅니다"""
    try:
        api = KaggleApi()
        api.authenticate()
        
        # 임시 디렉토리 생성
        temp_dir = tempfile.mkdtemp()
        try:
            # 파일 다운로드
            api.dataset_download_file(dataset_ref, file_name, temp_dir)
            
            # 다운로드된 파일 경로
            file_path = os.path.join(temp_dir, file_name)
            
            # 파일 확장자 확인
            if file_name.endswith('.csv'):
                # CSV 파일 읽기
                df = pd.read_csv(file_path)
                preview = df.head(rows).to_dict()
                return {
                    "success": True,
                    "preview": preview,
                    "rows": min(rows, len(df)),
                    "total_rows": len(df),
                    "columns": list(df.columns)
                }
            else:
                # 텍스트 파일 읽기
                with open(file_path, 'r') as f:
                    lines = f.readlines()[:rows]
                return {
                    "success": True,
                    "preview": lines,
                    "rows": len(lines)
                }
        finally:
            # 임시 디렉토리 정리
            shutil.rmtree(temp_dir)
    except Exception as e:
        return {
            "success": False,
            "message": f"데이터셋 미리보기 실패: {str(e)}"
        }

# 데이터셋 다운로드 함수
@mcp.tool('download_dataset', "Kaggle 데이터셋 다운로드")
async def download_dataset(
    dataset_ref: str,
    client_path: str,
    unzip: bool = True
) -> dict:
    """
    Kaggle 데이터셋을 다운로드하고 파일 URI를 반환합니다
    
    Arguments:
        dataset_ref: Kaggle 데이터셋 참조 (예: "owner/dataset-name")
        client_path: 클라이언트에 저장할 경로
        unzip: 데이터셋 압축 해제 여부
    """
    try:
        api = KaggleApi()
        api.authenticate()
        
        # 서버 측 임시 디렉토리 생성
        temp_dir = tempfile.mkdtemp()
        dataset_id = f"{dataset_ref.replace('/', '-')}-{os.path.basename(temp_dir)}"
        server_dir = os.path.join('/tmp/kaggle-mcp-data', dataset_id)
        os.makedirs(server_dir, exist_ok=True)
        
        # 데이터셋 다운로드
        api.dataset_download_files(dataset_ref, path=server_dir, unzip=unzip)
        
        # 다운로드된 파일 목록 확인
        file_list = []
        for root, _, files in os.walk(server_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, server_dir)
                
                # 클라이언트 경로 생성
                client_file_path = os.path.join(client_path, rel_path)
                
                file_info = {
                    "name": rel_path,
                    "client_path": client_file_path,
                    "size": os.path.getsize(file_path),
                    "mime_type": "application/octet-stream",
                    "resource_uri": f"kaggle-data://{dataset_id}/{rel_path}",
                    "server_path": file_path
                }
                
                # 파일 MIME 타입 추정
                if rel_path.endswith('.csv'):
                    file_info["mime_type"] = "text/csv"
                elif rel_path.endswith('.json'):
                    file_info["mime_type"] = "application/json"
                elif rel_path.endswith('.txt'):
                    file_info["mime_type"] = "text/plain"
                    
                file_list.append(file_info)
        
        return {
            "success": True,
            "message": f"데이터셋 '{dataset_ref}'가 성공적으로 다운로드되었습니다. 파일은 리소스 URI를 통해 접근할 수 있습니다.",
            "files": file_list,
            "dataset_ref": dataset_ref,
            "dataset_id": dataset_id,
            "dataset_dir": server_dir,
            "client_base_path": client_path
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"데이터셋 다운로드 실패: {str(e)}"
        }

# 파일 저장 함수 (컨테이너에서 클라이언트로 파일 전송)
@mcp.tool('save_kaggle_file', "Kaggle 리소스 URI의 파일을 클라이언트에 저장")
async def save_kaggle_file(
    resource_uri: str,
    client_path: str,
    force_overwrite: bool = False
) -> dict:
    """
    리소스 URI의 파일을 클라이언트에 저장합니다
    
    Arguments:
        resource_uri: 파일의 리소스 URI (download_dataset에서 반환됨)
        client_path: 클라이언트에 저장할 경로
        force_overwrite: 기존 파일 덮어쓰기 여부
    """
    try:
        # 리소스 URI 파싱
        if not resource_uri.startswith("kaggle-data://"):
            return {
                "success": False,
                "message": "유효하지 않은 리소스 URI입니다. 'kaggle-data://' 형식이어야 합니다."
            }
        
        # URI에서 데이터셋 ID와 파일 경로 추출
        uri_parts = resource_uri[14:].split('/', 1)
        if len(uri_parts) != 2:
            return {
                "success": False,
                "message": "유효하지 않은 리소스 URI 형식입니다."
            }
        
        dataset_id, file_rel_path = uri_parts
        server_dir = os.path.join('/tmp/kaggle-mcp-data', dataset_id)
        server_file_path = os.path.join(server_dir, file_rel_path)
        
        # 서버 파일 존재 확인
        if not os.path.exists(server_file_path):
            return {
                "success": False,
                "message": f"서버에서 파일을 찾을 수 없습니다: {server_file_path}"
            }
        
        # 클라이언트 디렉토리 생성
        client_dir = os.path.dirname(client_path)
        if client_dir:
            os.makedirs(client_dir, exist_ok=True)
        
        # 파일 존재 여부 확인
        if os.path.exists(client_path) and not force_overwrite:
            return {
                "success": False,
                "message": f"파일이 이미 존재합니다: {client_path}. 덮어쓰려면 force_overwrite=True로 설정하세요."
            }
        
        # 파일 내용 읽기
        with open(server_file_path, 'rb') as f:
            file_content = f.read()
        
        # 파일 내용 저장
        with open(client_path, 'wb') as f:
            f.write(file_content)
        
        return {
            "success": True,
            "message": f"파일이 성공적으로 저장되었습니다: {client_path}",
            "file_path": client_path,
            "size": os.path.getsize(client_path),
            "resource_uri": resource_uri
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"파일 저장 실패: {str(e)}"
        }

# CSV 파일 분석 함수
@mcp.tool('analyze_dataset', "CSV 데이터셋 분석")
async def analyze_dataset(
    file_path: str,
    sample_size: int = 5
) -> dict:
    """
    CSV 파일의 통계 정보를 분석합니다
    
    Arguments:
        file_path: CSV 파일 경로
        sample_size: 샘플로 반환할 행 수
    """
    try:
        # CSV 파일 읽기
        df = pd.read_csv(file_path)
        
        # 기본 정보
        file_info = {
            "path": file_path,
            "columns": list(df.columns),
            "shape": list(df.shape),
            "sample": df.head(sample_size).to_dict(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        # 통계 정보
        statistics = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                statistics[col] = df[col].describe().to_dict()
        
        # 결측치 정보
        missing_values = {col: int(df[col].isna().sum()) for col in df.columns}
        missing_percent = {col: float(df[col].isna().sum() / len(df) * 100) for col in df.columns}
        
        # 상관관계 (숫자형 컬럼만)
        numeric_cols = df.select_dtypes(include=['number']).columns
        correlation = {}
        if len(numeric_cols) > 0:
            corr_matrix = df[numeric_cols].corr().to_dict()
            correlation = corr_matrix
        
        return {
            "success": True,
            "file_info": file_info,
            "statistics": statistics,
            "missing_values": missing_values,
            "missing_percent": missing_percent,
            "correlation": correlation
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"데이터셋 분석 실패: {str(e)}"
        }

if __name__ == "__main__":
    # 명령줄 인수 파싱 추가
    parser = argparse.ArgumentParser(description="Kaggle MCP 서버")
    parser.add_argument("--kaggle_username", help="Kaggle API 사용자 이름")
    parser.add_argument("--kaggle_key", help="Kaggle API 키")
    args = parser.parse_args()

    # 인증 정보가 제공된 경우 환경 변수 설정
    if args.kaggle_username and args.kaggle_key:
        os.environ['KAGGLE_USERNAME'] = args.kaggle_username
        os.environ['KAGGLE_KEY'] = args.kaggle_key
        print(f"Kaggle API 인증 정보가 설정되었습니다. 사용자: {args.kaggle_username}")
    else:
        print("Kaggle API 인증 정보가 제공되지 않았습니다. 필요시 authenticate 도구를 사용하세요.")

    print("Kaggle MCP 서버 시작...")
    mcp.run(
        transport="stdio"
    ) 
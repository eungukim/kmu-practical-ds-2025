from mcp.server.fastmcp import FastMCP
import pandas as pd
import os
import tempfile
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import argparse
from pathlib import Path
from fastmcp.resources import FileResource
import shutil
import uuid

# 환경 변수에서 설정 가져오기
host = os.getenv("MCP_HOST", "0.0.0.0")
port = int(os.getenv("MCP_PORT", "8000"))
transport = os.getenv("MCP_TRANSPORT", "stdio")
kaggle_username = os.getenv("KAGGLE_USERNAME")
kaggle_key = os.getenv("KAGGLE_KEY")

# 파일 저장을 위한 임시 디렉토리 
TEMP_DATA_DIR = os.path.join(tempfile.gettempdir(), "kaggle-mcp-data")
os.makedirs(TEMP_DATA_DIR, exist_ok=True)

# KaggleAPI 인스턴스 초기화
api = KaggleApi()

# Kaggle API 인증 함수
def authenticate_kaggle_api():
    """Kaggle API 인증을 수행하고 인증 상태를 반환합니다"""
    try:
        # 환경 변수를 통한 인증 설정
        if kaggle_username and kaggle_key:
            os.environ["KAGGLE_USERNAME"] = kaggle_username
            os.environ["KAGGLE_KEY"] = kaggle_key
        
        api.authenticate()
        
        # 인증 상태 확인
        if not api.get_config_value('username') or not api.get_config_value('key'):
            return False, "Kaggle API 인증 정보가 설정되지 않았습니다. 환경 변수 또는 ~/.kaggle/kaggle.json 파일을 확인하세요."
        return True, "인증 성공"
    except Exception as e:
        return False, f"인증 실패: {str(e)}"

mcp = FastMCP(
    name="kaggle-mcp-server",
    instructions="Kaggle API를 활용한 데이터셋 조회, 다운로드 및 분석을 수행하는 MCP 서버",
    host=host,
    port=port,
    debug=True  # 디버그 모드 활성화
)

@mcp.tool('list_datasets', "Kaggle 데이터셋 목록 조회")
async def list_datasets(
    search_query: str = "",
    max_results: int = 10,
    sort_by: str = "hottest"  # 기본값을 유효한 값으로 변경
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
        auth_status, auth_message = authenticate_kaggle_api()
        if not auth_status:
            return {
                "success": False,
                "message": auth_message
            }
        
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
@mcp.tool('dataset_info', "Kaggle 데이터셋 상세 정보 조회")
async def dataset_info(
    dataset_ref: str  # owner/dataset-name 형식
) -> dict:
    """특정 Kaggle 데이터셋의 상세 정보를 조회합니다"""
    try:
        # Kaggle API 인증
        auth_status, auth_message = authenticate_kaggle_api()
        if not auth_status:
            return {
                "success": False,
                "message": auth_message
            }
        
        # 데이터셋 상세 정보 조회
        owner, dataset_name = dataset_ref.split('/')
        # owner 매개변수가 지원되지 않으므로 검색어로만 필터링
        datasets = api.dataset_list(search=dataset_name)
        
        # 일치하는 데이터셋 찾기
        dataset = None
        for ds in datasets:
            if ds.ref == dataset_ref:
                dataset = ds
                break
                
        if not dataset:
            return {
                "success": False,
                "message": f"데이터셋을 찾을 수 없습니다: {dataset_ref}"
            }
        
        # 파일 목록 가져오기
        files = api.dataset_list_files(dataset_ref).files
        file_list = []
        for f in files:
            file_info = {'name': f.name}
            if hasattr(f, 'size'):
                file_info['size'] = f.size
            file_list.append(file_info)
        
        # 데이터셋 정보 구성
        dataset_info = {
            "ref": dataset_ref,
            "title": dataset.title,
            "files": file_list
        }
        
        # 안전하게 속성 추가
        if hasattr(dataset, 'description'):
            dataset_info["description"] = dataset.description
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
        if hasattr(dataset, 'license'):
            dataset_info["license"] = dataset.license
        
        return {
            "success": True,
            "dataset": dataset_info
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"데이터셋 정보 조회 실패: {str(e)}"
        }

@mcp.tool('preview_dataset', "Kaggle 데이터셋 미리보기")
async def preview_dataset(
    dataset_ref: str,  # owner/dataset-name 형식
    file_name: str,
    rows: int = 10
) -> dict:
    """데이터셋의 특정 파일을 미리봅니다 (CSV 또는 다른 표 형식 파일을 지원)"""
    try:
        # Kaggle API 인증
        auth_status, auth_message = authenticate_kaggle_api()
        if not auth_status:
            return {
                "success": False,
                "message": auth_message
            }
        
        # 임시 디렉토리에 다운로드
        temp_dir = tempfile.mkdtemp()
        api.dataset_download_file(dataset_ref, file_name, path=temp_dir)
        
        # 파일 경로
        file_path = ""
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file == file_name or file == f"{file_name}.zip":
                    file_path = os.path.join(root, file)
                    break
        
        # 파일 확장자에 따라 처리
        if file_path.endswith('.zip'):
            import zipfile
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                file_path = os.path.join(temp_dir, file_name)
        
        # 파일 타입에 따른 처리
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_path)
            preview_data = df.head(rows).to_dict()
            columns = df.columns.tolist()
            dtypes = df.dtypes.astype(str).to_dict()
            shape = df.shape
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
            preview_data = df.head(rows).to_dict()
            columns = df.columns.tolist()
            dtypes = df.dtypes.astype(str).to_dict()
            shape = df.shape
        else:
            # 텍스트 파일로 가정하고 처리
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines()[:rows]]
            preview_data = {"lines": lines}
            columns = []
            dtypes = {}
            shape = (len(lines), 0)
        
        return {
            "success": True,
            "file_name": file_name,
            "preview": preview_data,
            "columns": columns if hasattr(locals(), 'columns') else [],
            "dtypes": dtypes if hasattr(locals(), 'dtypes') else {},
            "shape": shape if hasattr(locals(), 'shape') else (0, 0)
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"데이터셋 미리보기 실패: {str(e)}"
        }

@mcp.tool('list_competitions', "Kaggle 대회 목록 조회")
async def list_competitions(
    search_query: str = "",
    category: str = "all",
    max_results: int = 10
) -> dict:
    """Kaggle 대회 목록을 조회합니다"""
    try:
        # Kaggle API 인증
        auth_status, auth_message = authenticate_kaggle_api()
        if not auth_status:
            return {
                "success": False,
                "message": auth_message
            }
        
        # 대회 검색
        competitions = api.competitions_list(search=search_query, category=category, page_size=max_results)
        
        result = []
        for competition in competitions:
            result.append({
                "ref": competition.ref,
                "title": competition.title,
                "url": competition.url,
                "deadline": str(competition.deadline),
                "category": competition.category,
                "reward": competition.reward,
                "team_count": competition.teamCount
            })
        
        return {
            "success": True,
            "count": len(result),
            "competitions": result
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"대회 목록 조회 실패: {str(e)}"
        }

@mcp.tool('analyze_dataset', "다운로드된 CSV 데이터셋 분석")
async def analyze_dataset(
    file_path: str,
    delimiter: str = ",",
    sample_size: int = 5
) -> dict:
    """다운로드된 CSV 파일을 분석합니다"""
    try:
        # CSV 파일 로드
        df = pd.read_csv(file_path, delimiter=delimiter)
        
        # 기본 통계
        stats = df.describe(include='all').to_dict()
        
        # 결측값 정보
        missing_values = df.isnull().sum().to_dict()
        missing_percent = (df.isnull().mean() * 100).to_dict()
        
        # 데이터 타입 정보
        dtypes = df.dtypes.astype(str).to_dict()
        
        # 상관관계 (수치형 변수만)
        try:
            corr = df.corr(numeric_only=True).to_dict()
        except:
            corr = {"message": "상관관계를 계산할 수 없습니다."}
        
        return {
            "success": True,
            "file_info": {
                "path": file_path,
                "columns": df.columns.tolist(),
                "shape": df.shape,
                "sample": df.head(sample_size).to_dict(),
                "dtypes": dtypes
            },
            "statistics": stats,
            "missing_values": missing_values,
            "missing_percent": missing_percent,
            "correlation": corr
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"데이터셋 분석 실패: {str(e)}"
        }

@mcp.tool('download_dataset', "Kaggle 데이터셋을 다운로드하고 파일 URI 반환")
async def download_dataset(
    dataset_ref: str,  # owner/dataset-name 형식
    client_path: str,  # 클라이언트 측 저장 경로
    unzip: bool = True
) -> dict:
    """Kaggle 데이터셋을 다운로드하고 파일 리소스 URI를 반환합니다"""
    try:
        # Kaggle API 인증
        auth_status, auth_message = authenticate_kaggle_api()
        if not auth_status:
            return {
                "success": False,
                "message": auth_message
            }
        
        # 고유 ID로 데이터셋 저장 디렉토리 생성
        dataset_id = f"{dataset_ref.replace('/', '-')}-{uuid.uuid4().hex[:8]}"
        dataset_dir = os.path.join(TEMP_DATA_DIR, dataset_id)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Kaggle API로 데이터셋 다운로드
        api.dataset_download_files(dataset_ref, path=dataset_dir, unzip=unzip)
        
        # 다운로드된 파일 목록
        file_list = []
        
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, dataset_dir)
                file_size = os.path.getsize(file_path)
                
                # 파일 MIME 타입 추측
                import mimetypes
                mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
                
                # 클라이언트 측 파일 경로
                client_file_path = os.path.join(client_path, relative_path)
                
                # 리소스 URI 생성
                safe_path = relative_path.replace('/', '__')
                resource_uri = f"kaggle-data://{dataset_id}/{safe_path}"
                
                file_list.append({
                    "name": file,
                    "client_path": client_file_path,
                    "size": file_size,
                    "mime_type": mime_type,
                    "resource_uri": resource_uri,
                    "server_path": file_path
                })
        
        return {
            "success": True,
            "message": f"데이터셋 '{dataset_ref}'가 성공적으로 다운로드되었습니다. 파일은 리소스 URI를 통해 접근할 수 있습니다.",
            "files": file_list,
            "dataset_ref": dataset_ref,
            "dataset_id": dataset_id,
            "dataset_dir": dataset_dir,
            "client_base_path": client_path
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"데이터셋 다운로드 실패: {str(e)}"
        }

@mcp.tool('save_kaggle_file', "리소스 URI의 파일을 클라이언트에 저장")
async def save_kaggle_file(
    resource_uri: str,
    client_path: str,
    force_overwrite: bool = False
) -> dict:
    """리소스 URI로 식별되는 파일을 클라이언트에 저장합니다"""
    try:
        # URI 파싱
        if not resource_uri.startswith("kaggle-data://"):
            return {
                "success": False,
                "message": f"지원되지 않는 리소스 URI 형식입니다: {resource_uri}"
            }
        
        # URI에서 경로 추출
        parts = resource_uri.replace("kaggle-data://", "").split("/", 1)
        if len(parts) != 2:
            return {
                "success": False,
                "message": f"잘못된 리소스 URI 형식입니다: {resource_uri}"
            }
        
        dataset_id, file_path = parts
        
        # 파일 경로 복원
        file_path = file_path.replace('__', '/')
        server_path = os.path.join(TEMP_DATA_DIR, dataset_id, file_path)
        
        # 파일이 존재하는지 확인
        if not os.path.exists(server_path):
            return {
                "success": False,
                "message": f"서버에 파일이 존재하지 않습니다: {server_path}"
            }
        
        # 클라이언트 파일 경로의 디렉토리 확인 및 생성
        os.makedirs(os.path.dirname(client_path), exist_ok=True)
        
        # 파일이 이미 존재하는지 확인
        if os.path.exists(client_path) and not force_overwrite:
            return {
                "success": False,
                "message": f"파일이 이미 존재합니다: {client_path}. 덮어쓰려면 force_overwrite=True 옵션을 사용하세요."
            }
        
        # 파일 복사
        shutil.copy2(server_path, client_path)
        
        # 파일 크기 및 경로 정보 반환
        file_size = os.path.getsize(client_path)
        
        return {
            "success": True,
            "message": f"파일이 성공적으로 저장되었습니다: {client_path}",
            "file_path": client_path,
            "size": file_size,
            "resource_uri": resource_uri
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"파일 저장 실패: {str(e)}"
        }

@mcp.tool('save_all_kaggle_files', "데이터셋의 모든 파일을 클라이언트에 저장")
async def save_all_kaggle_files(
    dataset_id: str,
    files: list,
    client_base_path: str,
    force_overwrite: bool = False
) -> dict:
    """데이터셋의 모든 파일을 클라이언트에 저장합니다"""
    try:
        results = []
        success_count = 0
        failed_count = 0
        
        # 기본 디렉토리 생성
        os.makedirs(client_base_path, exist_ok=True)
        
        for file_info in files:
            resource_uri = file_info.get("resource_uri")
            client_path = file_info.get("client_path")
            
            if not resource_uri or not client_path:
                failed_count += 1
                results.append({
                    "success": False,
                    "message": "리소스 URI 또는 클라이언트 경로가 누락되었습니다",
                    "file_info": file_info
                })
                continue
            
            # 개별 파일 저장
            save_result = await save_kaggle_file(resource_uri, client_path, force_overwrite)
            results.append(save_result)
            
            if save_result["success"]:
                success_count += 1
            else:
                failed_count += 1
        
        return {
            "success": success_count > 0,
            "message": f"총 {len(files)}개 파일 중 {success_count}개 저장 성공, {failed_count}개 실패",
            "dataset_id": dataset_id,
            "results": results
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"파일 일괄 저장 실패: {str(e)}"
        }

@mcp.tool('get_file_content', "저장된 파일 내용 가져오기")
async def get_file_content(
    file_path: str,
    as_text: bool = False,
    encoding: str = "utf-8"
) -> dict:
    """다운로드한 파일의 내용을 가져옵니다"""
    try:
        # 파일이 존재하는지 확인
        if not os.path.exists(file_path):
            return {
                "success": False,
                "message": f"파일을 찾을 수 없습니다: {file_path}"
            }
        
        # 파일 읽기
        if as_text:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
                
            return {
                "success": True,
                "content": content,
                "mime_type": "text/plain",
                "encoding": encoding
            }
        else:
            # 바이너리 파일을 Base64로 인코딩
            import base64
            with open(file_path, 'rb') as f:
                content = base64.b64encode(f.read()).decode('ascii')
                
            # MIME 타입 추측
            import mimetypes
            mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
                
            return {
                "success": True,
                "content": content,
                "mime_type": mime_type,
                "encoding": "base64"
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"파일 읽기 실패: {str(e)}"
        }

# 건강 상태 체크 엔드포인트
@mcp.resource("health://check")
def health_check() -> dict:
    """서버 상태 확인"""
    auth_status, auth_message = authenticate_kaggle_api()
    return {
        "status": "healthy", 
        "version": "1.0.0",
        "kaggle_auth": {
            "status": "authenticated" if auth_status else "unauthenticated",
            "message": auth_message
        }
    }

# 파일 리소스 정의 - kaggle-data URI
@mcp.resource("kaggle-data://{dataset_id}/{file_path}")
async def serve_kaggle_data(dataset_id: str, file_path: str) -> bytes:
    """Kaggle 데이터셋 파일 리소스를 제공하는 엔드포인트"""
    try:
        # 파일 경로 복원
        file_path = file_path.replace('__', '/')
        server_path = os.path.join(TEMP_DATA_DIR, dataset_id, file_path)
        
        if os.path.exists(server_path):
            with open(server_path, 'rb') as f:
                return f.read()
        else:
            print(f"요청된 파일을 찾을 수 없습니다: {server_path}")
            return b""
    except Exception as e:
        print(f"파일 제공 오류: {str(e)}")
        return b""

@mcp.tool('clear_kaggle_data', "임시 저장된 Kaggle 데이터셋 파일 제거")
async def clear_kaggle_data(
    dataset_id: str = None
) -> dict:
    """임시 저장된 Kaggle 데이터셋 파일을 제거합니다"""
    try:
        removed_count = 0
        total_bytes_freed = 0
        
        # dataset_id가 주어진 경우 해당 데이터셋만 처리
        if dataset_id:
            dataset_dir = os.path.join(TEMP_DATA_DIR, dataset_id)
            if os.path.exists(dataset_dir):
                # 디렉토리 크기 계산
                for root, dirs, files in os.walk(dataset_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        total_bytes_freed += os.path.getsize(file_path)
                        removed_count += 1
                
                # 디렉토리 삭제
                shutil.rmtree(dataset_dir)
            else:
                return {
                    "success": False,
                    "message": f"데이터셋 디렉토리를 찾을 수 없습니다: {dataset_id}"
                }
        # 모든 Kaggle 데이터 제거
        else:
            # 모든 디렉토리 처리
            for item in os.listdir(TEMP_DATA_DIR):
                item_path = os.path.join(TEMP_DATA_DIR, item)
                if os.path.isdir(item_path):
                    # 디렉토리 크기 계산
                    for root, dirs, files in os.walk(item_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            total_bytes_freed += os.path.getsize(file_path)
                            removed_count += 1
                    
                    # 디렉토리 삭제
                    shutil.rmtree(item_path)
        
        return {
            "success": True,
            "message": f"{removed_count}개 파일이 디스크에서 제거되었습니다 (약 {total_bytes_freed / (1024*1024):.2f}MB 해제됨)",
            "removed_count": removed_count,
            "bytes_freed": total_bytes_freed
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"데이터 제거 실패: {str(e)}"
        }

@mcp.tool('list_kaggle_data', "임시 저장된 Kaggle 데이터셋 목록 조회")
async def list_kaggle_data(
    dataset_id: str = None
) -> dict:
    """임시 저장된 Kaggle 데이터셋 목록을 조회합니다"""
    try:
        result = []
        total_size = 0
        
        if dataset_id:
            # 특정 데이터셋만 조회
            dataset_dir = os.path.join(TEMP_DATA_DIR, dataset_id)
            if os.path.exists(dataset_dir):
                dataset_info = {
                    "dataset_id": dataset_id,
                    "files": [],
                    "size": 0
                }
                
                for root, dirs, files in os.walk(dataset_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, dataset_dir)
                        file_size = os.path.getsize(file_path)
                        dataset_info["size"] += file_size
                        total_size += file_size
                        
                        # 파일 정보 추가
                        safe_path = relative_path.replace('/', '__')
                        resource_uri = f"kaggle-data://{dataset_id}/{safe_path}"
                        
                        dataset_info["files"].append({
                            "name": file,
                            "path": relative_path,
                            "size": file_size,
                            "resource_uri": resource_uri
                        })
                
                result.append(dataset_info)
            else:
                return {
                    "success": False,
                    "message": f"데이터셋 디렉토리를 찾을 수 없습니다: {dataset_id}"
                }
        else:
            # 모든 데이터셋 조회
            for item in os.listdir(TEMP_DATA_DIR):
                item_path = os.path.join(TEMP_DATA_DIR, item)
                if os.path.isdir(item_path):
                    dataset_info = {
                        "dataset_id": item,
                        "files": [],
                        "size": 0
                    }
                    
                    for root, dirs, files in os.walk(item_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            relative_path = os.path.relpath(file_path, item_path)
                            file_size = os.path.getsize(file_path)
                            dataset_info["size"] += file_size
                            total_size += file_size
                            
                            # 파일 정보 추가 (최대 10개만)
                            if len(dataset_info["files"]) < 10:
                                safe_path = relative_path.replace('/', '__')
                                resource_uri = f"kaggle-data://{item}/{safe_path}"
                                
                                dataset_info["files"].append({
                                    "name": file,
                                    "path": relative_path,
                                    "size": file_size,
                                    "resource_uri": resource_uri
                                })
                    
                    dataset_info["file_count"] = len(dataset_info["files"])
                    result.append(dataset_info)
        
        return {
            "success": True,
            "count": len(result),
            "total_size_mb": total_size / (1024*1024),
            "datasets": result
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"데이터셋 목록 조회 실패: {str(e)}"
        }

if __name__ == "__main__":
    print("Kaggle MCP 서버 시작...")
    print(f"서버 주소: {host}:{port}, 전송 방식: {transport}")
    print(f"임시 데이터 디렉토리: {TEMP_DATA_DIR}")
    
    # 인증 상태 확인
    auth_status, auth_message = authenticate_kaggle_api()
    if auth_status:
        print(f"Kaggle 인증 상태: 성공 (사용자: {api.get_config_value('username')})")
    else:
        print(f"Kaggle 인증 상태: 실패 ({auth_message})")
    
    # transport만 설정
    mcp.run(transport=transport)
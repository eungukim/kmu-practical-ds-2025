import pandas as pd
import os
import tempfile
import json
from typing import List, Optional, Dict, Any
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import argparse

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="kaggle-mcp-server",
    instructions="Kaggle API를 활용한 데이터셋 조회, 다운로드 및 분석을 수행하는 MCP 서버"
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
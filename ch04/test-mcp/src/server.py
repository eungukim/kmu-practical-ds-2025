from mcp.server.fastmcp import FastMCP
import sys
import os
import json


# 환경 변수에서 설정 가져오기
host = os.getenv("MCP_HOST", "0.0.0.0")
port = int(os.getenv("MCP_PORT", "8000"))
transport = os.getenv("MCP_TRANSPORT", "stdio")

mcp = FastMCP(
    name="test-server",
    instructions="Test 서버",
    host=host,
    port=port,
    debug=True  # 디버그 모드 활성화
)

# 계산기 도구 추가
@mcp.tool()
def add(a: int, b: int) -> int:
    """두 숫자를 더합니다"""
    return a + b

# 에코 리소스 추가
@mcp.resource("echo://{message}")
def echo_resource(message: str) -> str:
    """메시지를 반향합니다"""
    return f"ECHO: {message}"

# 건강 상태 체크 엔드포인트
@mcp.resource("health://check")
def health_check() -> dict:
    """서버 상태 확인"""
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    print("서버 시작 중...")
    print(f"서버 주소: {host}:{port}, 전송 방식: {transport}")
    
    # transport만 설정
    mcp.run(transport=transport)
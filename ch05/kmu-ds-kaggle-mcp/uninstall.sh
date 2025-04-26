#!/bin/bash

# MCP 설정에서 KMU-DS-Kaggle 항목 삭제
CONFIG_NAME="KMU-DS-Kaggle"
MCP_JSON_PATH="$HOME/.cursor/mcp.json"

# 백업 생성
cp "$MCP_JSON_PATH" "$MCP_JSON_PATH.bak"

# JSON 파일에서 해당 항목 삭제
TMP_FILE=$(mktemp)
jq "del(.mcpServers[\"$CONFIG_NAME\"])" "$MCP_JSON_PATH" > "$TMP_FILE" && mv "$TMP_FILE" "$MCP_JSON_PATH"
echo "MCP 설정에서 $CONFIG_NAME 항목이 삭제되었습니다."

# 실행 중인 서버 프로세스 종료
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
SERVER_PATH="$SCRIPT_DIR/src/server.py"

echo "실행 중인 서버 프로세스를 종료합니다..."
pkill -f "$SERVER_PATH" || echo "실행 중인 서버 프로세스가 없습니다."

# 가상환경 삭제
echo "가상환경을 삭제합니다..."
rm -rf venv
echo "가상환경이 삭제되었습니다."

echo "제거가 완료되었습니다."

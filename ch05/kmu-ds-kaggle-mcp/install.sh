#!/bin/bash

# 가상환경 설치
python3 -m venv venv
echo "가상환경이 생성되었습니다."

# 가상환경에 requirements.txt 로 의존성 설치
source venv/bin/activate
pip install -r requirements.txt
echo "의존성 설치가 완료되었습니다."

# ~/.cursor/mcp.json 에 mcp 설정 추가
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PYTHON_PATH="$SCRIPT_DIR/venv/bin/python"
SERVER_PATH="$SCRIPT_DIR/src/server.py"

CONFIG_NAME="KMU-DS-Kaggle"
MCP_JSON_PATH="$HOME/.cursor/mcp.json"

# 백업 생성
cp "$MCP_JSON_PATH" "$MCP_JSON_PATH.bak"

# JSON 파일 업데이트
TMP_FILE=$(mktemp)
jq --arg name "$CONFIG_NAME" \
   --arg py "$PYTHON_PATH" \
   --arg server "$SERVER_PATH" \
   '.mcpServers[$name] = {"command": $py, "args": [$server]}' \
   "$MCP_JSON_PATH" > "$TMP_FILE" && mv "$TMP_FILE" "$MCP_JSON_PATH"

echo "MCP 설정이 추가되었습니다."




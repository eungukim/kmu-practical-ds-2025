import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = "https://api.openai.com/v1"

headers = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json",
    "OpenAI-Beta": "assistants=v2"
}

def list_assistants():
    print("사용 가능한 어시스턴트 목록을 조회합니다...")
    response = requests.get(
        f"{OPENAI_API_BASE}/assistants",
        headers=headers,
        params={"limit": 100}  # 최대 100개까지 조회
    )
    
    if response.status_code != 200:
        print(f"오류 발생: {response.status_code}")
        print(response.text)
        return
    
    assistants = response.json()
    
    if "data" not in assistants or len(assistants["data"]) == 0:
        print("사용 가능한 어시스턴트가 없습니다.")
        return
    
    print(f"\n총 {len(assistants['data'])} 개의 어시스턴트를 찾았습니다.\n")
    
    for idx, assistant in enumerate(assistants["data"], 1):
        print(f"[{idx}] ID: {assistant['id']}")
        print(f"    이름: {assistant.get('name', '이름 없음')}")
        print(f"    모델: {assistant.get('model', '모델 정보 없음')}")
        
        # 도구 정보 출력
        tools = assistant.get("tools", [])
        tool_types = [tool["type"] for tool in tools]
        print(f"    도구: {', '.join(tool_types) if tool_types else '없음'}")
        
        # 지침 정보 출력 (너무 길면 줄임)
        instructions = assistant.get("instructions", "")
        if instructions:
            if len(instructions) > 100:
                instructions = instructions[:100] + "..."
            print(f"    지침: {instructions}")
        
        print()  # 구분을 위한 빈 줄

def set_assistant_id():
    """어시스턴트 ID를 선택하여 .env 파일에 저장합니다."""
    list_assistants()
    
    # 사용자에게 ID 입력 요청
    assistant_id = input("\n사용할 어시스턴트 ID를 입력하세요: ")
    
    # ID 확인
    response = requests.get(
        f"{OPENAI_API_BASE}/assistants/{assistant_id}",
        headers=headers
    )
    
    if response.status_code != 200:
        print(f"오류: 입력한 ID '{assistant_id}'로 어시스턴트를 찾을 수 없습니다.")
        return
    
    assistant = response.json()
    print(f"선택한 어시스턴트: {assistant.get('name', '이름 없음')} (ID: {assistant['id']})")
    
    # .env 파일 업데이트
    env_path = ".env"
    env_content = ""
    
    # 기존 .env 파일이 있으면 읽기
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            env_lines = f.readlines()
        
        # ASSISTANT_ID 항목이 있는지 확인
        updated = False
        new_env_lines = []
        for line in env_lines:
            if line.startswith("ASSISTANT_ID="):
                new_env_lines.append(f"ASSISTANT_ID={assistant_id}\n")
                updated = True
            else:
                new_env_lines.append(line)
        
        # ASSISTANT_ID 항목이 없으면 추가
        if not updated:
            new_env_lines.append(f"ASSISTANT_ID={assistant_id}\n")
        
        env_content = "".join(new_env_lines)
    else:
        # .env 파일이 없으면 새로 생성
        env_content = f"ASSISTANT_ID={assistant_id}\n"
    
    # .env 파일 쓰기
    with open(env_path, "w") as f:
        f.write(env_content)
    
    print(f"ASSISTANT_ID가 .env 파일에 저장되었습니다: {assistant_id}")
    print("이제 스크립트를 다시 실행하면 선택한 어시스턴트를 사용할 수 있습니다.")

if __name__ == "__main__":
    choice = input("1: 어시스턴트 목록 조회\n2: 어시스턴트 ID 설정\n선택: ")
    
    if choice == "1":
        list_assistants()
    elif choice == "2":
        set_assistant_id()
    else:
        print("잘못된 선택입니다.") 
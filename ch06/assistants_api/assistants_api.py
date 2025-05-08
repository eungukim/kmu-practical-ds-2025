import requests, os, time
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# API 키와 기본 URL 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = "https://api.openai.com/v1"
ASSISTANT_ID = os.getenv("ASSISTANT_ID")  # 리뷰데이터분석 Assistant의 ID

# API 요청에 사용할 헤더 설정
headers = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json",
    "OpenAI-Beta": "assistants=v2"
}

# 1) CSV 파일 업로드 (최대 512 MB, purpose="assistants")
print("파일 업로드 중...")
with open("file.csv", "rb") as f:
    files = {"file": f}
    response = requests.post(
        f"{OPENAI_API_BASE}/files",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        data={"purpose": "assistants"},
        files=files
    )
file_data = response.json()
# 오류 확인
if "error" in file_data:
    print(f"파일 업로드 오류: {file_data['error']['message']}")
    exit(1)
file_id = file_data["id"]
print(f"파일 업로드 완료: {file_id}")

# 2) Thread 생성
print("Thread 생성 중...")
response = requests.post(
    f"{OPENAI_API_BASE}/threads",
    headers=headers
)
thread = response.json()
# 오류 확인
if "error" in thread:
    print(f"Thread 생성 오류: {thread['error']['message']}")
    exit(1)
thread_id = thread["id"]
print(f"Thread 생성 완료: {thread_id}")

# 3) 메시지 추가 (파일 첨부)
print("메시지 추가 중...")
response = requests.post(
    f"{OPENAI_API_BASE}/threads/{thread_id}/messages",
    headers=headers,
    json={
        "role": "user",
        "content": "이 리뷰 데이터의 주요 키워드 분석, 감성 분석 및 제품의 장단점을 분석해줘. 다음 분석을 수행해줘:\n1. 평점 분포 분석\n2. 가장 자주 언급되는 키워드 추출 및 빈도 시각화\n3. 긍정 리뷰와 부정 리뷰의 주요 특징 비교\n4. 제품의 주요 장점과 단점 요약",
        "attachments": [
            {
                "file_id": file_id,
                "tools": [{"type": "code_interpreter"}]
            }
        ]
    }
)
message = response.json()
# 오류 확인
if "error" in message:
    print(f"메시지 추가 오류: {message['error']['message']}")
    exit(1)
print(f"메시지 추가 완료: {message['id']}")

# 4) Run 실행 (폴링 방식으로 변경)
print("Run 실행 중...")
response = requests.post(
    f"{OPENAI_API_BASE}/threads/{thread_id}/runs",
    headers=headers,
    json={
        "assistant_id": ASSISTANT_ID,
        "additional_instructions": "이 데이터를 분석해줘.",
        "tool_choice": {
            "type": "code_interpreter"
        }
    }
)
run = response.json()
# 오류 확인
if "error" in run:
    print(f"Run 생성 오류: {run['error']['message']}")
    exit(1)
run_id = run["id"]
print(f"Run 생성 완료: {run_id}")

# 5) 폴링하여 완료 확인
print("처리 중...")
while True:
    # Run 상태 확인
    response = requests.get(
        f"{OPENAI_API_BASE}/threads/{thread_id}/runs/{run_id}",
        headers=headers
    )
    run_status = response.json()
    # 오류 확인
    if "error" in run_status:
        print(f"상태 확인 오류: {run_status['error']['message']}")
        break
        
    status = run_status.get("status")
    print(f"현재 상태: {status}")
    
    # 완료 또는 실패 상태 확인
    if status in ["completed", "failed", "expired"]:
        break
    time.sleep(2)  # 2초 대기 후 다시 확인

# 6) 최종 답변 출력
if status == "completed":
    print("\n분석 완료! 메시지를 가져오는 중...")
    # Thread의 모든 메시지 가져오기
    response = requests.get(
        f"{OPENAI_API_BASE}/threads/{thread_id}/messages",
        headers=headers
    )
    messages = response.json()
    # 오류 확인
    if "error" in messages:
        print(f"메시지 조회 오류: {messages['error']['message']}")
        exit(1)
        
    print("\n최종 답변:")
    # 어시스턴트의 응답 메시지 출력
    if "data" in messages and len(messages["data"]) > 0:
        for message_item in messages["data"]:
            if message_item["role"] == "assistant":
                for content in message_item["content"]:
                    if content["type"] == "text":
                        print(content["text"]["value"])
else:
    # 실패 시 오류 정보 출력
    print(f"\n처리 실패: {status}")
    if "last_error" in run_status:
        print(f"오류 내용: {run_status['last_error']}")

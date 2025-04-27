# OpenAI Assistants API 튜토리얼

이 튜토리얼은 OpenAI의 Assistants API를 사용하여 데이터 분석 작업을 자동화하는 방법을 설명합니다.

## 개요

Assistants API는 OpenAI의 강력한 AI 모델을 활용하여 복잡한 작업을 수행할 수 있는 인터페이스를 제공합니다. 이 API를 통해 다음과 같은 기능을 활용할 수 있습니다:

- 파일 업로드 및 분석
- 코드 인터프리터를 통한 데이터 처리
- 지속적인 대화 스레드 관리
- 함수 호출 기능

## 주요 구성 요소

1. **Assistant**: 특정 목적을 위해 구성된 AI 도우미
2. **Thread**: 사용자와 Assistant 간의 대화 기록
3. **Message**: Thread 내에서 주고받는 메시지
4. **Run**: Assistant가 Thread 내의 메시지를 처리하는 실행 단위
5. **Tool**: Assistant가 사용할 수 있는 도구 (코드 인터프리터, 함수 호출 등)

## 기본 사용 흐름

1. 파일 업로드 (CSV, 이미지 등)
2. Thread 생성
3. Thread에 메시지 추가
4. Assistant를 사용하여 Run 실행
5. Run 상태 확인 및 결과 조회

## 코드 예제

### 1. 환경 설정

```python
import requests, json, os, time, sseclient
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
```

### 2. 파일 업로드

```python
# CSV 파일 업로드 (최대 512 MB, purpose="assistants")
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
file_id = file_data["id"]
print(f"파일 업로드 완료: {file_id}")
```

### 3. Thread 생성

```python
# Thread 생성
print("Thread 생성 중...")
response = requests.post(
    f"{OPENAI_API_BASE}/threads",
    headers=headers
)
thread = response.json()
thread_id = thread["id"]
print(f"Thread 생성 완료: {thread_id}")
```

### 4. 메시지 추가

```python
# 메시지 추가 (파일 첨부)
print("메시지 추가 중...")
response = requests.post(
    f"{OPENAI_API_BASE}/threads/{thread_id}/messages",
    headers=headers,
    json={
        "role": "user",
        "content": "이 리뷰 데이터의 주요 키워드 분석, 감성 분석 및 제품의 장단점을 분석해줘.",
        "attachments": [
            {
                "file_id": file_id,
                "tools": [{"type": "code_interpreter"}]
            }
        ]
    }
)
message = response.json()
print(f"메시지 추가 완료: {message['id']}")
```

### 5. Run 실행

```python
# Run 실행
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
run_id = run["id"]
print(f"Run 생성 완료: {run_id}")
```

### 6. Run 상태 확인

```python
# 폴링하여 완료 확인
print("처리 중...")
while True:
    # Run 상태 확인
    response = requests.get(
        f"{OPENAI_API_BASE}/threads/{thread_id}/runs/{run_id}",
        headers=headers
    )
    run_status = response.json()
    status = run_status.get("status")
    print(f"현재 상태: {status}")
    
    # 완료 또는 실패 상태 확인
    if status in ["completed", "failed", "expired"]:
        break
    time.sleep(2)  # 2초 대기 후 다시 확인
```

### 7. 결과 조회

```python
# 최종 답변 출력
if status == "completed":
    print("\n분석 완료! 메시지를 가져오는 중...")
    # Thread의 모든 메시지 가져오기
    response = requests.get(
        f"{OPENAI_API_BASE}/threads/{thread_id}/messages",
        headers=headers
    )
    messages = response.json()
    
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
```

## 오류 처리

각 API 호출에서 발생할 수 있는 오류를 다음과 같이 처리할 수 있습니다:

```python
# 응답에서 오류 확인
if "error" in response_data:
    print(f"오류 발생: {response_data['error']['message']}")
    exit(1)
```

## 주의사항

1. 파일 업로드 크기는 최대 512 MB로 제한됩니다.
2. 대화 스레드는 자동으로 만료되지 않으므로 필요한 경우 직접 삭제해야 합니다.
3. 코드 인터프리터 도구는 복잡한 데이터 분석 작업을 수행할 수 있지만 실행 시간에 제한이 있습니다.

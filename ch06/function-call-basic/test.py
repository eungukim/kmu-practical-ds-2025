import openai
import json
from openai import OpenAI

# 1. 함수 정의
def get_current_weather(location, unit="섭씨"):
    """예시 함수: 실제 API 대신 모의 데이터 반환"""
    return json.dumps({"location": location, "temperature": "22", "unit": unit})

# 2. 함수 스키마 설정
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "특정 위치의 현재 날씨 조회",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "도시명"},
                    "unit": {"type": "string", "enum": ["섭씨", "화씨"]}
                },
                "required": ["location"]
            }
        }
    }
]

# 3. 초기 메시지 설정
messages = [{"role": "user", "content": "서울 날씨 알려줘"}]

# 4. API 호출
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

response_message = response.choices[0].message

# 5. 함수 호출 감지 시 실행
if response_message.tool_calls:
    tool_call = response_message.tool_calls[0]
    function_name = tool_call.function.name
    
    # 매개변수 추출
    function_args = json.loads(tool_call.function.arguments)
    
    # 실제 함수 실행
    function_response = get_current_weather(
        location=function_args.get("location"),
        unit=function_args.get("unit")
    )
    
    # 6. 결과 재전송
    messages.append(response_message.model_dump())
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": function_name,
        "content": function_response
    })

    second_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    print(second_response.choices[0].message.content)
else:
    print(response_message.content)

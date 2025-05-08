# agent.py
import openai, json
from tools import describe_dataset, correlation_matrix, DescribeArgs, CorrelationArgs
from utils import TOOLS

MODEL = "gpt-4o-mini"  # 또는 gpt-4.1-mini
client = openai.OpenAI()

FUNC_MAP = {
    "describe": (describe_dataset, DescribeArgs),
    "correlation": (correlation_matrix, CorrelationArgs),
}

def call_agent(user_prompt: str):
    messages = [
        {"role": "system",
         "content": "You are a data-analysis assistant. "
                    "Speak in Korean. "
                    "When a tool is helpful, respond *ONLY* with tool_calls."},
        {"role": "user", "content": user_prompt},
        
    ]

    while True:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
        )
        choice = resp.choices[0]
        # 1) assistant 메시지(tool_calls 포함) 저장
        messages.append(choice.message)
        # 2) 완료되었으면 리턴
        if choice.finish_reason == "stop":
            return choice.message.content
        # 3) tool 호출 결과를 name 필드와 함께 저장
        for tc in choice.message.tool_calls:
            fn, Validator = FUNC_MAP[tc.function.name]
            args = Validator(**json.loads(tc.function.arguments))
            out = fn(**args.model_dump())
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,             # 👈 호출 ID
                "name": tc.function.name,          # 👈 함수 이름
                "content": json.dumps(out),
            })

if __name__ == "__main__":
    prompt = (
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        "파일을 가져와서 먼저 전체 기술통계를 보여 주고, "
        "sepal_length·petal_length·petal_width 3개 컬럼의 상관계수를 분석해줘"
    )
    answer = call_agent(prompt)
    print(answer)

import streamlit as st
import openai
import json
import pandas as pd
import io
from tools import describe_dataset, correlation_matrix, DescribeArgs, CorrelationArgs
from utils import TOOLS, pydantic_to_openai_schema

# 설정
MODEL = "gpt-4o-mini"
client = openai.OpenAI()

# 도구 함수 맵핑
FUNC_MAP = {
    "describe": (describe_dataset, DescribeArgs),
    "correlation": (correlation_matrix, CorrelationArgs),
}

# 페이지 설정
st.set_page_config(page_title="데이터 분석 에이전트", page_icon="📊")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", 
         "content": "You are a data-analysis assistant. "
                    "When a tool is helpful, respond *ONLY* with tool_calls. "
                    "If user uploads a file, use 'df_key' instead of 'csv_url' in your tool calls."}
    ]

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "df" not in st.session_state:
    st.session_state.df = None

# 제목 및 설명
st.title("📊 데이터 분석 에이전트")
st.markdown("CSV 데이터에 대한 분석을 요청하세요.")

# 사이드바에 파일 업로드 기능 추가
with st.sidebar:
    st.header("데이터 업로드")
    uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])
    
    if uploaded_file is not None:
        # 파일을 DataFrame으로 변환
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_file = uploaded_file.name
            st.session_state.df = df
            
            # 데이터 미리보기 표시
            st.success(f"'{uploaded_file.name}' 파일이 업로드되었습니다!")
            st.subheader("데이터 미리보기")
            st.dataframe(df.head())
            
            # 컬럼 정보 표시
            st.subheader("컬럼 정보")
            st.write(f"컬럼 수: {len(df.columns)}")
            st.write("컬럼 목록:")
            st.write(", ".join(df.columns.tolist()))
        except Exception as e:
            st.error(f"파일 로딩 중 오류가 발생했습니다: {str(e)}")

# 이전 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 채팅 함수 수정
def call_agent(user_input):
    # 사용자 메시지 추가
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # 응답 처리
    with st.spinner("분석 중..."):
        while True:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=st.session_state.chat_history,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0,
            )
            choice = resp.choices[0]
            
            # 메시지 저장
            st.session_state.chat_history.append(choice.message)
            
            # 완료 확인
            if choice.finish_reason == "stop":
                # 분석 결과 반환
                return choice.message.content
            
            # 도구 호출 처리
            for tc in choice.message.tool_calls:
                # 도구 호출 정보 표시
                tool_name = tc.function.name
                
                # 함수 인자 준비
                args = json.loads(tc.function.arguments)
                
                # 세션에 업로드된 파일이 있는 경우, df_key 인자 추가
                if st.session_state.df is not None and not args.get('csv_url'):
                    args['df_key'] = 'df'
                
                # 인자 검증
                validator_cls = FUNC_MAP[tool_name][1]
                validated_args = validator_cls(**args)
                
                # 함수 실행
                fn = FUNC_MAP[tool_name][0]
                
                # df_key가 있으면 세션 상태에서 DataFrame 가져오기
                if args.get('df_key') == 'df' and st.session_state.df is not None:
                    out = fn(**validated_args.model_dump(), df=st.session_state.df)
                else:
                    out = fn(**validated_args.model_dump())
                
                # 결과 저장
                st.session_state.chat_history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tool_name,
                    "content": json.dumps(out),
                })

# 사용 예시 표시
st.markdown("### 사용 예시")
if st.session_state.uploaded_file:
    st.markdown(f"- '{st.session_state.uploaded_file}' 파일의 통계 분석을 보여줘")
    st.markdown(f"- '{st.session_state.uploaded_file}' 파일에서 처음 3개 컬럼의 상관관계를 분석해줘")
else:
    st.markdown("- https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv 파일의 통계를 보여줘")
    st.markdown("- https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv 파일에서 age, fare, survived 컬럼의 상관관계를 분석해줘")

# 사용자 입력
user_input = st.chat_input("데이터 분석 질문을 입력하세요...")

if user_input:
    # 사용자 메시지 표시
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # AI 응답 생성
    with st.chat_message("assistant"):
        response = call_agent(user_input)
        st.markdown(response)
    
    # 응답 저장
    st.session_state.messages.append({"role": "assistant", "content": response}) 
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List, Union, Dict, Any

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.tools import PythonAstREPLTool
from langchain_openai import ChatOpenAI

# --- 세션 상태 및 상수 초기화 ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

class MessageRole:
    USER = "user"
    ASSISTANT = "assistant"

class MessageType:
    TEXT = "text"
    FIGURE = "figure"
    CODE = "code"
    DATAFRAME = "dataframe"

# --- 사이드바 설정 ---
with st.sidebar:
    st.markdown("### 🔑 OpenAI API 키")
    user_api_key = st.text_input("OpenAI API Key", type="password", help="API 키를 입력하세요.")

    st.markdown("### 📄 CSV 파일 업로드")
    uploaded_file = st.file_uploader(
        "분석할 CSV 파일을 업로드 해주세요.", type=["csv"]
    )

    st.markdown("---")
    apply_btn = st.button("✓ 데이터 분석 시작", use_container_width=True)
    clear_btn = st.button("↻ 대화 초기화", use_container_width=True)

# --- 메인 화면 ---
st.title(" CSV 데이터 분석 챗봇 💬")
st.markdown("`pandas-ai`와 `LangChain` 에이전트를 사용하여 CSV 파일의 데이터를 분석하고 시각화합니다.")

def print_messages():
    """세션에 저장된 메시지를 순서대로 출력합니다."""
    for role, content_list in st.session_state["messages"]:
        with st.chat_message(role):
            for content_type, content_value in content_list:
                if content_type == MessageType.TEXT:
                    st.markdown(content_value)
                elif content_type == MessageType.FIGURE:
                    st.pyplot(content_value)
                elif content_type == MessageType.CODE:
                    with st.expander("실행된 코드 보기", expanded=False):
                        st.code(content_value, language="python")
                elif content_type == MessageType.DATAFRAME:
                    st.dataframe(content_value)

def add_message(role: str, content: List[Union[str, Any]]):
    """메시지를 세션 상태에 추가합니다."""
    # content는 [type, value] 형태의 리스트
    st.session_state.messages.append([role, [content]])


def create_agent(dataframe: pd.DataFrame, api_key: str):
    """Pandas DataFrame을 다루는 LangChain 에이전트를 생성합니다."""
    try:
        llm = ChatOpenAI(
            model="gpt-4-turbo", # 필요시 모델 변경 (e.g., gpt-4, gpt-4.1-mini)
            temperature=0,
            api_key=api_key
        )
        tool = PythonAstREPLTool(locals={"df": dataframe})
        return create_pandas_dataframe_agent(
            llm=llm,
            df=dataframe,
            agent_executor_kwargs={"handle_parsing_errors": True},
            verbose=False,
            agent_type="tool-calling",
            allow_dangerous_code=True,
            prefix=(
                "You are a professional data analyst and expert in Pandas. "
                "You must use the Pandas DataFrame `df` to answer the user's request. "
                "\n\n[IMPORTANT] DO NOT create or overwrite the `df` variable in your code. \n\n"
                "If you generate visualization code, please use `plt.show()` at the end. "
                "I prefer seaborn for visualization, but matplotlib is also fine."
                "\n\n<Visualization Preference>\n"
                "- [IMPORTANT] Use `English` for your visualization title and labels."
                "- Use a `muted` color palette, white background, and no grid."
                "\nRecommend setting cmap or palette for seaborn plots. "
                "The final answer should be in Korean."
                "\n\n###\n\n<Column Guidelines>\n"
                "If the user asks about columns not in `df.columns`, you may refer to the most similar columns listed."
            ),
            tools=[tool]
        )
    except Exception as e:
        st.error(f"에이전트 생성 중 오류가 발생했습니다: {e}")
        return None

# --- 버튼 로직 처리 ---
if clear_btn:
    st.session_state["messages"] = []
    st.rerun()

if apply_btn:
    if not user_api_key:
        st.warning("사이드바에 OpenAI API 키를 입력해주세요.")
    elif not uploaded_file:
        st.warning("분석할 CSV 파일을 업로드해주세요.")
    else:
        with st.spinner("데이터를 설정하는 중입니다..."):
            loaded_data = pd.read_csv(uploaded_file)
            st.session_state["df"] = loaded_data
            st.session_state["agent"] = create_agent(loaded_data, user_api_key)
        if st.session_state["agent"]:
            st.success("설정이 완료되었습니다. 이제 질문을 시작하세요!")
            st.dataframe(loaded_data.head())


# 저장된 메시지 출력
print_messages()

# --- 사용자 입력 및 에이전트 실행 ---
if user_input := st.chat_input("데이터에 대해 궁금한 점을 질문하세요!"):
    if "agent" not in st.session_state:
        st.warning("먼저 사이드바에서 API 키와 CSV 파일을 설정하고 '데이터 분석 시작' 버튼을 눌러주세요.")
    else:
        # 사용자 질문을 메시지에 추가하고 화면에 표시
        add_message(MessageRole.USER, [MessageType.TEXT, user_input])
        st.chat_message(MessageRole.USER).write(user_input)

        with st.chat_message(MessageRole.ASSISTANT):
            # 스트리밍 출력을 처리할 컨테이너
            response_container = st.container()
            final_answer = ""

            # 에이전트 스트리밍 실행
            response_stream = st.session_state.agent.stream({"input": user_input})

            for chunk in response_stream:
                # 파이썬 코드 실행 블록 (Tool 사용)
                if "actions" in chunk:
                    for action in chunk["actions"]:
                        tool_input = action.tool_input
                        if isinstance(tool_input, dict) and "query" in tool_input:
                            code = tool_input["query"]
                            with st.status("코드 실행 중...", expanded=True) as status:
                                status.write("생성된 코드를 실행하고 있습니다.")
                                st.code(code, language="python")
                                # 실행된 코드를 메시지에 추가
                                add_message(MessageRole.ASSISTANT, [MessageType.CODE, code])
                                status.update(label="실행 완료!", state="complete")

                # 코드 실행 결과 (Observation)
                elif "steps" in chunk:
                    for step in chunk["steps"]:
                        observation = step.observation
                        if "Error" in str(observation):
                            st.error(f"코드 실행 중 오류가 발생했습니다:\n{observation}")
                            add_message(MessageRole.ASSISTANT, [MessageType.TEXT, f"**오류 발생**:\n```\n{observation}\n```"])
                        # 시각화 결과 처리
                        if plt.get_fignums():
                            fig = plt.gcf()
                            st.pyplot(fig)
                            add_message(MessageRole.ASSISTANT, [MessageType.FIGURE, fig])
                            plt.clf() # 다음 시각화를 위해 현재 figure를 초기화

                # 최종 답변
                elif "output" in chunk:
                    final_answer += chunk["output"]
                    response_container.markdown(final_answer)

            # 최종 답변을 메시지에 저장
            if final_answer:
                 st.session_state.messages.append([MessageRole.ASSISTANT, [[MessageType.TEXT, final_answer]]])


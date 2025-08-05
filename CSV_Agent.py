from typing import List, Union
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.tools import PythonAstREPLTool
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging
from langchain_teddynote.messages import AgentStreamParser, AgentCallbacks
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# TeddyNote의 langsmith 로깅
logging.langsmith("CSV Agent 챗봇")

# Streamlit 앱 설정
st.title("CSV 데이터 분석 챗봇 💬")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 상수 정의
class MessageRole:
    USER = "user"
    ASSISTANT = "assistant"

class MessageType:
    TEXT = "text"
    FIGURE = "figure"
    CODE = "code"
    DATAFRAME = "dataframe"

def print_messages():
    for role, content_list in st.session_state["messages"]:
        with st.chat_message(role):
            for content in content_list:
                if isinstance(content, list):
                    message_type, message_content = content
                    if message_type == MessageType.TEXT:
                        st.markdown(message_content)
                    elif message_type == MessageType.FIGURE:
                        st.pyplot(message_content)
                    elif message_type == MessageType.CODE:
                        with st.status("코드 출력", expanded=False):
                            st.code(message_content, language="python")
                    elif message_type == MessageType.DATAFRAME:
                        st.dataframe(message_content)
                else:
                    raise ValueError(f"알 수 없는 콘텐츠 유형: {content}")

def add_message(role: MessageRole, content: List[Union[MessageType, str]]):
    messages = st.session_state["messages"]
    if messages and messages[-1][0] == role:
        messages[-1][1].extend([content])
    else:
        messages.append([role, [content]])

# 사이드바 설정
with st.sidebar:
    st.markdown("🔑 **OpenAI API 키를 입력하세요**")
    user_api_key = st.text_input("OpenAI API Key", type="password")
    clear_btn = st.button("대화 초기화")
    uploaded_file = st.file_uploader(
        "CSV 파일을 업로드 해주세요.", type=["csv"], accept_multiple_files=False
    )
    selected_model = st.selectbox(
        "OpenAI 모델을 선택해주세요.", ["gpt-4.1-mini", "gpt-4.1-nano"], index=0
    )
    apply_btn = st.button("데이터 분석 시작")

# API Key 처리
if user_api_key:
    # 환경변수에 OPENAI_API_KEY 설정
    os.environ["OPENAI_API_KEY"] = user_api_key

def tool_callback(tool) -> None:
    if tool_name := tool.get("tool"):
        if tool_name == "python_repl_ast":
            tool_input = tool.get("tool_input", {})
            query = tool_input.get("query")
            if query:
                df_in_result = None
                with st.status("데이터 분석 중...", expanded=True) as status:
                    st.markdown(f"``````")
                    add_message(MessageRole.ASSISTANT, [MessageType.CODE, query])
                    if "df" in st.session_state:
                        result = st.session_state["python_tool"].invoke({"query": query})
                        if isinstance(result, pd.DataFrame):
                            df_in_result = result
                    status.update(label="코드 출력", state="complete", expanded=False)
                if df_in_result is not None:
                    st.dataframe(df_in_result)
                    add_message(MessageRole.ASSISTANT, [MessageType.DATAFRAME, df_in_result])
                if "plt.show" in query:
                    fig = plt.gcf()
                    st.pyplot(fig)
                    add_message(MessageRole.ASSISTANT, [MessageType.FIGURE, fig])
                return result
            else:
                st.error("데이터프레임이 정의되지 않았습니다. CSV 파일을 먼저 업로드해주세요.")
                return

def observation_callback(observation) -> None:
    if "observation" in observation:
        obs = observation["observation"]
        if isinstance(obs, str) and "Error" in obs:
            st.error(obs)
            st.session_state["messages"][-1][1].clear()

def result_callback(result: str) -> None:
    pass

def create_agent(dataframe, selected_model="gpt-4.1-mini"):
    # OpenAI API 키가 입력되지 않은 경우 오류 표시
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        st.error("OpenAI API 키를 입력해주세요 (사이드바)")
        return None
    return create_pandas_dataframe_agent(
        ChatOpenAI(model=selected_model, temperature=0, api_key=openai_key),
        dataframe,
        verbose=False,
        agent_type="tool-calling",
        allow_dangerous_code=True,
        prefix=(
            "You are a professional data analyst and expert in Pandas. "
            "You must use Pandas DataFrame(`df`) to answer user's request. "
            "\n\n[IMPORTANT] DO NOT create or overwrite the `df` variable in your code. \n\n"
            "If you are willing to generate visualization code, please use `plt.show()` at the end of your code. "
            "I prefer seaborn code for visualization, but you can use matplotlib as well."
            "\n\n<Visualization Preference>\n"
            "- [IMPORTANT] Use `English` for your visualization title and labels."
            "- `muted` cmap, white background, and no grid for your visualization."
            "\nRecommend to set cmap, palette parameter for seaborn plot if it is applicable. "
            "The language of final answer should be written in Korean. "
            "\n\n###\n\n<Column Guidelines>\n"
            "If user asks with columns that are not listed in `df.columns`, you may refer to the most similar columns listed below.\n"
        ),
    )

def ask(query):
    if "agent" in st.session_state:
        st.chat_message("user").write(query)
        add_message(MessageRole.USER, [MessageType.TEXT, query])

        agent = st.session_state["agent"]
        response = agent.stream({"input": query})

        ai_answer = ""
        parser_callback = AgentCallbacks(tool_callback, observation_callback, result_callback)
        stream_parser = AgentStreamParser(parser_callback)

        with st.chat_message("assistant"):
            for step in response:
                stream_parser.process_agent_steps(step)
                if "output" in step:
                    ai_answer += step["output"]
            st.write(ai_answer)
        add_message(MessageRole.ASSISTANT, [MessageType.TEXT, ai_answer])

if clear_btn:
    st.session_state["messages"] = []

if apply_btn:
    if not user_api_key:
        st.warning("OpenAI API 키를 입력해주세요.")
    elif not uploaded_file:
        st.warning("파일을 업로드 해주세요.")
    else:
        loaded_data = pd.read_csv(uploaded_file)
        st.session_state["df"] = loaded_data
        st.session_state["python_tool"] = PythonAstREPLTool()
        st.session_state["python_tool"].locals["df"] = loaded_data
        st.session_state["agent"] = create_agent(loaded_data, selected_model)
        st.success("설정이 완료되었습니다. 대화를 시작해 주세요!")

print_messages()

user_input = st.chat_input("궁금한 내용을 물어보세요!")
if user_input:
    ask(user_input)


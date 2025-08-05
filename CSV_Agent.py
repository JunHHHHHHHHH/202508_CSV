from typing import List, Union
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.tools import PythonAstREPLTool
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging
from langchain_teddynote.messages import AgentStreamParser, AgentCallbacks

# TeddyNote 로그
logging.langsmith("CSV Agent 챗봇")

# Streamlit 앱 제목
st.title("CSV 데이터 분석 챗봇 💬")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "df_list" not in st.session_state:
    st.session_state["df_list"] = []
if "file_names" not in st.session_state:
    st.session_state["file_names"] = []
if "agent" not in st.session_state:
    st.session_state["agent"] = None
if "python_tool" not in st.session_state:
    st.session_state["python_tool"] = None

# 상수 선언

class MessageRole:
    USER = "user"
    ASSISTANT = "assistant"

class MessageType:
    TEXT = "text"
    FIGURE = "figure"
    CODE = "code"
    DATAFRAME = "dataframe"

# 메시지 출력 함수
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
                        with st.expander("코드 보기"):
                            st.code(message_content, language="python")
                    elif message_type == MessageType.DATAFRAME:
                        st.dataframe(message_content)
                    else:
                        raise ValueError(f"알 수 없는 콘텐츠 유형: {content}")

# 메시지 추가 함수
def add_message(role: MessageRole, content: List[Union[MessageType, str]]):
    messages = st.session_state["messages"]
    if messages and messages[-1][0] == role:
        messages[-1][1].extend([content])
    else:
        messages.append([role, [content]])

# API 키 입력 및 파일 업로드 UI
with st.sidebar:
    st.markdown("🔑 **OpenAI API 키를 입력하세요**")
    user_api_key = st.text_input("OpenAI API Key", type="password")
    clear_btn = st.button("대화 초기화")
    uploaded_files = st.file_uploader("CSV 파일들을 업로드 해주세요 (다중 업로드 지원)", type=["csv"], accept_multiple_files=True)
    pre_process_option = st.checkbox("결측치 제거", value=False)
    outlier_detect_option = st.checkbox("이상치 자동 탐지", value=False)
    apply_btn = st.button("데이터 분석 시작")

# API 키 환경 변수 설정
if user_api_key:
    os.environ["OPENAI_API_KEY"] = user_api_key

# 초기화 버튼
if clear_btn:
    st.session_state["messages"] = []
    st.session_state["df_list"] = []
    st.session_state["file_names"] = []
    st.session_state["agent"] = None
    st.session_state["python_tool"] = None
    st.experimental_rerun()

# 전처리: 결측치 처리 및 이상치 탐지
def preprocess_df(df, drop_na=False, outlier_detect=False):
    info_msgs = []
    # 결측치 제거
    if drop_na:
        before_shape = df.shape
        df = df.dropna()
        info_msgs.append(f"결측치 제거 수행. {before_shape} → {df.shape}")
    # 이상치 탐지 - z-score 기반
    if outlier_detect:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            info_msgs.append("숫자형 컬럼이 없어 이상치 탐지를 수행할 수 없습니다.")
            return df, info_msgs
        from scipy.stats import zscore
        z_scores = np.abs(zscore(df[numeric_cols], nan_policy='omit'))
        threshold = 3
        outliers = (z_scores > threshold).any(axis=1)
        num_outliers = outliers.sum()
        info_msgs.append(f"이상치로 감지된 행 수: {num_outliers}")
        df_clean = df.loc[~outliers]
        info_msgs.append(f"이상치 제거 후 데이터 크기: {df_clean.shape}")
        return df_clean, info_msgs
    return df, info_msgs

# 자동 데이터 개요 출력
def auto_data_overview(df, name=None):
    st.subheader(f"{name or '데이터'} 기본 개요 및 통계")
    st.write(f"행렬 크기: {df.shape}")
    st.write("데이터 타입:")
    st.write(df.dtypes)
    st.write("결측치 개수:")
    st.write(df.isnull().sum())
    st.write("기본 기술통계:")
    st.write(df.describe(include='all'))
    st.markdown("---")

# 자동 변수별 시각화 추천 및 출력 (Plotly)
def auto_visualization(df, name="데이터"):
    st.subheader(f"{name} 자동 시각화")
    for col in df.columns:
        # 수치형
        if pd.api.types.is_numeric_dtype(df[col]):
            fig = px.histogram(df, x=col, nbins=30, title=f"{col} 분포 (Histogram)")
        # 날짜형
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            # 숫자형 숫자 컬럼 선택 (최초 발견)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                y_col = numeric_cols[0]
                fig = px.line(df.sort_values(by=col), x=col, y=y_col, title=f"{col} vs {y_col} (Line Plot)")
            else:
                continue
        # 범주형
        else:
            counts = df[col].value_counts().reset_index()
            counts.columns = [col, "count"]
            fig = px.bar(counts, x=col, y="count", title=f"{col} 빈도 (Bar Chart)")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

# 다중 파일 비교 간단 예시: shape, 컬럼명 차이 시각화
def multi_file_summary(df_list, file_names):
    st.subheader("업로드된 데이터 파일 요약 및 비교")
    summary = []
    for i, df in enumerate(df_list):
        summary.append({
            "파일명": file_names[i],
            "행(row)": df.shape[0],
            "열(column)": df.shape[1],
            "컬럼명": ", ".join(df.columns)
        })
    df_summary = pd.DataFrame(summary)
    st.table(df_summary)

    # 컬럼명 비교
    all_columns = [set(df.columns) for df in df_list]
    common_cols = set.intersection(*all_columns)
    unique_cols = [set(df.columns) - common_cols for df in df_list]
    
    st.write(f"공통 컬럼: {sorted(common_cols)}")
    for i, uc in enumerate(unique_cols):
        st.write(f"{file_names[i]} 고유 컬럼: {sorted(uc)}")
    st.markdown("---")

# Langchain Agent 생성
def create_agent(dataframe, selected_model="gpt-4.1-mini"):
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
            "If you are willing to generate visualization code, you must use fig, ax = plt.subplots() and st.pyplot(fig) to show the figure in Streamlit. "
            "Prefer seaborn code for visualization, but matplotlib is also allowed."
            "\n\n\n"
            "- [IMPORTANT] Use `English` for your visualization title and labels."
            "- Please use palette='muted' for seaborn (not cmap), and for matplotlib use a valid colormap (for example, 'viridis')."
            "- White background, and no grid for your visualization."
            "\nRecommend to set cmap, palette parameter for seaborn plot if applicable. "
            "The language of final answer should be Korean."
            "\n\n###\n\n\n"
            "If user asks with columns that are not listed in `df.columns`, you may refer to the most similar columns listed below.\n"
        ),
    )

# Agent용 콜백 함수 (기존 동일)
def tool_callback(tool) -> None:
    if tool_name := tool.get("tool"):
        if tool_name == "python_repl_ast":
            tool_input = tool.get("tool_input", {})
            query = tool_input.get("query")
            if query:
                df_in_result = None
                with st.status("데이터 분석 중...", expanded=True) as status:
                    add_message(MessageRole.ASSISTANT, [MessageType.CODE, query])
                    if "df" in st.session_state:
                        sns.set_theme(style="white")
                        st.session_state["python_tool"].locals["pd"] = pd
                        st.session_state["python_tool"].locals["sns"] = sns
                        st.session_state["python_tool"].locals["plt"] = plt
                        try:
                            result = st.session_state["python_tool"].invoke({"query": query})
                        except Exception as e:
                            st.error(f"오류 발생: {e}")
                            return
                        if isinstance(result, pd.DataFrame):
                            df_in_result = result
                        status.update(label="코드 출력", state="complete", expanded=False)
                        if df_in_result is not None:
                            st.dataframe(df_in_result)
                            add_message(MessageRole.ASSISTANT, [MessageType.DATAFRAME, df_in_result])
                        if "plt.show" in query or "st.pyplot" in query:
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

# 사용자 질문 처리 함수
def ask(query):
    if "agent" in st.session_state and st.session_state["agent"]:
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

# 업로드 후 처리 및 분석 시작
if apply_btn:
    if not user_api_key:
        st.warning("OpenAI API 키를 입력해주세요.")
    elif not uploaded_files or len(uploaded_files) == 0:
        st.warning("파일을 업로드 해주세요.")
    else:
        # 여러 파일 처리
        dfs = []
        names = []
        preprocess_infos = []
        for file in uploaded_files:
            df_load = pd.read_csv(file)
            df_processed, info_msgs = preprocess_df(df_load, drop_na=pre_process_option, outlier_detect=outlier_detect_option)
            dfs.append(df_processed)
            names.append(file.name)
            preprocess_infos.append((file.name, info_msgs))

        st.session_state["df_list"] = dfs
        st.session_state["file_names"] = names
        
        # 기본 개요 & 전처리 안내
        for name, msgs in preprocess_infos:
            if msgs:
                st.write(f"**[{name}] 전처리 정보:**")
                for msg in msgs:
                    st.write("- " + msg)

        # 다중 파일 요약 및 비교
        multi_file_summary(dfs, names)

        # 각 파일별 자동 분석 및 시각화
        for i, df_single in enumerate(dfs):
            auto_data_overview(df_single, names[i])
            auto_visualization(df_single, names[i])

        # Agent는 첫번째 데이터프레임 기준 생성 (추후 병합도 가능)
        st.session_state["python_tool"] = PythonAstREPLTool()
        st.session_state["python_tool"].locals["pd"] = pd
        st.session_state["python_tool"].locals["sns"] = sns
        st.session_state["python_tool"].locals["plt"] = plt
        st.session_state["python_tool"].locals["df"] = dfs[0]  # 기본 df 지정
        st.session_state["agent"] = create_agent(dfs[0])

        st.success("데이터 분석이 완료되었습니다. 궁금한 내용을 물어보세요!")

# 메시지 출력
print_messages()

# 사용자 입력 처리
user_input = st.chat_input("궁금한 내용을 물어보세요!")
if user_input:
    ask(user_input)
    print_messages()

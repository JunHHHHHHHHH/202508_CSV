import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from typing import List, Union, Dict, Any

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

class StreamlitCallbackHandler(BaseCallbackHandler):
    """Streamlit UI에 Agent의 중간 과정을 스트리밍하기 위한 콜백 핸들러"""
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        st.session_state.container.markdown(token, unsafe_allow_html=True)
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        pass  # 코드 실행 과정을 화면에 노출하지 않음

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = {}
    if "selected_file" not in st.session_state:
        st.session_state["selected_file"] = None
    if "df" not in st.session_state:
        st.session_state["df"] = None
    if "agent" not in st.session_state:
        st.session_state["agent"] = None
    if "chart_gallery" not in st.session_state:
        st.session_state["chart_gallery"] = []

def create_agent(df: pd.DataFrame, api_key: str):
    if not api_key:
        st.error("OpenAI API 키를 사이드바에 입력해주세요.")
        return None
    return create_pandas_dataframe_agent(
        llm=ChatOpenAI(model="gpt-4-turbo", temperature=0, openai_api_key=api_key, streaming=True),
        df=df,
        agent_type="openai-tools",
        verbose=True,
        prompt="""
        당신은 최고의 데이터 분석가이자 파이썬 전문가입니다. 'df'라는 이름의 Pandas DataFrame을 사용하여 사용자의 질문에 답변해야 합니다.
        1. 시각화 요청 시 반드시 Plotly를 사용하세요.
        2. 제공된 `df`를 직접 수정하는 코드는 절대 생성하지 마세요.
        3. 모든 답변은 반드시 한국어로 제공하세요.
        4. 데이터 탐색 시 df.columns, df.head(), df.info(), df.describe() 등을 활용하세요.
        5. 주요 통계치 요약, 핵심 인사이트 등은 markdown 테이블로 깔끔하게 정리해서 리포트에 포함해 주세요.
        예시:
        | 항목     | 평균   | 표준편차 |
        |----------|--------|---------|
        | hurdles  | 13.17  | 0.40    |
        | highjump | 1.81   | 0.04    |
        | shot     | 15.24  | 0.81    |
        """,
        allow_dangerous_code=True,
    )

def display_dashboard(df: pd.DataFrame):
    if df is None:
        st.warning("먼저 사이드바에서 CSV 파일을 업로드하고 선택해주세요.")
        return
    st.subheader("📊 데이터 대시보드")
    st.dataframe(df.head())
    col1, col2 = st.columns(2)
    with col1:
        st.metric("전체 행 수", f"{df.shape[0]:,}")
    with col2:
        st.metric("전체 열 수", f"{df.shape[1]:,}")
    buffer = pd.io.common.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())
    st.dataframe(df.describe().transpose())
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    if not missing_data.empty:
        fig = px.bar(missing_data, x=missing_data.index, y=missing_data.values,
                    title="결측치 개수", labels={'x': '컬럼', 'y': '결측치 수'},
                    template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("🎉 데이터에 결측치가 없습니다!")
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(numeric_cols) > 0:
        st.markdown("#### 🔹 수치형 데이터 (Numeric)")
        selected_numeric = st.selectbox("분포를 확인할 수치형 컬럼을 선택하세요.", options=numeric_cols)
        fig_hist = px.histogram(df, x=selected_numeric, title=f"'{selected_numeric}' 컬럼 분포", template='plotly_white')
        st.plotly_chart(fig_hist, use_container_width=True)
    if len(categorical_cols) > 0:
        st.markdown("#### 🔹 범주형 데이터 (Categorical)")
        selected_categorical = st.selectbox("분포를 확인할 범주형 컬럼을 선택하세요.", options=categorical_cols)
        fig_bar = px.bar(df[selected_categorical].value_counts(), title=f"'{selected_categorical}' 컬럼 분포", template='plotly_white')
        st.plotly_chart(fig_bar, use_container_width=True)

def display_chat_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["type"] == "text":
                st.markdown(msg["content"])
            elif msg["type"] == "dataframe":
                st.dataframe(msg["content"])
            elif msg["type"] == "figure":
                st.plotly_chart(msg["content"], use_container_width=True)

def add_message(role, content, msg_type="text"):
    st.session_state.messages.append({"role": role, "content": content, "type": msg_type})

def run_agent(query: str, display_prompt: bool = True):
    if st.session_state.agent is None:
        st.error("에이전트가 초기화되지 않았습니다. API 키를 확인하고 파일을 다시 업로드해주세요.")
        return
    if display_prompt:
        add_message("user", query)
        st.chat_message("user").markdown(query)
    with st.chat_message("assistant"):
        st.session_state.container = st.empty()
        try:
            response = st.session_state.agent.invoke(
                {"input": query},
                {"callbacks": [StreamlitCallbackHandler()]}
            )
            final_answer = response.get("output", "죄송합니다, 답변을 생성하지 못했습니다.")
            intermediate_steps = response.get("intermediate_steps", [])
            for step in intermediate_steps:
                tool_output = step[1]
                if isinstance(tool_output, go.Figure):
                    st.plotly_chart(tool_output, use_container_width=True)
                    add_message("assistant", tool_output, "figure")
                    st.session_state.chart_gallery.append(tool_output)
                elif isinstance(tool_output, pd.DataFrame):
                    st.dataframe(tool_output)
                    add_message("assistant", tool_output, "dataframe")
            add_message("assistant", final_answer, "text")
        except Exception as e:
            error_message = f"오류가 발생했습니다: {e}"
            st.error(error_message)
            add_message("assistant", error_message)

def main():
    init_session_state()
    st.set_page_config(page_title="🤖 AI CSV 분석 챗봇", page_icon="📊", layout="wide")
    st.title("🤖 AI CSV 분석 챗봇 (v2.0)")
    st.markdown("CSV 파일을 업로드하고 데이터에 대해 질문하거나 자동 분석 기능을 사용해보세요.")
    with st.sidebar:
        st.header("설정")
        api_key = st.text_input("🔑 OpenAI API Key", type="password", key="api_key_input")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        uploaded_files = st.file_uploader("📁 CSV 파일 업로드", type=["csv"], accept_multiple_files=True)
        if uploaded_files:
            for file in uploaded_files:
                if file.name not in st.session_state.uploaded_files:
                    st.session_state.uploaded_files[file.name] = pd.read_csv(file)
            file_names = list(st.session_state.uploaded_files.keys())
            st.session_state.selected_file = st.selectbox("분석할 파일을 선택하세요.", options=file_names)
            if st.session_state.selected_file:
                st.session_state.df = st.session_state.uploaded_files[st.session_state.selected_file]
        if st.button("🔄️ 대화 초기화"):
            st.session_state.messages = []
            st.session_state.chart_gallery = []
            st.rerun()
    if st.session_state.df is not None and (st.session_state.agent is None or "df_name" not in st.session_state or st.session_state.df_name != st.session_state.selected_file):
        with st.spinner("AI 에이전트를 준비하는 중입니다..."):
            st.session_state.agent = create_agent(st.session_state.df, os.environ.get("OPENAI_API_KEY", ""))
            st.session_state.df_name = st.session_state.selected_file
            st.toast(f"✅ '{st.session_state.selected_file}' 파일로 에이전트가 준비되었습니다!")
    tab1, tab2, tab3 = st.tabs(["💬 AI 챗봇", "📊 데이터 대시보드", "🖼️ 차트 갤러리"])
    with tab2:
        display_dashboard(st.session_state.df)
    with tab3:
        st.subheader("🖼️ 생성된 차트 모음")
        if not st.session_state.chart_gallery:
            st.info("아직 생성된 차트가 없습니다. AI 챗봇에게 시각화를 요청해보세요.")
        else:
            for i, chart in enumerate(st.session_state.chart_gallery):
                st.plotly_chart(chart, use_container_width=True)
                st.divider()
    with tab1:
        st.subheader("💬 AI에게 데이터에 대해 질문해보세요 (리포트 작성에 다소 시간이 소요됩니다)")
        if st.button("🤖 AI 자동 리포트 생성"):
            auto_report_prompt = """
            ## 탐색적 데이터 분석(EDA) 리포트

            업로드된 데이터프레임 `df`에 대한 종합적인 탐색적 데이터 분석(EDA) 리포트를 생성해줘.

            리포트에는 다음이 반드시 포함되어야 해:
            - **데이터 요약**: 데이터 크기, 컬럼 수, 주요 통계치 등.
            - **핵심 인사이트**: 가장 눈에 띄는 인사이트 5가지.
            - **상관관계 분석**: 수치형 변수들 간 상관관계(히트맵 포함).
            - **회귀분석**: 수치형 변수들 간 회귀분석(그래프 포함).
            - **이상치 분석**: 주요 컬럼 이상치(Box Plot 포함).
            - **결측치 및 데이터 품질**: 결측치, 타입 오류 등 문제점 분석.

            **특히, 주요 통계치 요약은 마크다운 테이블로 깔끔하게 정리해 주세요.**
            **아래 예시처럼 markdown 테이블로 주요 통계를 보여주세요.**

            예시:
            | 항목     | 평균   | 표준편차 |
            |----------|--------|---------|
            | hurdles  | 13.17  | 0.40    |
            | highjump | 1.81   | 0.04    |
            | shot     | 15.24  | 0.81    |
            | run200m  | 23.43  | 0.59    |
            | longjump | 6.65   | 0.42    |
            | javelin  | 44.60  | 2.05    |
            | run800m  | 127.79 | 3.00    |
            | score    | 6825.20| 310.60  |

            전문가 수준의 상세한 리포트를 마크다운 형식으로 작성해줘.
            """
            run_agent(auto_report_prompt, display_prompt=False)
        st.divider()
        display_chat_history()
        if prompt := st.chat_input("데이터에 대해 질문을 입력하세요..."):
            run_agent(prompt)

if __name__ == "__main__":
    main()



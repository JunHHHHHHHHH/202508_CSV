import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from typing import Dict, Any, Optional
import io
import logging

# --- Optional external logging (TeddyNote)
# logging.langsmith("CSV_Agent_Chatbot_v2")

# --- 1. Constants and Settings ---
STREAMLIT_THEME = "plotly_white"
LANGCHAIN_PROMPT = """
당신은 최고의 데이터 분석가이자 파이썬 전문가입니다. 'df'라는 이름의 Pandas DataFrame을 사용하여 사용자의 질문에 답변해야 합니다.

주요 지침:
1. **시각화**: 반드시 Plotly만 사용(`import plotly.express as px` 또는 `import plotly.graph_objects as go`). 
   Matplotlib, Seaborn은 사용 금지. 차트 생성 시, 제목/레이블은 영문. `template='plotly_white'` 사용.
   차트 반환은 코드 블럭의 마지막에 `fig = px.bar(...); fig` 형태로.
2. **코드 실행**: `df = ...`처럼 원본 데이터프레임을 직접 수정하는 코드는 생성 금지.
3. **답변 언어**: 모든 최종 답변은 반드시 한국어로.
4. **데이터 탐색**: `df.columns`, `df.head()`, `df.info()`, `df.describe()` 등으로 데이터 구조 설명.
"""

# --- 2. State Management Functions ---
def init_session_state():
    """세션 상태 초기화"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = {}  # 파일 이름: 데이터프레임
    if "selected_file" not in st.session_state:
        st.session_state.selected_file = None
    if "df" not in st.session_state:
        st.session_state.df = None
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "chart_gallery" not in st.session_state:
        st.session_state.chart_gallery = []
    if "last_code_shown" not in st.session_state:
        st.session_state.last_code_shown = None

# --- 3. Agent Callback Handlers ---
class CustomStreamlitCallbackHandler:
    """Agent의 중간 과정을 스트리밍하며, 불필요한 코드 노출을 최소화"""
    def __init__(self):
        self.is_showing_code = False
        self.container = None
        self.last_code = None

    def set_container(self, container):
        self.container = container

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.container:
            self.container.markdown(token, unsafe_allow_html=True)

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any):
        """코드 셀을 조건부로 보여주고, 중복/의미 없는 코드는 감춤"""
        # 이전에 보인 코드면 표시하지 않음
        if st.session_state.last_code_shown != input_str:
            with st.status("코드 실행 중... 👨‍💻", expanded=True):
                if not input_str.startswith(('import ', '#', '')):
                    st.code(input_str, language="python")
                    st.session_state.last_code_shown = input_str

# --- 4. UI Components & Functions ---
def display_dashboard(df: pd.DataFrame):
    """업로드된 데이터 자동 분석 대시보드"""
    st.subheader("📊 데이터 대시보드")

    if df is None:
        st.warning("CSV 파일을 업로드 후 선택해주세요.")
        return

    # 데이터 개요
    st.markdown("### 1. 데이터 개요 (Overview)")
    st.dataframe(df.head())
    col1, col2 = st.columns(2)
    with col1:
        st.metric("전체 행 수", f"{df.shape[0]:,}")
    with col2:
        st.metric("전체 열 수", f"{df.shape[1]:,}")

    # 데이터 정보
    st.markdown("### 2. 데이터 정보 (Data Types & Memory)")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    # 기본 통계
    st.markdown("### 3. 기본 통계 (Descriptive Statistics)")
    st.dataframe(df.describe().transpose())

    # 결측치 분석
    st.markdown("### 4. 결측치 분석 (Missing Values)")
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    if not missing_data.empty:
        fig = px.bar(
            missing_data, x=missing_data.index, y=missing_data.values,
            title="결측치 개수", labels={'x': '컬럼', 'y': '결측치 수'},
            template=STREAMLIT_THEME
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("🎉 데이터에 결측치가 없습니다!")

    # 데이터 분포
    st.markdown("### 5. 데이터 분포 (Data Distribution)")
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    if len(numeric_cols) > 0:
        st.markdown("#### 🔹 수치형 데이터 (Numeric)")
        selected_numeric = st.selectbox("분포를 확인할 수치형 컬럼을 선택하세요", numeric_cols)
        fig_hist = px.histogram(
            df, x=selected_numeric,
            title=f"'{selected_numeric}' 컬럼 분포",
            template=STREAMLIT_THEME
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    if len(categorical_cols) > 0:
        st.markdown("#### 🔹 범주형 데이터 (Categorical)")
        selected_categorical = st.selectbox("분포를 확인할 범주형 컬럼을 선택하세요", categorical_cols)
        fig_bar = px.bar(
            df[selected_categorical].value_counts(),
            title=f"'{selected_categorical}' 컬럼 분포",
            template=STREAMLIT_THEME
        )
        st.plotly_chart(fig_bar, use_container_width=True)

def display_chat_history():
    """채팅 기록 표시"""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["type"] == "text":
                st.markdown(msg["content"])
            elif msg["type"] == "dataframe":
                st.dataframe(msg["content"])
            elif msg["type"] == "figure":
                st.plotly_chart(msg["content"], use_container_width=True)

def add_message(role: str, content: Any, msg_type: str = "text"):
    """채팅에 메시지 추가"""
    st.session_state.messages.append({"role": role, "content": content, "type": msg_type})

def create_agent(df: pd.DataFrame, api_key: str):
    """Pandas DataFrame Agent 생성"""
    from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
    from langchain_openai import ChatOpenAI
    if not api_key:
        st.error("OpenAI API 키를 입력해주세요.")
        return None
    return create_pandas_dataframe_agent(
        llm=ChatOpenAI(
            model="gpt-4-turbo", temperature=0, openai_api_key=api_key, streaming=True
        ),
        df=df,
        agent_type="openai-tools",
        verbose=False,  # verbose=False로 간접 노이즈 감소
        prompt=LANGCHAIN_PROMPT,
        allow_dangerous_code=True,  # 실제 프로덕션에서는 주의
    )

def reset_conversation():
    """대화 초기화"""
    st.session_state.messages = []
    st.session_state.chart_gallery = []
    st.session_state.last_code_shown = None
    st.rerun()

def run_agent(query: str, display_prompt: bool = True):
    """에이전트 실행 및 결과 처리"""
    if st.session_state.agent is None:
        st.error("에이전트 미생성. API 키와 파일을 확인하세요.")
        return

    if display_prompt:
        add_message("user", query)
        st.chat_message("user").markdown(query)

    with st.chat_message("assistant"):
        container = st.empty()
        callback = CustomStreamlitCallbackHandler()
        callback.set_container(container)

        try:
            response = st.session_state.agent.invoke(
                {"input": query},
                {"callbacks": [callback]}
            )
            result = response.get("output", "답변 생성 중 문제 발생")
            df_result = None
            fig_result = None

            for step in response.get("intermediate_steps", []):
                tool_output = step[1]
                if isinstance(tool_output, (go.Figure, px._figure.Figure)):
                    st.plotly_chart(tool_output, use_container_width=True)
                    add_message("assistant", tool_output, "figure")
                    st.session_state.chart_gallery.append(tool_output)
                elif isinstance(tool_output, pd.DataFrame):
                    st.dataframe(tool_output)
                    add_message("assistant", tool_output, "dataframe")

            add_message("assistant", result, "text")
        except Exception as e:
            error_msg = f"오류 발생: {e}"
            st.error(error_msg)
            add_message("assistant", error_msg)

# --- 5. Main App Logic ---
def main():
    init_session_state()

    st.set_page_config(
        page_title="🤖 AI CSV 분석 챗봇",
        page_icon="📊",
        layout="wide",
    )

    st.title("🤖 AI CSV 분석 챗봇 (v2.1)")
    st.markdown("CSV 파일을 업로드하고 데이터에 대해 질문하거나, 자동 분석 기능을 사용해보세요.")

    # 사이드바
    with st.sidebar:
        st.header("설정")
        api_key = st.text_input("🔑 OpenAI API Key", type="password", key="api_key_input")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        uploaded_files = st.file_uploader(
            "📁 CSV 파일 업로드", type=["csv"], accept_multiple_files=True
        )
        if uploaded_files:
            for file in uploaded_files:
                if file.name not in st.session_state.uploaded_files:
                    st.session_state.uploaded_files[file.name] = pd.read_csv(file)
            file_names = list(st.session_state.uploaded_files.keys())
            st.session_state.selected_file = st.selectbox(
                "분석할 파일을 선택하세요", file_names
            )
            if st.session_state.selected_file:
                st.session_state.df = st.session_state.uploaded_files[st.session_state.selected_file]

        if st.button("🔄️ 대화 초기화"):
            reset_conversation()

    # 에이전트 생성 (파일 변경 시)
    if st.session_state.df is not None and (st.session_state.agent is None or "df_name" not in st.session_state or st.session_state.df_name != st.session_state.selected_file):
        with st.spinner("AI 에이전트 준비 중..."):
            st.session_state.agent = create_agent(st.session_state.df, os.environ.get("OPENAI_API_KEY"))
            st.session_state.df_name = st.session_state.selected_file
            st.toast(f"✅ '{st.session_state.selected_file}' 파일로 에이전트가 준비되었습니다!")

    # 메인 콘텐츠: 채팅, 대시보드, 차트 갤러리 탭
    tab_chat, tab_dashboard, tab_gallery = st.tabs(["💬 AI 챗봇", "📊 데이터 대시보드", "🖼️ 차트 갤러리"])

    with tab_dashboard:
        display_dashboard(st.session_state.df)

    with tab_gallery:
        st.subheader("🖼️ 생성된 차트 모음")
        if not st.session_state.chart_gallery:
            st.info("차트가 없습니다. AI 챗봇에게 시각화를 요청해주세요.")
        else:
            for i, chart in enumerate(st.session_state.chart_gallery):
                st.plotly_chart(chart, use_container_width=True)
                st.divider()

    with tab_chat:
        st.subheader("💬 AI에게 데이터에 대해 질문하세요")

        if st.button("🤖 AI 자동 리포트 생성"):
            run_agent("""
            ## 탐색적 데이터 분석(EDA) 리포트

            업로드된 데이터프레임 `df`에 대한 종합적인 리포트를 생성해줘.

            1. **데이터 요약**: 전체 크기, 컬럼 수, 주요 통계치 요약
            2. **핵심 인사이트**: 중요한 비즈니스/데이터 인사이트 3가지
            3. **상관관계 분석**: 수치형 변수 간 상관관계, 히트맵 시각화
            4. **이상치(Outlier) 분석**: 주요 수치형 컬럼 이상치 분석, Box Plot 시각화
            5. **데이터 품질 문제**: 결측치, 데이터 타입 오류 등 문제점 및 해결 방안

            전문가 수준의 상세한 리포트를 마크다운 형식으로 작성해줘.
            """,
            display_prompt=False)

        st.divider()

        display_chat_history()

        if prompt := st.chat_input("데이터에 대해 질문을 입력하세요..."):
            run_agent(prompt)

if __name__ == "__main__":
    main()


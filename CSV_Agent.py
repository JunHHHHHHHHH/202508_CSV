import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from typing import List, Union, Dict, Any

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging
from langchain.callbacks.base import BaseCallbackHandler

# --- 1. 초기 설정 및 페이지 구성 ---
st.set_page_config(
    page_title="🤖 AI CSV 분석 챗봇",
    page_icon="📊",
    layout="wide",
)

# TeddyNote의 langsmith 로깅 (선택 사항)
# logging.langsmith("CSV_Agent_Chatbot_v2")

# --- 2. 세션 상태 관리 ---
def init_session_state():
    """세션 상태 초기화"""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = {}  # 파일 이름과 데이터프레임을 저장
    if "selected_file" not in st.session_state:
        st.session_state["selected_file"] = None
    if "df" not in st.session_state:
        st.session_state["df"] = None
    if "agent" not in st.session_state:
        st.session_state["agent"] = None
    if "chart_gallery" not in st.session_state:
        st.session_state["chart_gallery"] = []

# --- 3. LangChain Agent 및 콜백 핸들러 ---
class StreamlitCallbackHandler(BaseCallbackHandler):
    """Streamlit UI에 Agent의 중간 과정을 스트리밍하기 위한 콜백 핸들러"""
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        st.session_state.container.markdown(token, unsafe_allow_html=True)

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        with st.status("코드 실행 중... 👨‍💻", expanded=True):
            st.code(input_str, language="python")

def create_agent(df: pd.DataFrame, api_key: str):
    """Pandas DataFrame Agent 생성"""
    if not api_key:
        st.error("OpenAI API 키를 사이드바에 입력해주세요.")
        return None
    
    prompt = """
    당신은 최고의 데이터 분석가이자 파이썬 전문가입니다. 'df'라는 이름의 Pandas DataFrame을 사용하여 사용자의 질문에 답변해야 합니다.
    
    주요 지침:
    1.  **시각화**: 시각화 요청 시, 반드시 **Plotly**를 사용하세요. Matplotlib 이나 Seaborn은 사용하지 마세요.
        -   `import plotly.express as px` 또는 `import plotly.graph_objects as go`를 사용하세요.
        -   생성된 Plotly Figure 객체는 코드 블록의 마지막 줄에 위치시켜야 반환됩니다. 예: `fig = px.bar(...)`, `fig`
        -   차트의 제목, 축 레이블 등은 `English`로 작성해주세요.
        -   `template='plotly_white'`를 사용하여 깔끔한 배경을 만드세요.
    2.  **코드 실행**: 제공된 `df`를 직접 수정하는 코드(예: `df = ...`)는 절대 생성하지 마세요.
    3.  **답변**: 모든 최종 답변은 반드시 **'한국어'**로 제공해야 합니다. 코드 실행 결과(데이터프레임, 차트 등)를 바탕으로 상세하고 친절하게 설명해주세요.
    4.  **데이터 탐색**: 사용자가 데이터에 대해 물어보면 `df.columns`, `df.head()`, `df.info()`, `df.describe()` 등을 활용하여 정확한 정보를 제공하세요.
    """
    
    return create_pandas_dataframe_agent(
        llm=ChatOpenAI(model="gpt-4-turbo", temperature=0, openai_api_key=api_key, streaming=True),
        df=df,
        agent_type="openai-tools",
        verbose=True,
        prompt=prompt,
        allow_dangerous_code=True,
    )

# --- 4. UI 컴포넌트 및 기능 함수 ---

def display_dashboard(df: pd.DataFrame):
    """업로드된 데이터에 대한 자동 분석 대시보드 표시"""
    st.subheader("📊 데이터 대시보드")
    
    if df is None:
        st.warning("먼저 사이드바에서 CSV 파일을 업로드하고 선택해주세요.")
        return

    # 1. 데이터 개요
    st.markdown("### 1. 데이터 개요 (Overview)")
    st.dataframe(df.head())
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("전체 행 수", f"{df.shape[0]:,}")
    with col2:
        st.metric("전체 열 수", f"{df.shape[1]:,}")

    st.markdown("### 2. 데이터 정보 (Data Types & Memory)")
    buffer = pd.io.common.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    # 3. 기본 통계
    st.markdown("### 3. 기본 통계 (Descriptive Statistics)")
    st.dataframe(df.describe().transpose())
    
    # 4. 결측치 분석
    st.markdown("### 4. 결측치 분석 (Missing Values)")
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    if not missing_data.empty:
        fig = px.bar(missing_data, x=missing_data.index, y=missing_data.values,
                     title="결측치 개수", labels={'x': '컬럼', 'y': '결측치 수'},
                     template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("🎉 데이터에 결측치가 없습니다!")

    # 5. 데이터 타입별 분포 시각화 (자동 차트 추천)
    st.markdown("### 5. 데이터 분포 (Data Distribution)")
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
    """채팅 기록 표시"""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["type"] == "text":
                st.markdown(msg["content"])
            elif msg["type"] == "dataframe":
                st.dataframe(msg["content"])
            elif msg["type"] == "figure":
                st.plotly_chart(msg["content"], use_container_width=True)

def add_message(role, content, msg_type="text"):
    """세션에 메시지 추가"""
    st.session_state.messages.append({"role": role, "content": content, "type": msg_type})

def run_agent(query: str, display_prompt: bool = True):
    """에이전트를 실행하고 결과를 처리. display_prompt로 프롬프트 표시 여부 제어"""
    if st.session_state.agent is None:
        st.error("에이전트가 초기화되지 않았습니다. API 키를 확인하고 파일을 다시 업로드해주세요.")
        return

    if display_prompt:
        add_message("user", query)
        st.chat_message("user").markdown(query)

    with st.chat_message("assistant"):
        st.session_state.container = st.empty() # 스트리밍 출력을 위한 컨테이너
        
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


# --- 5. 메인 앱 실행 로직 ---
def main():
    init_session_state()

    st.title("🤖 AI CSV 분석 챗봇 (v2.0)")
    st.markdown("CSV 파일을 업로드하고 데이터에 대해 질문하거나 자동 분석 기능을 사용해보세요.")

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
            st.session_state.selected_file = st.selectbox("분석할 파일을 선택하세요.", options=file_names)
            
            if st.session_state.selected_file:
                st.session_state.df = st.session_state.uploaded_files[st.session_state.selected_file]

        if st.button("🔄️ 대화 초기화"):
            st.session_state.messages = []
            st.session_state.chart_gallery = []
            st.rerun()

    # 에이전트 생성 (파일이 변경될 때마다)
    if st.session_state.df is not None and (st.session_state.agent is None or "df_name" not in st.session_state or st.session_state.df_name != st.session_state.selected_file):
        with st.spinner("AI 에이전트를 준비하는 중입니다..."):
            st.session_state.agent = create_agent(st.session_state.df, os.environ.get("OPENAI_API_KEY", ""))
            st.session_state.df_name = st.session_state.selected_file # 현재 df 이름 저장
            st.toast(f"✅ '{st.session_state.selected_file}' 파일로 에이전트가 준비되었습니다!")


    # 메인 콘텐츠 영역 (탭)
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
        st.subheader("💬 AI에게 데이터에 대해 질문해보세요")
        
        # 자동 분석 버튼
        if st.button("🤖 AI 자동 리포트 생성"):
            # AI에게 전달할 프롬프트. 리포트 제목을 마크다운으로 명시.
            auto_report_prompt = """
            ## 탐색적 데이터 분석(EDA) 리포트

            업로드된 데이터프레임 `df`에 대한 종합적인 탐색적 데이터 분석(EDA) 리포트를 생성해줘.

            리포트에는 다음 내용이 반드시 포함되어야 해:
            1.  **데이터 요약**: 데이터의 전체적인 크기, 컬럼 수, 주요 통계치에 대한 요약.
            2.  **핵심 인사이트**: 데이터를 분석하여 발견한 가장 중요한 비즈니스 또는 데이터 인사이트 3가지.
            3.  **상관 관계 분석**: 수치형 변수들 간의 상관 관계를 분석하고, 가장 강한 양의 상관관계와 음의 상관관계를 보이는 변수 쌍을 설명해줘. 히트맵 시각화도 함께 생성해줘.
            4.  **이상치(Outlier) 분석**: 주요 수치형 컬럼에서 잠재적인 이상치가 있는지 분석하고, 있다면 어떤 값인지 알려줘. Box Plot 시각화를 1개 생성해서 보여주면 좋아.
            5.  **데이터 품질 문제**: 결측치나 데이터 타입 오류 등 잠재적인 데이터 품질 문제를 언급하고, 간단한 해결 방안을 제안해줘.

            위 내용을 바탕으로 전문가 수준의 상세한 리포트를 마크다운 형식으로 작성해줘.
            """
            # display_prompt=False로 설정하여 프롬프트가 화면에 보이지 않게 함
            run_agent(auto_report_prompt, display_prompt=False)

        st.divider()

        # 채팅 기록 표시
        display_chat_history()

        # 사용자 입력
        if prompt := st.chat_input("데이터에 대해 질문을 입력하세요..."):
            run_agent(prompt)

if __name__ == "__main__":
    main()

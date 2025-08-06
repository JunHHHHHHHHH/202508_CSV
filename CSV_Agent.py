import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from typing import List, Union, Dict, Any, Optional
from datetime import datetime

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

# CSV 파일 로딩 캐싱
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    """CSV 파일을 로드하고 캐싱합니다."""
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"CSV 파일을 읽는 중 오류 발생: {str(e)}")
        return None

# 에이전트 생성 캐싱
@st.cache_resource
def create_cached_agent(df: pd.DataFrame, api_key: str) -> Optional[Any]:
    """pandas DataFrame 에이전트를 생성하고 캐싱합니다."""
    if not api_key:
        st.error("OpenAI API 키를 사이드바에 입력해주세요.")
        return None
    try:
        return create_pandas_dataframe_agent(
            llm=ChatOpenAI(
                model="gpt-4-turbo",
                temperature=0,
                openai_api_key=api_key,
                streaming=True
            ),
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
    except Exception as e:
        st.error(f"에이전트 생성 중 오류 발생: {str(e)}")
        return None

class StreamlitCallbackHandler(BaseCallbackHandler):
    """Streamlit UI에 Agent의 중간 과정을 스트리밍하기 위한 콜백 핸들러"""
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """스트리밍된 토큰을 수집하고 화면에 표시합니다."""
        self.text += token
        self.container.markdown(self.text, unsafe_allow_html=True)

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        pass  # 코드 실행 과정을 화면에 노출하지 않음

    def get_final_text(self) -> str:
        """수집된 최종 텍스트를 반환합니다."""
        return self.text

def init_session_state():
    """세션 상태 변수를 초기화합니다."""
    defaults = {
        "messages": [],
        "uploaded_files": {},
        "selected_file": None,
        "df": None,
        "agent": None,
        "df_name": None,
        "last_message": None,  # 마지막 메시지 추적
        "last_stream_id": None,  # 마지막 스트리밍 세션 ID
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def display_dashboard(df: pd.DataFrame):
    """데이터 대시보드를 표시하며 요약 통계와 시각화를 포함합니다."""
    if df is None:
        st.warning("먼저 사이드바에서 CSV 파일을 업로드하고 선택해주세요.")
        return

    st.subheader("📊 데이터 대시보드")
    st.dataframe(df.head(), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("전체 행 수", f"{df.shape[0]:,}")
    with col2:
        st.metric("전체 열 수", f"{df.shape[1]:,}")

    with st.expander("데이터 정보"):
        buffer = pd.io.common.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
        st.dataframe(df.describe().transpose(), use_container_width=True)

    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    if not missing_data.empty:
        fig = px.bar(
            missing_data,
            x=missing_data.index,
            y=missing_data.values,
            title="결측치 개수",
            labels={'x': '컬럼', 'y': '결측치 수'},
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("🎉 데이터에 결측치가 없습니다!")

    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    if len(numeric_cols) > 0:
        st.markdown("#### 🔹 수치형 데이터 (Numeric)")
        selected_numeric = st.selectbox(
            "분포를 확인할 수치형 컬럼을 선택하세요.",
            options=numeric_cols,
            key="numeric_select"
        )
        fig_hist = px.histogram(
            df,
            x=selected_numeric,
            title=f"'{selected_numeric}' 컬럼 분포",
            template='plotly_white',
            color_discrete_sequence=['#1f77b4']
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    if len(categorical_cols) > 0:
        st.markdown("#### 🔹 범주형 데이터 (Categorical)")
        selected_categorical = st.selectbox(
            "분포를 확인할 범주형 컬럼을 선택하세요.",
            options=categorical_cols,
            key="categorical_select"
        )
        fig_bar = px.bar(
            df[selected_categorical].value_counts(),
            title=f"'{selected_categorical}' 컬럼 분포",
            template='plotly_white',
            color_discrete_sequence=['#ff7f0e']
        )
        st.plotly_chart(fig_bar, use_container_width=True)

def display_chat_history():
    """메시지, DataFrame, 차트를 포함한 채팅 기록을 표시합니다."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["type"] == "text":
                st.markdown(msg["content"])
            elif msg["type"] == "dataframe":
                st.dataframe(msg["content"], use_container_width=True)
            elif msg["type"] == "figure":
                st.plotly_chart(msg["content"], use_container_width=True)

def add_message(role: str, content: Any, msg_type: str = "text"):
    """세션 상태에 메시지를 추가합니다. 중복 메시지 방지."""
    # 동일한 내용의 메시지가 이미 존재하는지 확인
    if st.session_state.last_message != content:
        st.session_state.messages.append({
            "role": role,
            "content": content,
            "type": msg_type,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        st.session_state.last_message = content

def run_agent(query: str, display_prompt: bool = True, stream_id: str = None):
    """주어진 쿼리로 에이전트를 실행하고 결과를 한 번에만 챗창에 표시합니다."""
    if st.session_state.agent is None:
        st.error("에이전트가 초기화되지 않았습니다. API 키를 확인하고 파일을 다시 업로드해주세요.")
        return

    # 프롬프트가 보이고, 중복된 stream_id 세션을 피하기 위해 session 저장
    if display_prompt:
        add_message("user", query, "text")
        with st.chat_message("user"):
            st.markdown(query)

    if stream_id and stream_id == st.session_state.get("last_stream_id", None):
        return  # 같은 스트리밍 세션이 이미 처리된 경우 중복 실행 방지

    with st.chat_message("assistant"):
        stream_container = st.empty()
        callback_handler = StreamlitCallbackHandler(stream_container)
        try:
            # 스트리밍 중에는 메시지 추가 하지 않음
            with st.spinner("분석 중..."):
                response = st.session_state.agent.invoke(
                    {"input": query},
                    {"callbacks": [callback_handler]}
                )
            stream_text = callback_handler.get_final_text()
            intermediate_steps = response.get("intermediate_steps", [])
            
            # 스트리밍 종료 후, 중간 단계 결과(차트, 표 등)와 텍스트를 각각 한 번씩 추가
            for step in intermediate_steps:
                tool_output = step[1]
                if isinstance(tool_output, go.Figure):
                    st.plotly_chart(tool_output, use_container_width=True)
                    add_message("assistant", tool_output, "figure")
                elif isinstance(tool_output, pd.DataFrame):
                    st.dataframe(tool_output, use_container_width=True)
                    add_message("assistant", tool_output, "dataframe")
                # 텍스트 메시지가 누적되는 문제 해결: 차트/표만 별도로 추가
            
            # 스트리밍된 마크다운 텍스트(리포트)는 한 번만 추가
            if stream_text.strip():
                st.markdown(stream_text)
                add_message("assistant", stream_text, "text")
            
            st.session_state["last_stream_id"] = stream_id  # 중복 방지 세션 추적
        except Exception as e:
            error_message = f"분석 중 오류 발생: {str(e)}"
            st.error(error_message)
            add_message("assistant", error_message, "text")

def setup_sidebar():
    """사이드바에 API 키 입력과 파일 업로더를 설정합니다."""
    with st.sidebar:
        st.header("설정")
        api_key = st.text_input("🔑 OpenAI API Key", type="password", key="api_key_input")
        if api_key:
            st.session_state["api_key"] = api_key
        
        uploaded_files = st.file_uploader(
            "📁 CSV 파일 업로드",
            type=["csv"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            with st.spinner("파일을 읽는 중..."):
                for file in uploaded_files:
                    if file.name not in st.session_state.uploaded_files:
                        df = load_csv(file)
                        if df is not None:
                            st.session_state.uploaded_files[file.name] = df
            file_names = list(st.session_state.uploaded_files.keys())
            st.session_state.selected_file = st.selectbox(
                "분석할 파일을 선택하세요.",
                options=file_names
            )
            if st.session_state.selected_file:
                st.session_state.df = st.session_state.uploaded_files[st.session_state.selected_file]
        
        if st.button("🔄️ 대화 초기화"):
            st.session_state.messages = []
            st.session_state.agent = None
            st.session_state.df_name = None
            st.session_state.last_message = None
            st.session_state.last_stream_id = None
            st.rerun()

def main():
    """Streamlit 앱을 실행하는 메인 함수입니다."""
    init_session_state()
    st.set_page_config(
        page_title="🤖 AI CSV 분석 챗봇",
        page_icon="📊",
        layout="wide"
    )
    st.title("🤖 AI CSV 분석 챗봇 (v2.7)")
    st.markdown("CSV 파일을 업로드하고 데이터에 대해 질문하거나 자동 분석 기능을 사용해보세요.")

    setup_sidebar()

    if st.session_state.df is not None and (
        st.session_state.agent is None or
        st.session_state.get("df_name") != st.session_state.selected_file
    ):
        with st.spinner("AI 에이전트를 준비하는 중입니다..."):
            st.session_state.agent = create_cached_agent(
                st.session_state.df,
                st.session_state.get("api_key", "")
            )
            st.session_state.df_name = st.session_state.selected_file
            if st.session_state.agent:
                st.toast(f"✅ '{st.session_state.selected_file}' 파일로 에이전트가 준비되었습니다!")
            else:
                st.error("에이전트 초기화에 실패했습니다. API 키를 확인해주세요.")

    tab1, tab2 = st.tabs(["💬 AI 챗봇", "📊 데이터 대시보드"])
    
    with tab2:
        display_dashboard(st.session_state.df)
    
    with tab1:
        st.subheader("💬 AI에게 데이터에 대해 질문해보세요")
        st.markdown("리포트 작성에 다소 시간이 소요될 수 있습니다.")
        if st.button("🤖 AI 자동 리포트 생성"):
            auto_report_prompt = """
            ## 탐색적 데이터 분석(EDA) 리포트

            업로드된 데이터프레임 `df`에 대한 종합적인 탐색적 데이터 분석(EDA) 리포트를 생성해줘.

            리포트에는 다음이 반드시 포함되어야 해:
            - **데이터 요약**: 데이터 크기, 컬럼 수, 주요 통계치 등(df.info(), df.describe() 사용).
            - **핵심 인사이트**: 가장 눈에 띄는 인사이트 5가지.
            - **상관관계 분석**: 수치형 변수들(df.select_dtypes(include=['number'])) 간 상관관계만 분석하고, 히트맵을 Plotly로 생성해. 문자열 데이터는 제외하고, 결측치는 df.dropna()로 처리하여 오류를 방지해.
            - **회귀분석**: `scikit-learn`을 사용하여 `score` 컬럼을 종속 변수로 하고, 나머지 수치형 변수들(df.select_dtypes(include=['number']).drop(columns=['score'], errors='ignore'))을 독립 변수로 회귀분석을 수행해. 결측치는 df.dropna()로 처리하고, 문자열 데이터는 제외하며, 산점도와 회귀선을 Plotly로 생성해.
            - **이상치 분석**: 주요 수치형 컬럼에 대해 Box Plot을 Plotly로 생성해 이상치를 시각화해. 결측치는 df.dropna()로 처리하고, 문자열 데이터는 제외해.
            - **결측치 및 데이터 품질**: 결측치(df.isnull().sum()), 타입 오류 등 문제점 분석.

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

            전문가 수준의 상세한 리포트를 마크다운 형식으로 작성해줘. 모든 분석은 반드시 성공적으로 수행되어야 하며, 오류가 발생하지 않도록 데이터 전처리를 철저히 수행해.
            """
            run_agent(auto_report_prompt, display_prompt=False, stream_id="auto_report")
        
        st.divider()
        display_chat_history()
        if prompt := st.chat_input("데이터에 대해 질문을 입력하세요..."):
            run_agent(prompt, stream_id=f"user_prompt_{datetime.now().strftime('%Y%m%d%H%M%S')}")

if __name__ == "__main__":
    main()

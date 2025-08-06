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
            6. 상관관계 분석, 회귀분석, 이상치 분석 시 pandas, numpy, plotly, statsmodels, scikit-learn 라이브러리를 사용하세요.
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
        "chart_gallery": [],
        "df_name": None,
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
    """세션 상태에 메시지를 추가합니다."""
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "type": msg_type,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

def run_agent(query: str, display_prompt: bool = True):
    """주어진 쿼리로 에이전트를 실행하고 결과를 표시합니다."""
    if st.session_state.agent is None:
        st.error("에이전트가 초기화되지 않았습니다. API 키를 확인하고 파일을 다시 업로드해주세요.")
        return

    if display_prompt:
        add_message("user", query)
        st.chat_message("user").markdown(query)

    with st.chat_message("assistant"):
        container = st.empty()
        callback_handler = StreamlitCallbackHandler(container)
        try:
            with st.spinner("분석 중..."):
                response = st.session_state.agent.invoke(
                    {"input": query},
                    {"callbacks": [callback_handler]}
                )
            final_answer = response.get("output", "죄송합니다, 답변을 생성하지 못했습니다.")
            intermediate_steps = response.get("intermediate_steps", [])
            
            for step in intermediate_steps:
                tool_output = step[1]
                if isinstance(tool_output, go.Figure):
                    st.plotly_chart(tool_output, use_container_width=True)
                    add_message("assistant", tool_output, "figure")
                    st.session_state.chart_gallery.append({
                        "chart": tool_output,
                        "title": f"Chart generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    })
                elif isinstance(tool_output, pd.DataFrame):
                    st.dataframe(tool_output, use_container_width=True)
                    add_message("assistant", tool_output, "dataframe")
            
            # 스트리밍된 텍스트를 최종 답변으로 사용, 중복 방지
            final_text = callback_handler.get_final_text()
            if final_text.strip():
                add_message("assistant", final_text, "text")
            else:
                add_message("assistant", final_answer, "text")
        except Exception as e:
            error_message = f"분석 중 오류 발생: {str(e)}"
            st.error(error_message)
            add_message("assistant", error_message)

def setup_sidebar():
    """사이드바에 API 키 입력과 파일 업로더를 설정합니다."""
    with st.sidebar:
        st.header("설정")
        st.warning("⚠️ 이 앱은 데이터를 안전하게 처리하지만, 민감한 데이터 업로드 시 주의하세요.")
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
            st.session_state.chart_gallery = []
            st.session_state.agent = None
            st.session_state.df_name = None
            st.rerun()

def main():
    """Streamlit 앱을 실행하는 메인 함수입니다."""
    init_session_state()
    st.set_page_config(
        page_title="🤖 AI CSV 분석 챗봇",
        page_icon="📊",
        layout="wide"
    )
    st.title("🤖 AI CSV 분석 챗봇 (v2.5)")
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

    tab1, tab2, tab3 = st.tabs(["💬 AI 챗봇", "📊 데이터 대시보드", "🖼️ 차트 갤러리"])
    
    with tab2:
        display_dashboard(st.session_state.df)
    
    with tab3:
        st.subheader("🖼️ 생성된 차트 모음")
        if not st.session_state.chart_gallery:
            st.info("아직 생성된 차트가 없습니다. AI 챗봇에게 시각화를 요청해보세요.")
        else:
            for item in st.session_state.chart_gallery:
                st.markdown(f"**{item['title']}**")
                st.plotly_chart(item["chart"], use_container_width=True)
                st.divider()
    
    with tab1:
        st.subheader("💬 AI에게 데이터에 대해 질문해보세요")
        st.markdown("리포트 작성에 다소 시간이 소요될 수 있습니다.")
        if st.button("🤖 AI 자동 리포트 생성"):
            auto_report_prompt = """
            ## 탐색적 데이터 분석(EDA) 리포트

            업로드된 데이터프레임 `df`에 대한 종합적인 탐색적 데이터 분석(EDA) 리포트를 생성해줘.

            리포트에는 다음이 반드시 포함되어야 해:
            - **데이터 요약**: 데이터 크기, 컬럼 수, 주요 통계치(df.describe())를 마크다운 테이블로 정리.
            - **핵심 인사이트**: 데이터에서 발견된 가장 중요한 인사이트 5가지를 번호 매겨 설명.
            - **상관관계 분석**: 수치형 변수들(df.select_dtypes(include=['number']))만 사용하여 상관관계 행렬을 계산하고, Plotly로 히트맵을 생성해. 문자열 데이터는 제외하여 오류를 방지해. 예: `df.select_dtypes(include=['number']).corr()`.
            - **회귀분석**: `score` 컬럼을 종속 변수로 하고, 나머지 수치형 변수들(df.select_dtypes(include=['number']).drop(columns=['score'], errors='ignore'))을 독립 변수로 사용하여 다중 선형 회귀분석을 수행해. `statsmodels` 또는 `scikit-learn`을 사용하고, 회귀 계수와 R² 값을 포함하며, Plotly로 산점도와 회귀선을 시각화해. 문자열 데이터는 제외하고, 결측치는 제거하거나 적절히 처리해.
            - **이상치 분석**: 주요 수치형 컬럼에 대해 Box Plot을 사용하여 이상치를 시각화해. `df.select_dtypes(include=['number'])`를 사용하고, 이상치가 발견되면 해당 컬럼과 값을 설명.
            - **결측치 및 데이터 품질**: 결측치(df.isnull().sum())와 데이터 타입 오류를 분석하고, 문제가 있다면 해결 방안을 제안.

            **중요**: 모든 분석은 pandas, numpy, plotly, statsmodels, scikit-learn 라이브러리를 사용하여 수행하고, 문자열 데이터는 반드시 제외해. 결측치가 있는 경우 `df.dropna()`로 처리하거나 적절히 대체해. 리포트는 한 번만 출력되도록 해.

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

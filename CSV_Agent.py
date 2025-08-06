import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from typing import List, Union, Dict, Any, Optional
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

# CSV 파일 로딩 캐싱
@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    """CSV 파일을 로드하고 캐싱합니다."""
    try:
        if file.size == 0:
            st.error("업로드된 CSV 파일이 비어 있습니다.")
            return None
        df = pd.read_csv(file)
        if df.empty:
            st.error("CSV 파일에 데이터가 없습니다.")
            return None
        return df
    except Exception as e:
        st.error(f"CSV 파일을 읽는 중 오류 발생: {str(e)}")
        return None

# 에이전트 생성 캐싱
@st.cache_resource(show_spinner=False)
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
        "last_message": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def generate_eda_report(df: pd.DataFrame) -> str:
    """EDA 리포트를 생성합니다."""
    report = "## 탐색적 데이터 분석(EDA) 리포트\n\n"

    # 데이터 요약
    report += "### 데이터 요약\n"
    report += f"- **데이터 크기**: {df.shape[0]:,} 행, {df.shape[1]} 열\n"
    buffer = pd.io.common.StringIO()
    df.info(buf=buffer)
    report += f"#### 데이터 정보\n```\n{buffer.getvalue()}\n```\n"

    # 주요 통계치
    stats = df.describe().transpose()
    report += "#### 주요 통계치\n"
    report += "| 컬럼 | 평균 | 표준편차 | 최소값 | 최대값 |\n"
    report += "|------|------|----------|--------|--------|\n"
    for col in stats.index:
        report += f"| {col} | {stats.loc[col, 'mean']:.2f} | {stats.loc[col, 'std']:.2f} | {stats.loc[col, 'min']:.2f} | {stats.loc[col, 'max']:.2f} |\n"

    # 결측치 분석
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    if not missing_data.empty:
        report += "\n### 결측치 분석\n"
        report += "| 컬럼 | 결측치 수 |\n"
        report += "|------|-----------|\n"
        for col, count in missing_data.items():
            report += f"| {col} | {count} |\n"
    else:
        report += "\n### 결측치 분석\n🎉 데이터에 결측치가 없습니다!\n"

    # 수치형 컬럼 분석
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        report += "\n### 상관관계 분석\n"
        corr = df[numeric_cols].corr()
        report += "수치형 변수 간 상관관계를 히트맵으로 시각화했습니다.\n"

        # 클러스터링 분석
        if len(numeric_cols) >= 2:
            report += "\n### 클러스터링 분석\n"
            report += "K-Means 클러스터링을 통해 데이터의 군집을 분석했습니다.\n"

        # 피처 중요도 분석
        if 'score' in df.columns:
            report += "\n### 피처 중요도 분석\n"
            report += "Random Forest를 사용하여 'score'에 대한 피처 중요도를 분석했습니다.\n"

    # 시계열 분석
    datetime_cols = df.select_dtypes(include=['datetime']).columns
    if len(datetime_cols) > 0 and len(numeric_cols) > 0:
        report += "\n### 시계열 분석\n"
        report += f"날짜 컬럼('{datetime_cols[0]}')을 기준으로 시계열 데이터를 분석했습니다.\n"

    return report

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
            template='plotly_white',
            color_discrete_sequence=['#1f77b4']
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
            color_discrete_sequence=['#1f77b4'],
            marginal="violin"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # KDE 플롯
        fig_kde = px.density_contour(
            df,
            x=selected_numeric,
            title=f"'{selected_numeric}' KDE 플롯",
            template='plotly_white',
            color_discrete_sequence=['#ff7f0e']
        )
        st.plotly_chart(fig_kde, use_container_width=True)

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
    if st.session_state.last_message != content:
        st.session_state.messages.append({
            "role": role,
            "content": content,
            "type": msg_type,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        st.session_state.last_message = content

def run_agent(query: str, display_prompt: bool = True, is_eda_report: bool = False):
    """주어진 쿼리로 에이전트를 실행하고 결과를 표시합니다."""
    if st.session_state.agent is None:
        st.error("에이전트가 초기화되지 않았습니다. API 키를 확인하고 파일을 다시 업로드해주세요.")
        return

    if is_eda_report:
        # EDA 리포트인 경우 기존 메시지 중복 제거
        st.session_state.messages = [msg for msg in st.session_state.messages if "탐색적 데이터 분석(EDA) 리포트" not in str(msg["content"])]

    if display_prompt:
        add_message("user", query, "text")
        with st.chat_message("user"):
            st.markdown(query)

    with st.chat_message("assistant"):
        container = st.empty()
        callback_handler = StreamlitCallbackHandler(container)
        try:
            with st.spinner("분석 중..."):
                response = st.session_state.agent.invoke(
                    {"input": query},
                    {"callbacks": [callback_handler]}
                )
            final_text = callback_handler.get_final_text()
            intermediate_steps = response.get("intermediate_steps", [])

            for step in intermediate_steps:
                tool_output = step[1]
                if isinstance(tool_output, go.Figure):
                    st.plotly_chart(tool_output, use_container_width=True)
                    add_message("assistant", tool_output, "figure")
                elif isinstance(tool_output, pd.DataFrame):
                    st.dataframe(tool_output, use_container_width=True)
                    add_message("assistant", tool_output, "dataframe")

            if final_text.strip():
                add_message("assistant", final_text, "text")
            else:
                st.error("분석 결과가 비어 있습니다.")
                add_message("assistant", "분석 결과가 비어 있습니다.", "text")

            # 추가적인 시각화 (EDA 리포트용)
            if is_eda_report:
                df = st.session_state.df
                numeric_cols = df.select_dtypes(include=['number']).columns

                # 상관관계 히트맵
                if len(numeric_cols) > 0:
                    corr = df[numeric_cols].corr()
                    fig_corr = px.imshow(
                        corr,
                        text_auto=True,
                        title="수치형 변수 상관관계 히트맵",
                        color_continuous_scale='Viridis',
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                    add_message("assistant", fig_corr, "figure")

                # 클러스터링
                if len(numeric_cols) >= 2:
                    with st.spinner("클러스터링 분석 중..."):
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(df[numeric_cols].dropna())
                        kmeans = KMeans(n_clusters=3, random_state=42)
                        clusters = kmeans.fit_predict(scaled_data)
                        df_cluster = df[numeric_cols].dropna().copy()
                        df_cluster['Cluster'] = clusters
                        fig_cluster = px.scatter(
                            df_cluster,
                            x=numeric_cols[0],
                            y=numeric_cols[1],
                            color='Cluster',
                            title="K-Means 클러스터링 결과",
                            template='plotly_white',
                            color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig_cluster, use_container_width=True)
                        add_message("assistant", fig_cluster, "figure")

                # 피처 중요도
                if 'score' in df.columns and len(numeric_cols) > 1:
                    X = df[numeric_cols].drop(columns=['score'], errors='ignore').dropna()
                    y = df['score'].loc[X.index]
                    rf = RandomForestRegressor(random_state=42)
                    rf.fit(X, y)
                    feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
                    fig_importance = px.bar(
                        feature_importance,
                        title="피처 중요도",
                        template='plotly_white',
                        color_discrete_sequence=['#1f77b4']
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                    add_message("assistant", fig_importance, "figure")

                # 시계열 분석
                datetime_cols = df.select_dtypes(include=['datetime']).columns
                if len(datetime_cols) > 0 and len(numeric_cols) > 0:
                    fig_time = px.line(
                        df,
                        x=datetime_cols[0],
                        y=numeric_cols[0],
                        title=f"'{numeric_cols[0]}'의 시계열 분석",
                        template='plotly_white',
                        color_discrete_sequence=['#ff7f0e']
                    )
                    st.plotly_chart(fig_time, use_container_width=True)
                    add_message("assistant", fig_time, "figure")

        except Exception as e:
            error_message = f"분석 중 오류 발생: {str(e)}"
            st.error(error_message)
            add_message("assistant", error_message, "text")

def setup_sidebar():
    """사이드바에 API 키 입력과 파일 업로더를 설정합니다."""
    with st.sidebar:
        st.header("설정")
        api_key = st.text_input("🔑 OpenAI API Key", type="password", key="api_key_input")
        if api_key and api_key != st.session_state.get("api_key"):
            st.session_state["api_key"] = api_key
            st.session_state.agent = None  # API 키 변경 시 에이전트 초기화
        
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
            selected_file = st.selectbox(
                "분석할 파일을 선택하세요.",
                options=file_names,
                index=file_names.index(st.session_state.selected_file) if st.session_state.selected_file in file_names else 0
            )
            if selected_file != st.session_state.selected_file:
                st.session_state.selected_file = selected_file
                st.session_state.df = st.session_state.uploaded_files[selected_file]
                st.session_state.agent = None  # 파일 변경 시 에이전트 초기화
                st.session_state.messages = []  # 메시지 초기화
                st.session_state.last_message = None
                # 이전 DataFrame 메모리 해제
                if st.session_state.df is not None:
                    del st.session_state.df
                st.session_state.df = st.session_state.uploaded_files[selected_file]
        
        if st.button("🔄️ 대화 초기화"):
            st.session_state.messages = []
            st.session_state.agent = None
            st.session_state.df_name = None
            st.session_state.last_message = None
            st.rerun()

def main():
    """Streamlit 앱을 실행하는 메인 함수입니다."""
    init_session_state()
    st.set_page_config(
        page_title="🤖 AI CSV 분석 챗봇",
        page_icon="📊",
        layout="wide"
    )
    st.title("🤖 AI CSV 분석 챗봇 (v2.8)")
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
            auto_report_prompt = generate_eda_report(st.session_state.df)
            run_agent(auto_report_prompt, display_prompt=False, is_eda_report=True)
        
        st.divider()
        display_chat_history()
        if prompt := st.chat_input("데이터에 대해 질문을 입력하세요..."):
            run_agent(prompt)

if __name__ == "__main__":
    main()

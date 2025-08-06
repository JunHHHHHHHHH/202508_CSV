from typing import List, Union, Dict, Any
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
import io
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import base64

# LangChain imports
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.tools import PythonAstREPLTool
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging
from langchain_teddynote.messages import AgentStreamParser, AgentCallbacks

warnings.filterwarnings('ignore')

# TeddyNote의 langsmith 로깅
logging.langsmith("Enhanced CSV Agent 챗봇")

# 페이지 설정
st.set_page_config(
    page_title="🚀 Advanced CSV Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일링
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .insight-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 제목
st.markdown("""
<div class="main-header">
    <h1>🚀 Advanced CSV Analytics Chatbot</h1>
    <p>AI-powered data analysis with automated insights and interactive visualizations</p>
</div>
""", unsafe_allow_html=True)

# 세션 상태 초기화
def initialize_session_state():
    defaults = {
        "messages": [],
        "analysis_history": [],
        "chart_gallery": [],
        "auto_insights": [],
        "processed_files": {},
        "current_analysis": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# 상수 정의
class MessageRole:
    USER = "user"
    ASSISTANT = "assistant"

class MessageType:
    TEXT = "text"
    FIGURE = "figure"
    CODE = "code"
    DATAFRAME = "dataframe"
    PLOTLY = "plotly"
    INSIGHT = "insight"

class AnalysisTemplate:
    EDA = "Exploratory Data Analysis"
    CORRELATION = "Correlation Analysis"
    TIME_SERIES = "Time Series Analysis"
    OUTLIER_DETECTION = "Outlier Detection"
    STATISTICAL_SUMMARY = "Statistical Summary"
    PREDICTIVE_MODELING = "Predictive Modeling"

# 데이터 분석 유틸리티 클래스
class DataAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        self.datetime_cols = []
        self._detect_datetime_columns()

    def _detect_datetime_columns(self):
        """날짜/시간 컬럼 자동 감지"""
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                try:
                    pd.to_datetime(self.df[col].head(100), errors='raise')
                    self.datetime_cols.append(col)
                except:
                    pass

    def get_basic_info(self) -> Dict[str, Any]:
        """기본 데이터 정보 반환"""
        return {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "memory_usage": self.df.memory_usage(deep=True).sum(),
            "numeric_columns": self.numeric_cols,
            "categorical_columns": self.categorical_cols,
            "datetime_columns": self.datetime_cols
        }

    def detect_outliers(self, method='iqr') -> Dict[str, List]:
        """이상치 탐지"""
        outliers = {}
        
        for col in self.numeric_cols:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_indices = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)].index.tolist()
                outliers[col] = outlier_indices
                
        return outliers

    def generate_correlation_insights(self) -> List[str]:
        """상관관계 인사이트 생성"""
        if len(self.numeric_cols) < 2:
            return ["수치형 변수가 부족하여 상관관계 분석을 수행할 수 없습니다."]
        
        corr_matrix = self.df[self.numeric_cols].corr()
        insights = []
        
        # 강한 상관관계 찾기
        for i, col1 in enumerate(self.numeric_cols):
            for j, col2 in enumerate(self.numeric_cols[i+1:], i+1):
                corr_val = corr_matrix.loc[col1, col2]
                if abs(corr_val) > 0.7:
                    insights.append(f"'{col1}'과 '{col2}' 간에 {'강한 양의' if corr_val > 0 else '강한 음의'} 상관관계 발견 (r={corr_val:.3f})")
        
        if not insights:
            insights.append("변수들 간에 특별히 강한 상관관계는 발견되지 않았습니다.")
            
        return insights

    def auto_visualization_suggestions(self) -> List[Dict[str, str]]:
        """데이터 타입에 따른 시각화 제안"""
        suggestions = []
        
        # 수치형 변수들
        if len(self.numeric_cols) >= 2:
            suggestions.append({
                "type": "scatter_matrix",
                "description": "수치형 변수들 간의 산점도 행렬",
                "columns": self.numeric_cols[:5]  # 최대 5개만
            })
            
        # 범주형 변수들
        for col in self.categorical_cols[:3]:  # 최대 3개만
            if self.df[col].nunique() <= 20:
                suggestions.append({
                    "type": "countplot",
                    "description": f"'{col}' 변수의 분포",
                    "column": col
                })
        
        # 시계열 데이터
        if self.datetime_cols and self.numeric_cols:
            suggestions.append({
                "type": "time_series",
                "description": "시계열 트렌드 분석",
                "datetime_col": self.datetime_cols[0],
                "numeric_col": self.numeric_cols[0]
            })
            
        return suggestions

# 시각화 생성기 클래스
class VisualizationGenerator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.analyzer = DataAnalyzer(df)

    def create_overview_dashboard(self) -> go.Figure:
        """데이터 개요 대시보드 생성"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Data Shape', 'Missing Values', 'Data Types', 'Memory Usage'),
            specs=[[{"type": "indicator"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "indicator"}]]
        )
        
        # 데이터 모양
        fig.add_trace(go.Indicator(
            mode="number",
            value=self.df.shape[0] * self.df.shape[1],
            title={"text": f"Total Cells<br>({self.df.shape[0]} × {self.df.shape[1]})"}
        ), row=1, col=1)
        
        # 결측값
        missing_data = self.df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if not missing_data.empty:
            fig.add_trace(go.Bar(
                x=missing_data.index,
                y=missing_data.values,
                name="Missing Values"
            ), row=1, col=2)
        
        # 데이터 타입
        dtype_counts = self.df.dtypes.value_counts()
        fig.add_trace(go.Pie(
            labels=dtype_counts.index.astype(str),
            values=dtype_counts.values,
            name="Data Types"
        ), row=2, col=1)
        
        # 메모리 사용량
        memory_mb = self.df.memory_usage(deep=True).sum() / 1024 / 1024
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=memory_mb,
            title={'text': "Memory Usage (MB)"},
            gauge={'axis': {'range': [None, memory_mb * 2]}}
        ), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False, title_text="📊 Data Overview Dashboard")
        return fig

    def create_correlation_heatmap(self) -> go.Figure:
        """상관관계 히트맵 생성"""
        if len(self.analyzer.numeric_cols) < 2:
            return None
            
        corr_matrix = self.df[self.analyzer.numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.around(corr_matrix.values, decimals=2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="🔥 Correlation Heatmap",
            xaxis_title="Variables",
            yaxis_title="Variables",
            height=500
        )
        
        return fig

    def create_distribution_plots(self) -> List[go.Figure]:
        """분포 플롯들 생성"""
        figures = []
        
        for col in self.analyzer.numeric_cols[:6]:  # 최대 6개
            fig = go.Figure()
            
            # 히스토그램
            fig.add_trace(go.Histogram(
                x=self.df[col],
                name='Distribution',
                opacity=0.7,
                nbinsx=30
            ))
            
            # 박스플롯 추가
            fig.add_trace(go.Box(
                y=self.df[col],
                name='Box Plot',
                yaxis='y2'
            ))
            
            fig.update_layout(
                title=f"📈 Distribution of {col}",
                xaxis_title=col,
                yaxis_title="Frequency",
                yaxis2=dict(overlaying='y', side='right', title='Values'),
                height=400
            )
            
            figures.append(fig)
            
        return figures

# 메시지 처리 함수들
def print_messages():
    """저장된 메시지들을 화면에 출력"""
    for role, content_list in st.session_state["messages"]:
        with st.chat_message(role):
            for content in content_list:
                if isinstance(content, list) and len(content) == 2:
                    message_type, message_content = content
                    if message_type == MessageType.TEXT:
                        st.markdown(message_content)
                    elif message_type == MessageType.FIGURE:
                        st.pyplot(message_content)
                    elif message_type == MessageType.CODE:
                        with st.expander("🔍 코드 보기", expanded=False):
                            st.code(message_content, language="python")
                    elif message_type == MessageType.DATAFRAME:
                        st.dataframe(message_content, use_container_width=True)
                    elif message_type == MessageType.PLOTLY:
                        st.plotly_chart(message_content, use_container_width=True)
                    elif message_type == MessageType.INSIGHT:
                        st.markdown(f"""
                        <div class="insight-box">
                            <h4>💡 AI Insight</h4>
                            <p>{message_content}</p>
                        </div>
                        """, unsafe_allow_html=True)

def add_message(role: MessageRole, content: List[Union[MessageType, str]]):
    """메시지 추가"""
    messages = st.session_state["messages"]
    if messages and messages[-1][0] == role:
        messages[-1][1].extend([content])
    else:
        messages.append([role, [content]])

# 자동 분석 함수들
def perform_auto_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """자동 데이터 분석 수행"""
    analyzer = DataAnalyzer(df)
    viz_gen = VisualizationGenerator(df)
    
    analysis_results = {
        "basic_info": analyzer.get_basic_info(),
        "outliers": analyzer.detect_outliers(),
        "correlation_insights": analyzer.generate_correlation_insights(),
        "viz_suggestions": analyzer.auto_visualization_suggestions(),
        "overview_dashboard": viz_gen.create_overview_dashboard(),
        "correlation_heatmap": viz_gen.create_correlation_heatmap(),
        "distribution_plots": viz_gen.create_distribution_plots()
    }
    
    return analysis_results

def generate_auto_insights(df: pd.DataFrame) -> List[str]:
    """AI 자동 인사이트 생성"""
    insights = []
    analyzer = DataAnalyzer(df)
    
    # 기본 통계 인사이트
    insights.append(f"📊 데이터셋은 {df.shape[0]:,}개의 행과 {df.shape[1]}개의 열로 구성되어 있습니다.")
    
    # 결측값 인사이트
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        insights.append(f"⚠️ 총 {missing_count:,}개의 결측값이 발견되었습니다. 데이터 전처리가 필요할 수 있습니다.")
    else:
        insights.append("✅ 결측값이 없는 깨끗한 데이터입니다.")
    
    # 수치형 변수 인사이트
    if analyzer.numeric_cols:
        insights.append(f"🔢 {len(analyzer.numeric_cols)}개의 수치형 변수가 있어 통계 분석과 예측 모델링이 가능합니다.")
        
        # 왜도 분석
        for col in analyzer.numeric_cols[:3]:
            skewness = df[col].skew()
            if abs(skewness) > 1:
                insights.append(f"📈 '{col}' 변수는 {'우편향' if skewness > 0 else '좌편향'}된 분포를 보입니다 (왜도: {skewness:.2f})")
    
    # 범주형 변수 인사이트
    if analyzer.categorical_cols:
        high_cardinality_cols = [col for col in analyzer.categorical_cols if df[col].nunique() > 50]
        if high_cardinality_cols:
            insights.append(f"🏷️ {len(high_cardinality_cols)}개의 변수가 높은 cardinality를 가집니다: {', '.join(high_cardinality_cols[:3])}")
    
    return insights

# 콜백 함수들
def tool_callback(tool) -> None:
    """도구 실행 콜백"""
    if tool_name := tool.get("tool"):
        if tool_name == "python_repl_ast":
            tool_input = tool.get("tool_input", {})
            query = tool_input.get("query")
            if query:
                with st.status("🔍 분석 중...", expanded=True) as status:
                    add_message(MessageRole.ASSISTANT, [MessageType.CODE, query])

                    if "df" in st.session_state:
                        # 환경 설정
                        st.session_state["python_tool"].locals.update({
                            "pd": pd, "sns": sns, "plt": plt, "px": px, "go": go,
                            "np": np, "st": st
                        })
                        
                        try:
                            result = st.session_state["python_tool"].invoke({"query": query})
                            
                            # 결과 처리
                            if isinstance(result, pd.DataFrame):
                                st.dataframe(result, use_container_width=True)
                                add_message(MessageRole.ASSISTANT, [MessageType.DATAFRAME, result])
                            
                            # 시각화 처리
                            if "plt.show" in query or "st.pyplot" in query:
                                fig = plt.gcf()
                                st.pyplot(fig)
                                add_message(MessageRole.ASSISTANT, [MessageType.FIGURE, fig])
                                # 차트 갤러리에 추가
                                st.session_state["chart_gallery"].append({
                                    "type": "matplotlib",
                                    "figure": fig,
                                    "timestamp": datetime.now(),
                                    "query": query
                                })
                                
                            status.update(label="✅ 분석 완료", state="complete", expanded=False)
                            
                        except Exception as e:
                            st.error(f"❌ 오류 발생: {e}")
                            return
                    else:
                        st.error("❌ 데이터프레임이 정의되지 않았습니다. CSV 파일을 먼저 업로드해주세요.")

def observation_callback(observation) -> None:
    """관찰 결과 콜백"""
    if "observation" in observation:
        obs = observation["observation"]
        if isinstance(obs, str) and "Error" in obs:
            st.error(f"❌ {obs}")

def result_callback(result: str) -> None:
    """결과 콜백"""
    pass

def create_agent(dataframe, selected_model="gpt-4o-mini"):
    """강화된 에이전트 생성"""
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        st.error("❌ OpenAI API 키를 입력해주세요")
        return None
        
    enhanced_prefix = """
    You are an expert data scientist and analyst with deep knowledge in:
    - Statistical analysis and hypothesis testing
    - Data visualization best practices
    - Machine learning and predictive modeling
    - Data preprocessing and cleaning techniques
    
    Available libraries: pandas, numpy, matplotlib, seaborn, plotly, sklearn, scipy
    
    CRITICAL INSTRUCTIONS:
    1. Always use 'df' as the main DataFrame variable (DO NOT recreate it)
    2. For matplotlib: Use fig, ax = plt.subplots() and st.pyplot(fig)
    3. For plotly: Use st.plotly_chart(fig, use_container_width=True)
    4. Prefer seaborn and plotly for modern, beautiful visualizations
    5. Use English for plot titles and labels
    6. Apply consistent styling: white background, muted colors, no grid
    7. Always explain your analysis approach and findings
    8. Suggest next steps or additional analyses when appropriate
    
    Dataset info:
    - Shape: {shape}
    - Columns: {columns}
    - Numeric columns: {numeric_cols}
    - Categorical columns: {cat_cols}
    
    Respond in Korean for explanations, but use English for code comments and plot labels.
    """.format(
        shape=dataframe.shape,
        columns=list(dataframe.columns),
        numeric_cols=dataframe.select_dtypes(include=[np.number]).columns.tolist(),
        cat_cols=dataframe.select_dtypes(include=['object']).columns.tolist()
    )
    
    return create_pandas_dataframe_agent(
        ChatOpenAI(model=selected_model, temperature=0.1, api_key=openai_key),
        dataframe,
        verbose=False,
        agent_type="tool-calling",
        allow_dangerous_code=True,
        prefix=enhanced_prefix,
        max_iterations=5,
        early_stopping_method="generate"
    )

def ask(query):
    """질문 처리"""
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
            
            if ai_answer:
                st.write(ai_answer)
                add_message(MessageRole.ASSISTANT, [MessageType.TEXT, ai_answer])

# 사이드바 구성
with st.sidebar:
    st.markdown("## 🔧 설정")
    
    # API 키 입력
    st.markdown("### 🔑 OpenAI API Key")
    user_api_key = st.text_input("API Key", type="password", help="OpenAI API 키를 입력하세요")
    
    if user_api_key:
        os.environ["OPENAI_API_KEY"] = user_api_key
        st.success("✅ API 키가 설정되었습니다")
    
    st.markdown("---")
    
    # 파일 업로드
    st.markdown("### 📁 파일 업로드")
    uploaded_files = st.file_uploader(
        "CSV 파일 업로드", 
        type=["csv"], 
        accept_multiple_files=True,
        help="여러 파일을 동시에 업로드할 수 있습니다"
    )
    
    # 분석 템플릿 선택
    st.markdown("### 📋 분석 템플릿")
    selected_template = st.selectbox(
        "분석 유형 선택",
        [
            AnalysisTemplate.EDA,
            AnalysisTemplate.CORRELATION,
            AnalysisTemplate.TIME_SERIES,
            AnalysisTemplate.OUTLIER_DETECTION,
            AnalysisTemplate.STATISTICAL_SUMMARY,
            AnalysisTemplate.PREDICTIVE_MODELING
        ]
    )
    
    # 버튼들
    col1, col2 = st.columns(2)
    with col1:
        apply_btn = st.button("🚀 분석 시작", type="primary")
    with col2:
        clear_btn = st.button("🗑️ 초기화")
    
    auto_analysis_btn = st.button("🤖 자동 분석", help="AI가 자동으로 데이터를 분석합니다")
    
    st.markdown("---")
    
    # 차트 갤러리
    if st.session_state["chart_gallery"]:
        st.markdown("### 🖼️ 차트 갤러리")
        st.write(f"생성된 차트: {len(st.session_state['chart_gallery'])}개")
        
        if st.button("갤러리 보기"):
            st.session_state["show_gallery"] = True

# 메인 콘텐츠 영역
if clear_btn:
    st.session_state["messages"] = []
    st.session_state["chart_gallery"] = []
    st.session_state["auto_insights"] = []
    st.rerun()

# 파일 업로드 및 분석 시작
if apply_btn:
    if not user_api_key:
        st.warning("⚠️ OpenAI API 키를 입력해주세요.")
    elif not uploaded_files:
        st.warning("⚠️ CSV 파일을 업로드해주세요.")
    else:
        # 다중 파일 처리
        for uploaded_file in uploaded_files:
            try:
                df = pd.read_csv(uploaded_file)
                file_name = uploaded_file.name
                
                st.session_state["processed_files"][file_name] = df
                
                # 메인 DataFrame 설정 (첫 번째 파일)
                if len(st.session_state["processed_files"]) == 1:
                    st.session_state["df"] = df
                    st.session_state["python_tool"] = PythonAstREPLTool()
                    st.session_state["python_tool"].locals.update({
                        "df": df, "pd": pd, "sns": sns, "plt": plt, 
                        "px": px, "go": go, "np": np
                    })
                    st.session_state["agent"] = create_agent(df)
                
                st.success(f"✅ '{file_name}' 파일이 성공적으로 로드되었습니다!")
                
            except Exception as e:
                st.error(f"❌ '{uploaded_file.name}' 파일 로드 중 오류: {e}")

# 자동 분석 수행
if auto_analysis_btn and "df" in st.session_state:
    with st.spinner("🤖 AI가 데이터를 자동 분석 중입니다..."):
        df = st.session_state["df"]
        
        # 자동 분석 수행
        analysis_results = perform_auto_analysis(df)
        auto_insights = generate_auto_insights(df)
        
        # 결과 표시
        st.markdown("## 🤖 AI 자동 분석 결과")
        
        # 기본 정보 대시보드
        if analysis_results["overview_dashboard"]:
            st.plotly_chart(analysis_results["overview_dashboard"], use_container_width=True)
            add_message(MessageRole.ASSISTANT, [MessageType.PLOTLY, analysis_results["overview_dashboard"]])
        
        # 자동 인사이트
        st.markdown("### 💡 핵심 인사이트")
        for insight in auto_insights:
            st.markdown(f"- {insight}")
            add_message(MessageRole.ASSISTANT, [MessageType.INSIGHT, insight])
        
        # 상관관계 히트맵
        if analysis_results["correlation_heatmap"]:
            st.markdown("### 🔥 변수 간 상관관계")
            st.plotly_chart(analysis_results["correlation_heatmap"], use_container_width=True)
            add_message(MessageRole.ASSISTANT, [MessageType.PLOTLY, analysis_results["correlation_heatmap"]])
        
        # 상관관계 인사이트
        st.markdown("### 📊 상관관계 분석")
        for insight in analysis_results["correlation_insights"]:
            st.markdown(f"- {insight}")
        
        # 분포 플롯들
        if analysis_results["distribution_plots"]:
            st.markdown("### 📈 변수 분포 분석")
            cols = st.columns(2)
            for i, fig in enumerate(analysis_results["distribution_plots"][:4]):  # 최대 4개만
                with cols[i % 2]:
                    st.plotly_chart(fig, use_container_width=True)
                    add_message(MessageRole.ASSISTANT, [MessageType.PLOTLY, fig])
        
        # 이상치 탐지 결과
        outliers = analysis_results["outliers"]
        if any(len(indices) > 0 for indices in outliers.values()):
            st.markdown("### ⚠️ 이상치 탐지 결과")
            outlier_summary = []
            for col, indices in outliers.items():
                if len(indices) > 0:
                    outlier_summary.append(f"'{col}': {len(indices)}개 이상치 발견")
            
            for summary in outlier_summary:
                st.markdown(f"- {summary}")
        
        # 시각화 제안
        st.markdown("### 🎨 추천 시각화")
        viz_suggestions = analysis_results["viz_suggestions"]
        for suggestion in viz_suggestions[:3]:  # 최대 3개만
            st.markdown(f"- **{suggestion['type']}**: {suggestion['description']}")
        
        st.session_state["current_analysis"] = analysis_results
        st.success("✅ 자동 분석이 완료되었습니다! 이제 채팅으로 더 자세한 분석을 요청해보세요.")

# 템플릿 기반 분석 실행
def execute_template_analysis(template: str, df: pd.DataFrame):
    """템플릿 기반 분석 실행"""
    if template == AnalysisTemplate.EDA:
        query = """
        데이터셋에 대한 완전한 탐색적 데이터 분석(EDA)을 수행해주세요. 
        다음을 포함해주세요:
        1. 기본 통계 요약
        2. 결측값 분석
        3. 데이터 분포 시각화
        4. 주요 변수들의 관계 분석
        """
    elif template == AnalysisTemplate.CORRELATION:
        query = """
        변수들 간의 상관관계를 분석해주세요:
        1. 상관관계 매트릭스 히트맵 생성
        2. 강한 상관관계를 가진 변수 쌍 식별
        3. 상관관계의 실무적 의미 해석
        """
    elif template == AnalysisTemplate.TIME_SERIES:
        query = """
        시계열 데이터 분석을 수행해주세요:
        1. 시간에 따른 트렌드 분석
        2. 계절성 패턴 탐지
        3. 이상치 및 변화점 식별
        """
    elif template == AnalysisTemplate.OUTLIER_DETECTION:
        query = """
        데이터의 이상치를 탐지하고 분석해주세요:
        1. 통계적 방법으로 이상치 식별
        2. 이상치 시각화
        3. 이상치가 분석에 미치는 영향 평가
        """
    elif template == AnalysisTemplate.STATISTICAL_SUMMARY:
        query = """
        상세한 통계 요약을 제공해주세요:
        1. 기술통계량 계산
        2. 분포의 정규성 검정
        3. 변수별 특성 분석
        """
    elif template == AnalysisTemplate.PREDICTIVE_MODELING:
        query = """
        예측 모델링을 수행해주세요:
        1. 적절한 타겟 변수 식별
        2. 간단한 예측 모델 구축
        3. 모델 성능 평가
        """
    else:
        query = "데이터에 대한 일반적인 분석을 수행해주세요."
    
    return query

# 차트 갤러리 표시
if st.session_state.get("show_gallery", False):
    st.markdown("## 🖼️ 차트 갤러리")
    
    if st.session_state["chart_gallery"]:
        cols = st.columns(2)
        for i, chart_info in enumerate(st.session_state["chart_gallery"]):
            with cols[i % 2]:
                st.markdown(f"**생성 시간**: {chart_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                if chart_info["type"] == "matplotlib":
                    st.pyplot(chart_info["figure"])
                with st.expander("코드 보기"):
                    st.code(chart_info["query"], language="python")
                st.markdown("---")
    else:
        st.info("아직 생성된 차트가 없습니다.")
    
    if st.button("갤러리 닫기"):
        st.session_state["show_gallery"] = False
        st.rerun()

# 메인 채팅 인터페이스
st.markdown("## 💬 데이터 분석 채팅")

# 빠른 질문 버튼들
if "df" in st.session_state:
    st.markdown("### 🚀 빠른 분석")
    quick_questions = [
        "데이터 요약 통계를 보여주세요",
        "결측값이 있는 컬럼들을 확인해주세요", 
        "수치형 변수들의 분포를 시각화해주세요",
        "범주형 변수들의 빈도를 분석해주세요",
        "변수들 간의 상관관계를 분석해주세요"
    ]
    
    cols = st.columns(3)
    for i, question in enumerate(quick_questions):
        with cols[i % 3]:
            if st.button(question, key=f"quick_{i}"):
                ask(question)

# 메시지 출력
print_messages()

# 사용자 입력
user_input = st.chat_input("데이터에 대해 궁금한 것을 물어보세요! 예: '매출 데이터의 트렌드를 분석해주세요'")

if user_input:
    ask(user_input)

# 템플릿 분석 실행
if "df" in st.session_state and selected_template:
    template_query = execute_template_analysis(selected_template, st.session_state["df"])
    
    if st.sidebar.button("템플릿 분석 실행"):
        ask(template_query)

# 푸터
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8em;">
    🚀 Enhanced CSV Analytics Chatbot | Powered by OpenAI GPT-4 & Streamlit<br>
    💡 AI 자동 분석, 인터랙티브 시각화, 스마트 인사이트 제공
</div>
""", unsafe_allow_html=True)

# 성능 모니터링 (선택사항)
if st.session_state.get("df") is not None:
    with st.expander("📊 데이터셋 정보"):
        df = st.session_state["df"]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("행 수", f"{df.shape[0]:,}")
        with col2:
            st.metric("열 수", df.shape[1])
        with col3:
            st.metric("메모리 사용량", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        with col4:
            st.metric("결측값", f"{df.isnull().sum().sum():,}")
        
        # 컬럼 정보
        st.markdown("**컬럼 정보**")
        col_info = pd.DataFrame({
            '컬럼명': df.columns,
            '데이터타입': df.dtypes.values,
            '결측값': df.isnull().sum().values,
            '고유값수': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info, use_container_width=True)

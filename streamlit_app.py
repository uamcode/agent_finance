"""
한국 주식 AI 에이전트 - Streamlit 프론트엔드

LangSmith와 통합되어 에이전트 실행을 모니터링하고,
사용자 친화적인 채팅 인터페이스를 제공합니다.
"""

import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import json

# 현재 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 환경변수 로드
load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="한국 주식 AI 에이전트",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #E3F2FD;
    }
    .assistant-message {
        background-color: #F5F5F5;
    }
    .status-box {
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
    }
    .status-success {
        background-color: #C8E6C9;
        color: #2E7D32;
    }
    .status-error {
        background-color: #FFCDD2;
        color: #C62828;
    }
    .status-info {
        background-color: #BBDEFB;
        color: #1565C0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_agent():
    """에이전트를 로드합니다 (캐싱)"""
    try:
        from src.agent import agent, model_name, RAG_AVAILABLE, create_initial_state
        return agent, model_name, RAG_AVAILABLE, create_initial_state
    except Exception as e:
        st.error(f"에이전트 로드 실패: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, False, None


def create_state_with_history(user_input: str, create_initial_state_func) -> dict:
    """
    이전 대화 히스토리를 포함한 state 생성
    
    Args:
        user_input: 새로운 사용자 입력
        create_initial_state_func: 기본 state 생성 함수
        
    Returns:
        대화 히스토리가 포함된 AgentState
    """
    from langchain_core.messages import HumanMessage, AIMessage
    
    # 1. 기본 state 생성
    initial_state = create_initial_state_func(user_input)
    
    # 2. 이전 대화가 있으면 히스토리 추가
    if st.session_state.messages:
        history_messages = []
        
        # UI 메시지를 LangChain 메시지로 변환
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                history_messages.append(HumanMessage(content=msg["content"]))
            else:
                history_messages.append(AIMessage(content=msg["content"]))
        
        # 새 질문 추가
        history_messages.append(HumanMessage(content=user_input))
        
        # messages 덮어쓰기 (히스토리 포함)
        initial_state["messages"] = history_messages
    
    return initial_state


def initialize_session_state():
    """세션 상태를 초기화합니다"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "trace_url" not in st.session_state:
        st.session_state.trace_url = None
    
    if "agent_stats" not in st.session_state:
        st.session_state.agent_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0
        }


def get_langsmith_trace_url():
    """LangSmith 트레이스 URL을 생성합니다"""
    langsmith_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    project_name = os.getenv("LANGCHAIN_PROJECT", "Naver-Stock-Agent")
    
    if langsmith_enabled:
        return f"https://smith.langchain.com/projects/{project_name}"
    return None


def display_message(message, role="user"):
    """메시지를 표시합니다"""
    if role == "user":
        with st.chat_message("user"):
            st.markdown(message)
    else:
        with st.chat_message("assistant"):
            st.markdown(message)


def format_agent_response(response):
    """에이전트 응답을 포맷팅합니다"""
    if not response or "messages" not in response:
        return "응답을 받지 못했습니다."
    
    # 마지막 메시지 추출
    messages = response["messages"]
    if not messages:
        return "응답이 비어있습니다."
    
    last_message = messages[-1]
    
    # 메시지 내용 추출
    if hasattr(last_message, "content"):
        content = last_message.content
    elif isinstance(last_message, dict) and "content" in last_message:
        content = last_message["content"]
    else:
        content = str(last_message)
    
    return content


def extract_agent_steps(response):
    """에이전트 실행 단계를 추출합니다"""
    if not response or "messages" not in response:
        return []
    
    steps = []
    for msg in response["messages"]:
        if hasattr(msg, "name") and msg.name:
            steps.append({
                "agent": msg.name,
                "content": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            })
    
    return steps


def display_sidebar():
    """사이드바를 표시합니다"""
    with st.sidebar:
        st.markdown("### 한국 주식 AI 에이전트")
        st.markdown("---")
        
        # 모델 정보
        agent, model_name, rag_available, create_initial_state = load_agent()
        if agent:
            st.success("에이전트 로드 완료")
            st.info(f"**모델**: {model_name}")
            st.info(f"**RAG 사용 가능**: {'예' if rag_available else '아니오'}")
        else:
            st.error("에이전트 로드 실패")
        
        st.markdown("---")
        
        # LangSmith 정보
        langsmith_url = get_langsmith_trace_url()
        if langsmith_url:
            st.success("LangSmith 추적 활성화")
            st.markdown(f"[LangSmith 대시보드 열기]({langsmith_url})")
        else:
            st.warning("LangSmith 비활성화")
        
        st.markdown("---")
        
        # 통계
        st.markdown("### 사용 통계")
        stats = st.session_state.agent_stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("총 질의", stats["total_queries"])
            st.metric("성공", stats["successful_queries"])
        with col2:
            st.metric("실패", stats["failed_queries"])
            success_rate = (stats["successful_queries"] / stats["total_queries"] * 100) if stats["total_queries"] > 0 else 0
            st.metric("성공률", f"{success_rate:.1f}%")
        
        st.markdown("---")
        
        # 대화 히스토리 상태
        st.markdown("### 💬 대화 상태")
        message_count = len(st.session_state.messages)
        if message_count > 0:
            st.success(f"대화 히스토리: {message_count}개 메시지")
            st.info("💡 이전 대화 맥락이 유지됩니다")
        else:
            st.info("새로운 대화를 시작하세요")
        
        # 대화 히스토리 초기화
        if st.button("🔄 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        
        # 도움말
        with st.expander("사용 방법"):
            st.markdown("""
            **질문 예시:**
            - 삼성전자의 최근 종가를 알려줘
            - 거래량이 많은 상위 10개 종목은?
            - RSI가 30 이하인 종목을 찾아줘
            - 골든크로스가 뭐야?
            
            **팁:**
            - 구체적인 날짜를 명시하면 더 정확한 답변을 받을 수 있습니다
            - 기술적 분석 용어는 RAG 시스템이 설명해줍니다
            - LangSmith 대시보드에서 에이전트 실행 과정을 확인할 수 있습니다
            """)
        
        with st.expander("설정"):
            st.markdown("""
            **환경변수 (.env 파일):**
            - `CLOVASTUDIO_API_KEY`: ClovaX API 키
            - `OPENAI_API_KEY`: OpenAI API 키
            - `LANGCHAIN_TRACING_V2`: LangSmith 추적 활성화
            - `LANGCHAIN_API_KEY`: LangSmith API 키
            - `LANGCHAIN_PROJECT`: LangSmith 프로젝트 이름
            """)


def main():
    """메인 함수"""
    # 세션 상태 초기화
    initialize_session_state()
    
    # 사이드바 표시
    display_sidebar()
    
    # 헤더
    st.markdown('<div class="main-header">한국 주식 AI 에이전트</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">한국 주식 데이터를 자연어로 조회하세요</div>', unsafe_allow_html=True)
    
    # 에이전트 로드
    agent, model_name, rag_available, create_initial_state = load_agent()
    
    if not agent:
        st.error("에이전트를 로드할 수 없습니다. .env 파일과 API 키를 확인하세요.")
        return
    
    # 대화 히스토리 표시
    for message in st.session_state.messages:
        display_message(message["content"], message["role"])
    
    # 사용자 입력
    if user_input := st.chat_input("질문을 입력하세요 (예: 삼성전자의 최근 종가는?)"):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": user_input})
        display_message(user_input, "user")
        
        # 에이전트 실행
        with st.chat_message("assistant"):
            with st.status("생각 중...", expanded=True) as status:
                try:
                    st.write("질문 분석 중...")
                    
                    # 에이전트 실행 (대화 히스토리 포함)
                    initial_state = create_state_with_history(user_input, create_initial_state)
                    response = agent.invoke(initial_state)
                    
                    st.write("답변 생성 완료")
                    
                    # 에이전트 단계 표시
                    steps = extract_agent_steps(response)
                    if steps:
                        st.write(f"실행된 에이전트: {len(steps)}개")
                        for step in steps[-3:]:  # 마지막 3개만 표시
                            st.write(f"  - {step['agent']}")
                    
                    status.update(label="완료!", state="complete", expanded=False)
                    
                except Exception as e:
                    status.update(label="오류 발생", state="error", expanded=True)
                    error_message = f"죄송합니다. 오류가 발생했습니다:\n\n```\n{str(e)}\n```"
                    st.error(error_message)
                    
                    # 세션에 저장
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                    
                    # 통계 업데이트
                    st.session_state.agent_stats["total_queries"] += 1
                    st.session_state.agent_stats["failed_queries"] += 1
                    
                    # LangSmith 트레이스 링크 (디버깅용)
                    langsmith_url = get_langsmith_trace_url()
                    if langsmith_url:
                        st.info(f"[오류 추적 보기]({langsmith_url})")
                    return
            
            # status 블록 밖에서 답변 표시 (성공한 경우만)
            try:
                # 응답 포맷팅
                answer = format_agent_response(response)
                
                # 답변 표시
                st.markdown(answer)
                
                # 세션에 저장
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # 통계 업데이트
                st.session_state.agent_stats["total_queries"] += 1
                st.session_state.agent_stats["successful_queries"] += 1
                
                # LangSmith 트레이스 링크 표시
                langsmith_url = get_langsmith_trace_url()
                if langsmith_url:
                    st.info(f"[이 대화의 상세 추적 보기]({langsmith_url})")
            except:
                pass  # 에러는 이미 위에서 처리됨
    
    # 하단 푸터
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**데이터 소스**")
        st.text("KOSPI/KOSDAQ 주식 데이터")
    with col2:
        st.markdown("**AI 모델**")
        st.text(model_name if agent else "N/A")
    with col3:
        st.markdown("**모니터링**")
        st.text("LangSmith" if get_langsmith_trace_url() else "비활성화")


if __name__ == "__main__":
    main()

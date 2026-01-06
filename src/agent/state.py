"""
AgentState 정의

멀티 에이전트 시스템의 공유 상태를 정의합니다.
"""

from typing import Sequence, Optional, Any, Dict, List
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    멀티 에이전트 시스템의 공유 상태
    
    모든 에이전트는 이 상태를 읽고 업데이트합니다.
    """
    # 핵심 메시지
    messages: Sequence[BaseMessage]
    
    # SQL 관련
    sql_query: Optional[str]  # 생성된 SQL 쿼리
    query_results: Optional[str]  # DB 실행 결과
    tables_used: List[str]  # 사용된 테이블 목록
    
    # RAG 관련
    rag_context: Optional[str]  # RAG로 검색된 컨텍스트
    rag_used: bool  # RAG 사용 여부
    
    # 재시도 및 에러
    retry_count: int  # 현재 재시도 횟수
    max_retries: int  # 최대 재시도 횟수
    error_history: List[Dict[str, Any]]  # 에러 히스토리
    last_error: Optional[Dict[str, Any]]  # 마지막 에러
    
    # 실행 메타데이터
    session_id: Optional[str]  # 세션 ID
    start_time: Optional[str]  # 시작 시간
    agent_trace: List[str]  # 실행된 에이전트 목록
    routing_decision: Optional[str]  # 라우팅 결정 (SQL/RAG)
    
    # 사용자 요청
    original_query: str  # 원본 사용자 질문
    interpreted_query: Optional[str]  # 해석된 질문



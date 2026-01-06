"""
유틸리티 함수

State 초기화 및 기타 유틸리티 함수를 제공합니다.
"""

from datetime import datetime
from langchain_core.messages import HumanMessage
from .state import AgentState
from .config import MAX_RETRIES


def create_initial_state(user_query: str) -> AgentState:
    """
    AgentState를 초기화하여 반환
    
    Args:
        user_query: 사용자의 질문
        
    Returns:
        초기화된 AgentState
    """
    return {
        "messages": [HumanMessage(content=user_query)],
        "sql_query": None,
        "query_results": None,
        "tables_used": [],
        "rag_context": None,
        "rag_used": False,
        "retry_count": 0,
        "max_retries": MAX_RETRIES,
        "error_history": [],
        "last_error": None,
        "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "start_time": datetime.now().isoformat(),
        "agent_trace": [],
        "routing_decision": None,
        "original_query": user_query,
        "interpreted_query": None
    }



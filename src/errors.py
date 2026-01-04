"""
에이전트 에러 타입 정의
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel
import datetime


class ErrorSeverity(str, Enum):
    """에러 심각도"""
    LOW = "low"           # 재시도 가능
    MEDIUM = "medium"     # 재시도 제한적
    HIGH = "high"         # 즉시 중단
    CRITICAL = "critical" # 시스템 에러


class ErrorType(str, Enum):
    """에러 유형"""
    SQL_SYNTAX_ERROR = "sql_syntax_error"
    SQL_EXECUTION_ERROR = "sql_execution_error"
    TOOL_CALL_ERROR = "tool_call_error"
    LLM_API_ERROR = "llm_api_error"
    RAG_RETRIEVAL_ERROR = "rag_retrieval_error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"


class AgentError(BaseModel):
    """구조화된 에러 정보"""
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    agent_name: Optional[str] = None
    retry_count: int = 0
    original_error: Optional[str] = None
    timestamp: str


def classify_error(error: Exception, agent_name: str) -> AgentError:
    """에러를 분류하고 구조화"""
    error_str = str(error).lower()
    
    # SQL 에러
    if "syntax error" in error_str or "sql" in error_str:
        return AgentError(
            error_type=ErrorType.SQL_SYNTAX_ERROR,
            severity=ErrorSeverity.MEDIUM,
            message=f"SQL 쿼리 오류: {str(error)}",
            agent_name=agent_name,
            original_error=str(error),
            timestamp=datetime.datetime.now().isoformat()
        )
    
    # LLM API 에러
    elif "api" in error_str or "rate limit" in error_str:
        return AgentError(
            error_type=ErrorType.LLM_API_ERROR,
            severity=ErrorSeverity.HIGH,
            message=f"LLM API 오류: {str(error)}",
            agent_name=agent_name,
            original_error=str(error),
            timestamp=datetime.datetime.now().isoformat()
        )
    
    # 타임아웃
    elif "timeout" in error_str:
        return AgentError(
            error_type=ErrorType.TIMEOUT_ERROR,
            severity=ErrorSeverity.MEDIUM,
            message=f"실행 시간 초과: {str(error)}",
            agent_name=agent_name,
            original_error=str(error),
            timestamp=datetime.datetime.now().isoformat()
        )
    
    # 기타
    else:
        return AgentError(
            error_type=ErrorType.UNKNOWN_ERROR,
            severity=ErrorSeverity.MEDIUM,
            message=f"알 수 없는 오류: {str(error)}",
            agent_name=agent_name,
            original_error=str(error),
            timestamp=datetime.datetime.now().isoformat()
        )


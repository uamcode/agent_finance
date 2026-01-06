"""
도구 에러 처리

도구 실행 중 발생하는 에러를 처리합니다.
"""

from typing import Any
from langchain_core.messages import AIMessage, ToolMessage
from ..state import AgentState
from ..config import MAX_RETRIES
from ...errors import classify_error, ErrorSeverity
from ...logger import logger


def handle_tool_error(state: AgentState) -> dict:
    """
    개선된 에러 핸들러: 재시도 횟수 제한 + 에러 분류
    """
    error = state.get('error')
    retry_count = state.get('retry_count', 0)
    max_retries = state.get('max_retries', MAX_RETRIES)
    
    # 에러 분류
    agent_name = "unknown"
    if state.get('messages') and len(state['messages']) > 0:
        last_msg = state['messages'][-1]
        if hasattr(last_msg, 'name') and last_msg.name:
            agent_name = last_msg.name
    
    classified_error = classify_error(error, agent_name)
    
    # 에러 로깅
    logger.error(
        f"Tool error in {agent_name}",
        extra={
            "agent_name": agent_name,
            "error_type": classified_error.error_type,
            "severity": classified_error.severity,
            "retry_count": retry_count,
            "max_retries": max_retries
        }
    )
    
    # 재시도 가능 여부 판단
    if retry_count >= max_retries:
        # 최대 재시도 초과
        return {
            'messages': [
                AIMessage(
                    content=f"죄송합니다. 여러 번 시도했지만 작업을 완료할 수 없습니다.\n\n"
                            f"오류 유형: {classified_error.error_type}\n"
                            f"오류 메시지: {classified_error.message}\n\n"
                            f"다른 방식으로 질문해 주시겠어요?"
                )
            ]
        }
    
    # 심각도에 따른 처리
    if classified_error.severity == ErrorSeverity.CRITICAL:
        # 즉시 중단
        return {
            'messages': [
                AIMessage(
                    content=f"시스템 오류가 발생했습니다: {classified_error.message}\n"
                            f"관리자에게 문의해주세요."
                )
            ]
        }
    
    # 재시도
    tool_calls = []
    if state.get('messages') and len(state['messages']) > 0:
        last_msg = state['messages'][-1]
        if hasattr(last_msg, 'tool_calls'):
            tool_calls = last_msg.tool_calls
    
    return {
        'messages': [
            ToolMessage(
                content=f'오류가 발생했습니다 (재시도 {retry_count + 1}/{max_retries}):\n'
                        f'{classified_error.message}\n\n'
                        f'다른 방법으로 시도해주세요.',
                tool_call_id=tc['id'],
            )
            for tc in tool_calls
        ],
        'retry_count': retry_count + 1,
        'error_history': state.get('error_history', []) + [classified_error.dict()],
        'last_error': classified_error.dict()
    }




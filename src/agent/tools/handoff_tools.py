"""
Handoff 도구

에이전트 간 작업 전환을 위한 도구들을 정의합니다.
"""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel


class EmptyInput(BaseModel):
    """빈 입력 스키마 - handoff tool은 입력 매개변수가 없음"""
    pass


def create_handoff_tool(*, agent_name: str, description: str | None = None):
    """
    에이전트 간 작업 전환을 위한 도구를 생성합니다.
    
    Args:
        agent_name: 전환할 에이전트 이름
        description: 도구 설명
        
    Returns:
        Handoff 도구
    """
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help"

    def handoff_func() -> str:
        """
        에이전트로 전환합니다.
        실제 전환 로직은 graph의 노드에서 처리됩니다.
        """
        return f"Transferring to {agent_name}..."
    
    return StructuredTool(
        name=name,
        description=description,
        func=handoff_func,
        args_schema=EmptyInput
    )




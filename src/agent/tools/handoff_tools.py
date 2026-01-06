"""
Handoff 도구

에이전트 간 작업 전환을 위한 도구들을 정의합니다.
"""

from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from ..state import AgentState


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

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[AgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = {
            'role': 'tool',
            'content': f'Successfully transferred to {agent_name}',
            'name': name,
            'tool_call_id': tool_call_id,
        }
        return Command(
            goto=agent_name,
            update={**state, "messages": state['messages'] + [tool_message]},
            graph=Command.PARENT,
        )
    return handoff_tool




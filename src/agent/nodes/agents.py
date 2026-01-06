"""
에이전트 생성

create_react_agent를 사용하여 각 에이전트를 생성합니다.
"""

from langgraph.prebuilt import create_react_agent
from ..config import (
    model_name,
    list_tables_tool,
    get_schema_tool,
    retriever_tool,
    RAG_AVAILABLE,
)
from ..models import SubmitFinalAnswer
from ..prompts import (
    query_schema_prompt,
    query_gen_prompt,
    query_execute_prompt,
    final_answer_prompt,
    rag_prompt,
    query_interpreter_prompt,
)
from ..tools import model_check_query, db_query_tool


# SQL Schema Agent
sql_schema_agent = create_react_agent(
    model=model_name,
    tools=[list_tables_tool, get_schema_tool],
    prompt=query_schema_prompt,
    name='SQL_schema_agent'
)

# SQL Gen Agent
sql_gen_agent = create_react_agent(
    model=model_name,
    tools=[],
    prompt=query_gen_prompt,
    name='SQL_gen_agent'
)

# SQL Check Agent
sql_check_agent = create_react_agent(
    model=model_name,
    tools=[model_check_query],
    prompt=query_execute_prompt,
    name='SQL_check_agent'
)

# SQL Execute Agent
sql_execute_agent = create_react_agent(
    model=model_name,
    tools=[db_query_tool],
    prompt=query_execute_prompt,
    name='SQL_execute_agent'
)

# Final Answer Agent
final_answer_agent = create_react_agent(
    model=model_name,
    tools=[SubmitFinalAnswer],
    prompt=final_answer_prompt,
    name='Final_answer_agent'
)

# RAG Agent (조건부)
if RAG_AVAILABLE and retriever_tool:
    rag_agent = create_react_agent(
        model=model_name,
        tools=[retriever_tool],
        prompt=rag_prompt,
        name='RAG_agent'
    )
else:
    rag_agent = None

# Query Interpreter
query_interpreter = create_react_agent(
    model=model_name,
    tools=[],
    prompt=query_interpreter_prompt,
    name='Query_interpreter'
)

# Supervisor Agent
def create_supervisor_agent(supervisor_tools):
    """Supervisor 에이전트를 생성합니다."""
    supervisor_prompt = (
        "You are a supervisor managing some agents:\n"
        "a SQL_gen_agent. Assign when generate SQL query to this agent.\n"
        "a SQL_schema_agent. Assign this agent before generate SQL query.\n"
        "a SQL_execute_agent. Assign to execute sql query to this agent.\n"
        "a SQL_check_agent. Assign to check generated query works well to this agent.\n"
        "a Final_answer_agent. Assign to make final answer to this agent.\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "SQL_gen_agent가 쿼리를 생성하면, 반드시 SQL_check_agent를 거쳐 검증해야 한다.\n"
        "SQL_execute_agent는 SQL_check_agent에서 검증된 쿼리만 실행한다.\n"
        "Do not do any work yourself"
    )
    
    return create_react_agent(
        model=model_name,
        tools=supervisor_tools,
        prompt=supervisor_prompt,
        name='supervisor'
    )



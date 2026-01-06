"""
에이전트 생성

create_react_agent를 사용하여 각 에이전트를 생성합니다.
"""

from langgraph.prebuilt import create_react_agent
from ..config import (
    model_name,
    sql_gen_model,
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
from ..tools import validate_sql_syntax, db_query_tool


# SQL Schema Agent
sql_schema_agent = create_react_agent(
    model=model_name,
    tools=[list_tables_tool, get_schema_tool],
    prompt=query_schema_prompt,
    name='SQL_schema_agent'
)

# SQL Gen Agent (고성능 모델 사용)
sql_gen_agent = create_react_agent(
    model=sql_gen_model,
    tools=[],
    prompt=query_gen_prompt,
    name='SQL_gen_agent'
)

# SQL Check Agent
sql_check_agent = create_react_agent(
    model=model_name,
    tools=[validate_sql_syntax],
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
        "You are a supervisor managing a multi-agent workflow.\n\n"
        "Available agents:\n"
        "- SQL_schema_agent: Use this to START a new SQL query workflow. "
        "This agent will check database schema and then automatically proceed through "
        "SQL generation → validation → execution.\n"
        "- Final_answer_agent: Use this to provide the final answer to the user "
        "after SQL execution is complete.\n\n"
        "Rules:\n"
        "1. For NEW queries: assign to SQL_schema_agent first\n"
        "2. After SQL execution completes: assign to Final_answer_agent\n"
        "3. Assign work to ONE agent at a time\n"
        "4. Do NOT do any work yourself\n\n"
        "The SQL workflow (schema → gen → check → execute) runs automatically.\n"
        "You only need to START it and provide FINAL answer."
    )
    
    return create_react_agent(
        model=model_name,
        tools=supervisor_tools,
        prompt=supervisor_prompt,
        name='supervisor'
    )




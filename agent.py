# 라이브러리 호출
import pandas as pd
import numpy as np
import operator
import functools
from datetime import datetime,timedelta
import sqlite3
import shutil
from db import set_db
import os
from dotenv import load_dotenv

from typing import Sequence,Annotated,Literal
from typing_extensions import TypedDict,List
from pydantic import BaseModel, Field
from IPython.display import Image, display
from langchain_naver import ChatClovaX
from langchain.output_parsers import OutputFixingParser
from langchain_core.tools import Tool,tool
from langchain_core.messages import HumanMessage,BaseMessage,AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import AnyMessage, add_messages

# 에이전트 상태 정의
class State(TypedDict):
  messages: Annotated[list[AnyMessage], add_messages]

load_dotenv()
# 환경변수에서 키 꺼내기
api_key = os.getenv("CLOVASTUDIO_API_KEY")
if not api_key:
    raise RuntimeError("CLOVASTUDIO_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# 사용할 LLM 정의
llm=ChatClovaX(model='HCX-005',api_key=api_key,max_tokens=4096,temperature=0,top_k=3)

# 사용할 DB 정의 
db_path='stock_db.db'
db=set_db(db_path)

# SQLDatabaseToolkit 생성
sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
# SQLDatabaseToolkit에서 사용 가능한 도구 목록
sql_tools = sql_toolkit.get_tools()

# SQL 다루는 도구 정의
get_schema_tool = next(tool for tool in sql_tools if tool.name == "sql_db_schema")
get_schema_node = ToolNode([get_schema_tool], name="get_schema")

# query 실행 도구
run_query_tool = next(tool for tool in sql_tools if tool.name == "sql_db_query")
run_query_node = ToolNode([run_query_tool], name="run_query")

@tool
def db_query_tool(query: str) -> str:
    """
    Run SQL queries against a database and return results
    If the query executes but returns no data, return a user-friendly message.
    Returns an error message if the query is incorrect
    If an error is returned, rewrite the query, check, and retry
    """
    # 쿼리 실행
    result = db.run_no_throw(query)

    if result is None:
        return "해당 날짜에는 거래 데이터가 없습니다."

    # 오류: 결과가 없으면 오류 메시지 반환
    if not result:
        return "Error: Query failed. Please rewrite your query and try again."

    # 정상: 쿼리 실행 결과 반환
    return result

db_query_node=ToolNode([db_query_tool],name='db_query')


def list_tables(state: State):
    tool_call = {
        "name": "sql_db_list_tables",
        "args": {},
        "id": "abc123",
        "type": "tool_call",
    }
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])

    list_tables_tool = next(tool for tool in sql_tools if tool.name == "sql_db_list_tables")
    tool_message = list_tables_tool.invoke(tool_call)
    response = AIMessage(f"Available tables: {tool_message.content}")

    return {"messages": [tool_call_message, tool_message, response]}

lm=ChatClovaX(model='HCX-005',api_key=api_key,max_tokens=4096,temperature=0,top_k=3)

def call_get_schema(state: State):
    llm_with_tools = llm.bind_tools([get_schema_tool])
    response = llm_with_tools.invoke(state["messages"])

    return {"messages": [response]}

# 쿼리 아웃풋 정의
class QueryOutput(BaseModel):
    """Generated SQL query."""
    query: str

parser=PydanticOutputParser(pydantic_object=QueryOutput)
new_parser=OutputFixingParser.from_llm(parser=parser,llm=llm)

# 쿼리 생성 generate_query 정의
generate_query_system_prompt = """
You are a SQL expert with a strong attention to detail.
Always, upon receiving the user’s request, immediately strive to generate the most appropriate SQL query to answer their question.
You can define SQL queries, analyze query results and interpret query results to respond with an answer.
Read the messages below and identify the user question, table schemas, query statement and query result, or error if they exist.

1.If there’s not any query result that makes sense to answer the question, create a syntactically correct SQLite query to answer the user question. DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP, etc.) to the database.

2.If you create a query, respond ONLY with the query statement. For example: SELECT id, name FROM pets;

3. If a query was already executed, but there was an error, respond with the exact error message you found. For example: Error: Pets table doesn’t exist

4. If a query was already executed successfully, interpret the response and answer the question following this pattern: Answer: <<question answer>>.
For example: Answer: 오로스테크놀로지의 2024-11-28 종가는 12,345원입니다.

5. Always limit your query to at most 10 results, and if the user might need more, ask: Would you like to see more than 10 results?

최종답변은 한국어로 하세요.
"""

# 최종 상태를 나타내는 도구 설명
class SubmitFinalAnswer(BaseModel):
    """쿼리 결과를 기반으로 사용자에게 최종 답변 제출"""
    final_answer: str = Field(..., description="The final answer to the user")

llm=ChatClovaX(model='HCX-005',api_key=api_key,max_tokens=4096,temperature=0,top_k=3)

def generate_query(state: State):
    system_message = {
        "role": "system",
        "content": generate_query_system_prompt,
    }

    llm_with_tools = llm.bind_tools([db_query_tool,SubmitFinalAnswer])
    response = llm_with_tools.invoke([system_message] + state["messages"])

    return {"messages": [response]}

# 쿼리 생성이 제대로 된건지 확인하는 check query 정의
check_query_system_prompt = """
You are a SQL expert with a strong attention to detail.
Double check the {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes,
just reproduce the original query.

You will call the appropriate tool to execute the query after running this check.
""".format(dialect=db.dialect)


# 프롬프트 생성
query_check_prompt = ChatPromptTemplate.from_messages(
    [("system", check_query_system_prompt), ("placeholder", "{messages}")]
)
lm=ChatClovaX(model='HCX-005',api_key=api_key,max_tokens=4096,temperature=0,top_k=3)

llm_with_tools=llm.bind_tools([db_query_tool])

# Query Checker 체인 생성
query_check = query_check_prompt | llm_with_tools

# 쿼리의 정확성을 모델로 점검하기 위한 함수 정의
def check_query(state: State) -> dict[str, list[AIMessage]]:
    """
    Use this tool to check that your query is correct before you run it
    """
    return {"messages": [query_check.invoke({"messages": [state["messages"][-1]]})]}


# 조건부 에지 정의
def should_continue(state: State) -> Literal[END, "check_query", "generate_query"]:
    messages = state["messages"]

    last_message = messages[-1]
    if last_message.content.startswith("Answer:"):
        return END
    if last_message.content.startswith("Error:"):
        return "generate_query"
    else:
        return "check_query"

# 랭그래프 정의
builder = StateGraph(State)
builder.add_node(list_tables)
builder.add_node(call_get_schema)
builder.add_node(get_schema_node, "get_schema")
builder.add_node(generate_query,"generate_query")
builder.add_node(check_query,"check_query")
builder.add_node(db_query_node,'db_query')

builder.add_edge(START, "list_tables")
builder.add_edge("list_tables", "call_get_schema")
builder.add_edge("call_get_schema", "get_schema")
builder.add_edge("get_schema", "generate_query")
builder.add_conditional_edges(
    "generate_query",
    should_continue,
)
builder.add_edge("check_query", "db_query")
builder.add_edge("db_query", "generate_query")

agent = builder.compile()
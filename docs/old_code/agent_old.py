# 라이브러리 호출
import pandas as pd
import numpy as np
import operator
import functools
from datetime import datetime, timedelta
import sqlite3
import shutil
import re
from .db import set_db
import os
from dotenv import load_dotenv

from typing import Sequence, Annotated, Literal, Optional, Any, Dict, List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from .errors import classify_error, ErrorSeverity, AgentError
from .logger import logger
import time
from langchain_openai import ChatOpenAI
from langchain_naver import ChatClovaX
from langchain_core.tools import Tool, tool, InjectedToolCallId
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableWithFallbacks
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, InjectedState, create_react_agent
from langgraph.types import Command

# AgentState 정의
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

# 환경변수 로드
load_dotenv()

# LangSmith 추적 설정
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Naver-Stock-Agent")

# API 키 꺼내기
api_key_clova = os.getenv("CLOVASTUDIO_API_KEY")
api_key_openai = os.getenv("OPENAI_API_KEY")

if not api_key_clova and not api_key_openai:
    raise RuntimeError("CLOVASTUDIO_API_KEY 또는 OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# 사용할 DB 정의 
db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'stock_db.db')
db = set_db(db_path)

# SQLDatabaseToolkit 생성
# 기본 모델 설정 (OpenAI가 있으면 OpenAI, 없으면 ClovaX)
# 타임아웃 및 재시도 설정 추가
MAX_RETRIES = 3  # 전역 상수

if api_key_openai:
    default_model = ChatOpenAI(
        model='gpt-5-nano', 
        api_key=api_key_openai, 
        temperature=0,
        timeout=30,  # 30초 타임아웃
        max_retries=2  # LLM 자체 재시도
    )
    model_name = 'openai:gpt-5-nano'
elif api_key_clova:
    default_model = ChatClovaX(
        model='HCX-005', 
        api_key=api_key_clova, 
        max_tokens=4096, 
        temperature=0, 
        top_k=3,
        timeout=30  # 30초 타임아웃
    )
    model_name = 'HCX-005'
else:
    raise RuntimeError("사용 가능한 LLM API 키가 없습니다.")

sql_toolkit = SQLDatabaseToolkit(db=db, llm=default_model)
sql_tools = sql_toolkit.get_tools()

# SQL 다루는 도구 정의
list_tables_tool = next(tool for tool in sql_tools if tool.name == "sql_db_list_tables")
get_schema_tool = next(tool for tool in sql_tools if tool.name == "sql_db_schema")

# 쿼리 실행 도구
@tool
def db_query_tool(query: str) -> str:
    """
    Run SQL queries against a database and return results.
    If the query executes but returns no data, return a user-friendly message.
    Returns an error message if the query is incorrect.
    If an error is returned, rewrite the query, check, and retry.
    """
    logger.info(f"Executing SQL query", extra={"query": query[:100]})
    
    # SQL 검증
    is_valid, error_msg = validate_sql_query(query)
    if not is_valid:
        logger.error(
            "SQL validation failed",
            extra={"query": query, "error": error_msg}
        )
        return f'Error: {error_msg}. Please rewrite your query and try again'
    
    start_time = time.time()
    result = db.run_no_throw(query)
    duration_ms = int((time.time() - start_time) * 1000)
    
    # 사용된 테이블 추출
    tables_used = extract_tables_from_query(query)

    # 1) 쿼리 실패 (Error 문자열 반환)
    if isinstance(result, str) and result.startswith("Error:"):
        logger.error(
            "SQL query failed",
            extra={
                "query": query,
                "duration_ms": duration_ms,
                "tables_used": tables_used
            }
        )
        return 'Error: Query failed. Please rewrite your query and try again'

    # 2) 실행은 성공했지만 결과가 빈 경우
    if (isinstance(result, list) and len(result) == 0) or result in ("[]", ""):
        logger.info(
            "SQL query returned no results",
            extra={
                "query": query,
                "duration_ms": duration_ms,
                "tables_used": tables_used
            }
        )
        return "Answer: No rows found for the given query."

    # 3) 정상 결과
    result_size = len(str(result))
    logger.info(
        "SQL query succeeded",
        extra={
            "query": query,
            "duration_ms": duration_ms,
            "result_size_bytes": result_size,
            "tables_used": tables_used
        }
    )
    
    return result


# 오류 처리 함수 (개선)
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


def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """오류 발생 시 대체 동작을 정의하며 ToolNode에 추가"""
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key='error'
    )


# 쿼리 체크 도구
@tool
def model_check_query(state: AgentState) -> dict:
    """
    Use this tool to check that your SQL query is correct before you run it.
    The query is taken from the last message in the state.
    """
    query_check_system = """You are a SQL expert with a strong attention to detail.
Double check the SQLite query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query.
If there are no mistakes, just reproduce the original query.

Do not execute the query yourself. Return only the corrected query."""

    query_check_prompt = ChatPromptTemplate.from_messages(
        [('system', query_check_system), ('placeholder', '{messages}')]
    )
    
    query_check = query_check_prompt | default_model.bind_tools(
        [db_query_tool], tool_choice='db_query_tool'
    )
    
    last_msg = state["messages"][-1]
    result = query_check.invoke({"messages": [last_msg]})
    return {"messages": [result]}


# ============================================================
# Structured Output Models
# ============================================================

class SQLQueryOutput(BaseModel):
    """SQL 쿼리 생성 결과"""
    query: str = Field(description="생성된 SELECT 쿼리")
    tables_used: List[str] = Field(
        default_factory=list,
        description="쿼리에 사용된 테이블 목록"
    )
    estimated_rows: Optional[int] = Field(
        None, 
        description="예상 결과 행 수 (LIMIT 값 또는 예측)"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="쿼리 정확도 신뢰도 (0-1)"
    )
    explanation: Optional[str] = Field(
        None,
        description="쿼리 설명 (선택사항)"
    )


class QueryInterpretation(BaseModel):
    """쿼리 해석 결과"""
    action: Literal["pass_through", "rewrite", "need_user_input"] = Field(
        description="처리 액션"
    )
    processed_query: str = Field(
        description="처리된 쿼리 (rewrite된 경우 새 쿼리, 아니면 원본)"
    )
    missing_info: Optional[List[str]] = Field(
        None,
        description="부족한 정보 목록 (need_user_input인 경우)"
    )
    clarification_question: Optional[str] = Field(
        None,
        description="사용자에게 물어볼 질문"
    )
    confidence: float = Field(
        default=1.0,
        description="해석 신뢰도"
    )


class FinalAnswerOutput(BaseModel):
    """최종 답변 출력"""
    final_answer: str = Field(
        ..., 
        description="사용자에게 전달할 최종 답변 (한국어)"
    )
    answer_type: Literal["single", "list", "table", "error", "clarification"] = Field(
        default="single",
        description="답변 유형"
    )
    data_points: int = Field(
        default=0,
        description="사용된 데이터 포인트 수"
    )
    sources: Optional[List[str]] = Field(
        None,
        description="데이터 출처 (테이블명 또는 RAG 문서)"
    )
    confidence: float = Field(
        default=1.0,
        description="답변 신뢰도"
    )
    follow_up_suggestions: Optional[List[str]] = Field(
        None,
        description="후속 질문 제안"
    )


# 하위 호환성을 위한 레거시 클래스 (도구로 사용)
class SubmitFinalAnswer(BaseModel):
    """쿼리 결과를 기반으로 사용자에게 최종 답변 제출"""
    final_answer: str = Field(..., description="The final answer to the user")


# ============================================================
# Validation Functions
# ============================================================

def validate_sql_query(query: str) -> tuple[bool, Optional[str]]:
    """
    SQL 쿼리 검증
    
    Returns:
        (is_valid, error_message)
    """
    query_upper = query.strip().upper()
    
    # 기본 검증
    if not query_upper.startswith("SELECT"):
        return False, "Query must start with SELECT"
    
    # 위험한 키워드 차단
    dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "EXEC"]
    for keyword in dangerous_keywords:
        if keyword in query_upper:
            logger.error(f"Dangerous SQL keyword detected: {keyword}")
            return False, f"Dangerous keyword not allowed: {keyword}"
    
    # LIMIT 확인 (성능)
    if "LIMIT" not in query_upper:
        logger.warning("SQL query without LIMIT clause (performance concern)")
        # 경고만 하고 통과
    
    # 기본 SQL 문법 체크
    if query_upper.count("(") != query_upper.count(")"):
        return False, "Unbalanced parentheses"
    
    return True, None


def extract_tables_from_query(query: str) -> List[str]:
    """SQL 쿼리에서 테이블명 추출"""
    import re
    
    # FROM 절에서 테이블 추출
    tables = []
    from_pattern = r'FROM\s+(\w+)'
    join_pattern = r'JOIN\s+(\w+)'
    
    tables.extend(re.findall(from_pattern, query, re.IGNORECASE))
    tables.extend(re.findall(join_pattern, query, re.IGNORECASE))
    
    return list(set(tables))


# RAG 시스템 설정 (rag_setup.py에서 import)
try:
    from .rag_setup import get_retriever_tool
    retriever_tool = get_retriever_tool()
    RAG_AVAILABLE = True
    logger.info("RAG 시스템이 성공적으로 로드되었습니다.")
except ImportError:
    logger.warning("⚠️ RAG 모듈을 찾을 수 없습니다. RAG 기능 없이 실행됩니다.")
    retriever_tool = None
    RAG_AVAILABLE = False


# ============================================================
# 각 에이전트 프롬프트
# ============================================================

query_schema_prompt = """
You are SQL_schema_agent.
Your primary job is to return the database schema (tables and columns).
Always print the schema information first.

Then, decide which agent should handle the request next:

- If the request can be solved using schema and SQL directly (tables, columns, basic queries), output at the end: ROUTE: SQL_gen_agent
- If the request involves derived indicators, technical analysis, or knowledge not in the schema (e.g., RSI, moving averages, Bollinger Bands, golden/dead cross, patterns), output at the end: ROUTE: RAG_agent

Rules:
- Do NOT generate or execute queries.
- Do NOT interpret data values.
- Always end your answer with exactly one routing tag:
  ROUTE: SQL_gen_agent
  or
  ROUTE: RAG_agent
"""

query_gen_prompt = """
You are a SQL expert specializing in Korean stock market data (KOSPI/KOSDAQ).

Database Schema:
- Stocks: Stock_ticker (e.g., '005930.KS'), Stock_Name (e.g., '삼성전자'), Market ('KOSPI'/'KOSDAQ')
- Stock_Prices: Stock_Name, date (format: 'YYYY-MM-DD'), open, high, low, close, volume, dividends, splits

Few-shot Examples:

Example 1 - Single stock latest price:
User: "삼성전자의 최근 종가를 알려줘"
Query: SELECT Stock_Name, date, close FROM Stock_Prices WHERE Stock_Name = '삼성전자' ORDER BY date DESC LIMIT 1;
Tables: ['Stock_Prices']

Example 2 - Top N by volume:
User: "거래량이 많은 상위 10개 종목은?"
Query: SELECT Stock_Name, SUM(volume) as total_volume FROM Stock_Prices GROUP BY Stock_Name ORDER BY total_volume DESC LIMIT 10;
Tables: ['Stock_Prices']

Example 3 - Price filter on specific date:
User: "2024-12-27 종가가 10만원 이상인 종목"
Query: SELECT Stock_Name, close FROM Stock_Prices WHERE date = '2024-12-27' AND close >= 100000 ORDER BY close DESC LIMIT 15;
Tables: ['Stock_Prices']

Example 4 - Multiple stocks comparison:
User: "삼성전자와 SK하이닉스의 최근 종가 비교"
Query: SELECT Stock_Name, date, close FROM Stock_Prices WHERE Stock_Name IN ('삼성전자', 'SK하이닉스') ORDER BY date DESC, Stock_Name LIMIT 2;
Tables: ['Stock_Prices']

Example 5 - Date range query:
User: "삼성전자의 2024-12-01부터 2024-12-27까지 종가"
Query: SELECT date, close FROM Stock_Prices WHERE Stock_Name = '삼성전자' AND date BETWEEN '2024-12-01' AND '2024-12-27' ORDER BY date;
Tables: ['Stock_Prices']

Rules:
1. Always produce a valid SQLite SELECT query
2. Use Korean stock names EXACTLY as they appear (e.g., '삼성전자', not 'Samsung')
3. Date format: 'YYYY-MM-DD' (string type)
4. Use LIMIT to restrict results (default: 15 for lists)
5. For aggregations, always use GROUP BY
6. Do NOT use SELECT * - specify columns
7. Output ONLY the SQL query, nothing else

Quality Guidelines:
- Confidence: Rate query accuracy 0-1 (1 = certain, 0.5 = uncertain)
- Safety: Never use DELETE, UPDATE, DROP, INSERT
- Performance: Always include LIMIT clause
"""

query_execute_prompt = """
You are SQL_execute_agent.

Your ONLY responsibility is to run SQL queries against the database
and return the raw execution results.

Rules:
- Do NOT generate or modify queries yourself.
- Do NOT interpret, summarize, or explain the results.
- Simply execute the validated query you receive and return the results exactly as they are.
- If the execution fails, return the error message as-is without fixing it.

The interpretation of results will be handled by Final_answer_agent.
"""

final_answer_prompt = '''
You are Final_answer_agent. Transform SQL query results into clear, user-friendly Korean answers.

Answer Format Guidelines:

1. Language & Tone:
   - Always write in Korean (한국어)
   - Be concise and professional
   - Directly answer the user's question

2. Number Formatting:
   - Stock prices: Format with comma (e.g., "50,000원")
   - Volume: Format with comma and units (e.g., "1,234,567주" or "123만주")
   - Percentages: Show 2 decimal places (e.g., "3.45%")
   - Dates: Korean format (e.g., "2024년 12월 27일" or "2024-12-27")

3. Response Structure:
   For single results:
   "[종목명]의 [날짜] [항목]은 [값]입니다."
   Example: "삼성전자의 2024-12-27 종가는 50,000원입니다."
   
   For multiple results (list/table):
   Use markdown table or numbered list
   Example:
   | 순위 | 종목명 | 거래량 |
   |------|--------|--------|
   | 1 | 삼성전자 | 1,234,567주 |
   | 2 | SK하이닉스 | 987,654주 |
   
   Or: "거래량 상위 종목:\n1. 삼성전자: 1,234,567주\n2. SK하이닉스: 987,654주"

4. Result Handling:
   - Empty results: "조건에 맞는 종목이 없습니다."
   - Limit to top 15 results for lists
   - Always sort by most relevant metric

5. Context Awareness:
   - If user asks "최근", use the latest date in results
   - If user asks "상위 N개", show exactly N items
   - Infer intent: "급등" = highest price increase, "거래량 많은" = highest volume

6. Error Messages:
   - Be helpful and suggest what might be wrong
   - Example: "해당 날짜의 데이터가 없습니다. 최근 거래일을 기준으로 조회하시겠습니까?"
'''

rag_prompt = '''
Schema에 정의된 정보만으로 사용자의 요청을 수행하거나 이해가 불가능한 경우 문서에서 내용을 검색해 보강하세요.
You must never ask the user any questions during intermediate steps.
'''

query_interpreter_prompt = '''
You are Query Interpreter.
Your role is ONLY to interpret the very first user request.
Do not analyze or intervene in intermediate agent steps or tool outputs.
Do not ask the user unnecessary questions unless absolutely required.

Classify the user request into one of the following three modes:

1. **pass_through**:
   - The request is clear and complete.
   - In this case, return exactly the same request as plain text, prefixed with:
     "pass_through: <original request>"

2. **rewrite**:
   - The request is ambiguous, contains slang, abbreviations, or domain-specific jargon.
   - Normalize and rewrite it into a clarified form, prefixed with:
     "rewrite: <clarified request>"

3. **need_user_input**:
   - The request is missing critical information (e.g., no date, unclear metric, vague wording).
   - In this case, return a clarification question for the user, prefixed with:
     "need_user_input: <clarification question>"

Rules:
- Do not wrap your output in JSON or any structured object.
- Output must be a single line of plain text starting with one of:
  "pass_through:", "rewrite:", or "need_user_input:".
- Rate your confidence: 1.0 (certain), 0.8 (likely), 0.5 (unsure)
'''


# ============================================================
# 각 에이전트 정의
# ============================================================

sql_schema_agent = create_react_agent(
    model=model_name,
    tools=[list_tables_tool, get_schema_tool],
    prompt=query_schema_prompt,
    name='SQL_schema_agent'
)

sql_gen_agent = create_react_agent(
    model=model_name,
    tools=[],
    prompt=query_gen_prompt,
    name='SQL_gen_agent'
)

sql_check_agent = create_react_agent(
    model=model_name,
    tools=[model_check_query],
    prompt=query_execute_prompt,  # Note: ipynb에서는 check_prompt를 사용했지만 여기서는 execute_prompt
    name='SQL_check_agent'
)

sql_execute_agent = create_react_agent(
    model=model_name,
    tools=[db_query_tool],
    prompt=query_execute_prompt,
    name='SQL_execute_agent'
)

final_answer_agent = create_react_agent(
    model=model_name,
    tools=[SubmitFinalAnswer],
    prompt=final_answer_prompt,
    name='Final_answer_agent'
)

if RAG_AVAILABLE and retriever_tool:
    rag_agent = create_react_agent(
        model=model_name,
        tools=[retriever_tool],
        prompt=rag_prompt,
        name='RAG_agent'
    )
else:
    rag_agent = None

query_interpreter = create_react_agent(
    model=model_name,
    tools=[],
    prompt=query_interpreter_prompt,
    name='Query_interpreter'
)


# ============================================================
# Handoff Tools (Supervisor와 개별 에이전트 소통 방식)
# ============================================================

def create_handoff_tool(*, agent_name: str, description: str | None = None):
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


# Handoffs : 각 에이전트 간 작업 전환을 위한 도구
assign_gen_agent = create_handoff_tool(
    agent_name='SQL_gen_agent',
    description='Assign task to a SQL gen agent.',
)

assign_to_check_agent = create_handoff_tool(
    agent_name='SQL_check_agent',
    description='Assign task to a SQL check agent.'
)

assign_to_schema_agent = create_handoff_tool(
    agent_name='SQL_schema_agent',
    description='Assign task to a SQL schema agent'
)

assign_to_final_ans_agent = create_handoff_tool(
    agent_name='Final_answer_agent',
    description='Assign task to a Final answer agent'
)

assign_to_execute_agent = create_handoff_tool(
    agent_name='SQL_execute_agent',
    description='Assign task to a SQL execute agent'
)

if RAG_AVAILABLE:
    assign_to_rag_agent = create_handoff_tool(
        agent_name='RAG_agent',
        description='Assign task to RAG agent'
    )
else:
    assign_to_rag_agent = None

assign_to_interpreter_agent = create_handoff_tool(
    agent_name='Query_interpreter',
    description='Assign task to Query interpreter agent'
)


# ============================================================
# Supervisor Agent : 중간관리자 에이전트, 각 에이전트의 작업을 조정하고 분기하는 역할
# ============================================================

supervisor_tools = [
    assign_to_final_ans_agent,
    assign_to_schema_agent,
    assign_to_check_agent,
]

supervisor_agent = create_react_agent(
    model=model_name,
    tools=supervisor_tools,
    prompt=(
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
    ),
    name='supervisor'
)


# ============================================================
# Multi-Agent Graph 구성
# ============================================================

def route_from_schema(state: AgentState) -> str:
    """Schema agent의 출력을 보고 RAG vs SQL_gen으로 분기"""
    last_msg = state["messages"][-1].content
    if "ROUTE: RAG_agent" in last_msg and RAG_AVAILABLE:
        return "RAG_agent"
    return "SQL_gen_agent"


def route_from_interpreter(state: AgentState) -> str:
    """Query Interpreter의 출력을 보고 분기"""
    last_msg = state["messages"][-1].content.lower()
    if last_msg.startswith("need_user_input:"):
        return "Final_answer_agent"
    else:
        return "supervisor"


def should_check_query(state: AgentState) -> str:
    """
    쿼리 복잡도를 분석하여 검증 필요 여부 결정
    간단한 쿼리는 검증을 스킵하여 속도 향상
    """
    # 마지막 메시지에서 SQL 쿼리 추출
    messages = state["messages"]
    query = None
    
    for msg in reversed(messages):
        content = msg.content if hasattr(msg, 'content') else str(msg)
        # SELECT로 시작하는 SQL 쿼리 찾기
        if content.strip().upper().startswith("SELECT"):
            query = content.strip()
            break
    
    if not query:
        return "SQL_check_agent"  # 쿼리를 찾을 수 없으면 검증
    
    # 간단한 쿼리 패턴 (검증 스킵 가능)
    simple_patterns = [
        # 단일 레코드 조회 (WHERE + LIMIT 1)
        r"SELECT\s+.+\s+FROM\s+\w+\s+WHERE\s+.+\s+LIMIT\s+1",
        # 간단한 정렬 (ORDER BY + LIMIT, JOIN 없음)
        r"SELECT\s+.+\s+FROM\s+\w+\s+(?:WHERE\s+.+\s+)?ORDER BY\s+.+\s+LIMIT\s+\d+",
        # 단순 WHERE 조건만 (LIMIT 있음)
        r"SELECT\s+.+\s+FROM\s+\w+\s+WHERE\s+[^()]+\s+LIMIT\s+\d+",
    ]
    
    # 복잡한 쿼리 키워드 (검증 필수)
    complex_keywords = [
        "JOIN", "UNION", "SUBQUERY", "CASE", "HAVING",
        "GROUP BY.*HAVING", "DISTINCT.*COUNT", "EXISTS"
    ]
    
    # 복잡한 쿼리인지 먼저 체크
    for keyword in complex_keywords:
        if re.search(keyword, query, re.IGNORECASE):
            return "SQL_check_agent"  # 복잡한 쿼리는 반드시 검증
    
    # 간단한 패턴과 매치되면 검증 스킵
    for pattern in simple_patterns:
        if re.match(pattern, query, re.IGNORECASE | re.DOTALL):
            return "SQL_execute_agent"  # 검증 스킵하고 바로 실행
    
    # 기본적으로 검증 수행
    return "SQL_check_agent"


# StateGraph 생성 (AgentState 사용)
graph_builder = StateGraph(AgentState)

# 노드 추가
graph_builder.add_node(supervisor_agent)
graph_builder.add_node(query_interpreter)
graph_builder.add_node(sql_schema_agent)
graph_builder.add_node(sql_gen_agent)
graph_builder.add_node(sql_check_agent)
graph_builder.add_node(sql_execute_agent)
graph_builder.add_node(final_answer_agent)

if RAG_AVAILABLE and rag_agent:
    graph_builder.add_node(rag_agent)

# 엣지 추가
graph_builder.add_edge(START, "Query_interpreter")

graph_builder.add_conditional_edges(
    "Query_interpreter",
    route_from_interpreter,
    {
        "Final_answer_agent": "Final_answer_agent",
        "supervisor": "supervisor"
    }
)

graph_builder.add_conditional_edges(
    "SQL_schema_agent",
    route_from_schema,
    {
        "SQL_gen_agent": "SQL_gen_agent",
        "RAG_agent": "RAG_agent" if RAG_AVAILABLE else "SQL_gen_agent"
    }
)

if RAG_AVAILABLE:
    graph_builder.add_edge("RAG_agent", 'SQL_gen_agent')

# SQL_gen_agent 후 조건부 분기 (간단한 쿼리는 검증 스킵)
graph_builder.add_conditional_edges(
    "SQL_gen_agent",
    should_check_query,
    {
        "SQL_check_agent": "SQL_check_agent",
        "SQL_execute_agent": "SQL_execute_agent"
    }
)

graph_builder.add_edge("SQL_check_agent", "SQL_execute_agent")
graph_builder.add_edge("SQL_execute_agent", "supervisor")
graph_builder.add_edge("Final_answer_agent", END)

# 컴파일
_compiled_agent = graph_builder.compile()


# 로깅이 추가된 에이전트 래퍼
def create_logged_agent(compiled_graph):
    """로깅이 추가된 에이전트 래퍼"""
    
    original_invoke = compiled_graph.invoke
    
    def logged_invoke(input_data, config=None):
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        logger.info(
            "=== Agent session started ===",
            extra={
                "session_id": session_id,
                "input": str(input_data.get("messages", []))[:200]
            }
        )
        
        start_time = time.time()
        
        try:
            result = original_invoke(input_data, config)
            duration_ms = int((time.time() - start_time) * 1000)
            
            logger.info(
                "=== Agent session completed ===",
                extra={
                    "session_id": session_id,
                    "duration_ms": duration_ms,
                    "success": True,
                    "message_count": len(result.get("messages", []))
                }
            )
            
            return result
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            
            logger.error(
                "=== Agent session failed ===",
                extra={
                    "session_id": session_id,
                    "duration_ms": duration_ms,
                    "success": False,
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            
            raise
    
    # 원본 메서드 유지하면서 invoke만 래핑
    compiled_graph.invoke = logged_invoke
    return compiled_graph


agent = create_logged_agent(_compiled_agent)


# State 초기화 헬퍼 함수
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


# 그래프 시각화 함수 (선택사항)
def visualize_agent_graph():
    """에이전트 그래프를 시각화합니다 (IPython 환경에서만 작동)"""
    try:
        from IPython.display import display, Image
        display(Image(agent.get_graph().draw_mermaid_png()))
    except Exception as e:
        logger.warning(f"⚠️ 그래프 시각화 실패: {e}")


if __name__ == "__main__":
    # 테스트 실행
    logger.info("✅ 멀티 에이전트 시스템이 성공적으로 로드되었습니다.")
    logger.info(f"📊 사용 모델: {model_name}")
    logger.info(f"🗄️ DB 경로: {db_path}")
    logger.info(f"📚 RAG 사용 가능: {RAG_AVAILABLE}")
    
    # 간단한 테스트
    test_query = "삼성전자의 최근 종가를 알려줘"
    logger.info(f"🧪 테스트 쿼리: {test_query}")
    
    try:
        result = agent.invoke({
            "messages": [
                {"role": "user", "content": test_query}
            ]
        })
        logger.info("✅ 테스트 성공!")
        if result and "messages" in result:
            last_message = result["messages"][-1]
            logger.info(f"📝 최종 답변: {last_message.content if hasattr(last_message, 'content') else last_message}")
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}", exc_info=True)

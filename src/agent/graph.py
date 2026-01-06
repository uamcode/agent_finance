"""
Graph 구성

라우팅 함수와 StateGraph를 구성하고 컴파일합니다.
Supervisor 제거로 직접 연결 방식으로 최적화됨 (20-40초 속도 향상)
"""

import re
from datetime import datetime
import time
from langgraph.graph import END, START, StateGraph
from ..logger import logger
from .state import AgentState
from .config import RAG_AVAILABLE
from .nodes import (
    sql_schema_agent,
    sql_gen_agent,
    sql_check_agent,
    sql_execute_agent,
    final_answer_agent,
    rag_agent,
    query_interpreter,
)


# ============================================================
# 라우팅 함수들 - 조건부 분기를 처리하는 함수
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
        return "SQL_schema_agent"  # 직접 연결


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


# ============================================================
# Graph 빌드 - 에이전트를 연결하고 선언하는 단계 
# ============================================================

def build_graph():
    """
    Graph를 빌드하고 컴파일합니다.
    
    최적화된 파이프라인 구조:
    START → Query_interpreter → SQL_schema_agent → [RAG] → SQL_gen_agent 
    → [SQL_check] → SQL_execute_agent → Final_answer_agent → END
    
    Supervisor 제거로 20-40초 속도 향상
    """
    # StateGraph 생성
    graph_builder = StateGraph(AgentState)
    
    # 노드 추가 (Supervisor 제거)
    graph_builder.add_node(query_interpreter)
    graph_builder.add_node(sql_schema_agent)
    graph_builder.add_node(sql_gen_agent)
    graph_builder.add_node(sql_check_agent)
    graph_builder.add_node(sql_execute_agent)
    graph_builder.add_node(final_answer_agent)
    
    if RAG_AVAILABLE and rag_agent:
        graph_builder.add_node(rag_agent)
    
    # ============================================================
    # 엣지 추가 - 직접 연결 방식
    # ============================================================
    
    # 1. 시작: Query Interpreter
    graph_builder.add_edge(START, "Query_interpreter")
    
    # 2. Query Interpreter 분기
    graph_builder.add_conditional_edges(
        "Query_interpreter",
        route_from_interpreter,
        {
            "Final_answer_agent": "Final_answer_agent",  # 사용자 입력 필요 시
            "SQL_schema_agent": "SQL_schema_agent"  # 정상 쿼리
        }
    )
    
    # 3. Schema Agent 분기 (RAG vs SQL Gen)
    graph_builder.add_conditional_edges(
        "SQL_schema_agent",
        route_from_schema,
        {
            "SQL_gen_agent": "SQL_gen_agent",
            "RAG_agent": "RAG_agent" if RAG_AVAILABLE else "SQL_gen_agent"
        }
    )
    
    # 4. RAG → SQL Gen (옵션)
    if RAG_AVAILABLE:
        graph_builder.add_edge("RAG_agent", "SQL_gen_agent")
    
    # 5. SQL Gen 분기 (간단한 쿼리는 검증 스킵)
    graph_builder.add_conditional_edges(
        "SQL_gen_agent",
        should_check_query,
        {
            "SQL_check_agent": "SQL_check_agent",  # 복잡한 쿼리
            "SQL_execute_agent": "SQL_execute_agent"  # 간단한 쿼리
        }
    )
    
    # 6. Check → Execute
    graph_builder.add_edge("SQL_check_agent", "SQL_execute_agent")
    
    # 7. Execute → Final Answer (직접 연결! Supervisor 제거)
    graph_builder.add_edge("SQL_execute_agent", "Final_answer_agent")
    
    # 8. 종료
    graph_builder.add_edge("Final_answer_agent", END)
    
    # 컴파일
    compiled_graph = graph_builder.compile()
    
    logger.info("Graph compiled successfully (Supervisor-free optimized version)")
    
    return compiled_graph


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


# Graph 빌드 및 컴파일
_compiled_agent = build_graph()
agent = create_logged_agent(_compiled_agent)




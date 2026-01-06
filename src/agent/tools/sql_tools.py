"""
SQL 관련 도구

SQL 쿼리 검증, 실행, 테이블 추출 등을 처리합니다.
"""

import re
import time
from langchain_core.tools import tool
from ..config import db
from ...logger import logger


def validate_sql_query(query: str) -> tuple[bool, str | None]:
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


def extract_tables_from_query(query: str) -> list[str]:
    """SQL 쿼리에서 테이블명 추출"""
    tables = []
    from_pattern = r'FROM\s+(\w+)'
    join_pattern = r'JOIN\s+(\w+)'
    
    tables.extend(re.findall(from_pattern, query, re.IGNORECASE))
    tables.extend(re.findall(join_pattern, query, re.IGNORECASE))
    
    return list(set(tables))


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


@tool
def validate_sql_syntax(query: str) -> str:
    """
    Validate SQL query syntax and check for common mistakes.
    Returns the validated query or suggestions for fixes.
    
    Args:
        query: SQL query string to validate
        
    Returns:
        Validation result with the query or error messages
    """
    # 기본 검증
    is_valid, error_msg = validate_sql_query(query)
    
    if not is_valid:
        return f"SQL Validation Error: {error_msg}\n\nPlease fix the query and try again."
    
    # 일반적인 실수 체크
    warnings = []
    query_upper = query.upper()
    
    # NOT IN with potential NULL issues
    if "NOT IN" in query_upper:
        warnings.append("⚠️ NOT IN may have issues with NULL values. Consider using NOT EXISTS instead.")
    
    # Missing LIMIT
    if "LIMIT" not in query_upper:
        warnings.append("⚠️ Query without LIMIT clause may return too many results.")
    
    # UNION vs UNION ALL
    if "UNION" in query_upper and "UNION ALL" not in query_upper:
        warnings.append("💡 Consider UNION ALL if duplicates are acceptable (faster).")
    
    if warnings:
        return f"✅ Query is valid but has suggestions:\n" + "\n".join(warnings) + f"\n\nQuery: {query}"
    else:
        return f"✅ Query validation passed!\n\nQuery: {query}"


"""
Pydantic 모델 정의

Structured Output을 위한 Pydantic 모델들을 정의합니다.
"""

from typing import Literal, Optional, List
from pydantic import BaseModel, Field


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




# FastAPI 서버
from fastapi import FastAPI, Header, HTTPException, Query
from dotenv import load_dotenv
import pandas as pd
import os
from src.agent import agent, model_name, RAG_AVAILABLE
from pydantic import BaseModel
from langchain.schema import HumanMessage

# 환경변수 로드
load_dotenv()

# FastAPI 앱 초기화
app = FastAPI(
    title="네이버 주식 AI 에이전트 API",
    description="LangGraph 기반 멀티 에이전트 시스템으로 한국 주식 데이터를 자연어로 조회합니다.",
    version="2.0.0"
)


class AnswerResponse(BaseModel):
    answer: str
    model: str
    rag_available: bool


class HealthResponse(BaseModel):
    status: str
    model: str
    rag_available: bool
    langsmith_enabled: bool


@app.get("/", response_model=HealthResponse)
async def root():
    """
    API 상태를 확인합니다.
    """
    langsmith_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    
    return HealthResponse(
        status="healthy",
        model=model_name,
        rag_available=RAG_AVAILABLE,
        langsmith_enabled=langsmith_enabled
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    헬스 체크 엔드포인트
    """
    langsmith_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    
    return HealthResponse(
        status="healthy",
        model=model_name,
        rag_available=RAG_AVAILABLE,
        langsmith_enabled=langsmith_enabled
    )


@app.get("/agent", response_model=AnswerResponse)
async def agent_get(
    question: str = Query(..., description="질문 텍스트"),
    authorization: str = Header(None, alias="Authorization"),
    x_request_id: str = Header(None, alias="X-NCP-CLOVASTUDIO-REQUEST-ID"),
):
    """
    주식 관련 질문에 대한 답변을 생성합니다.
    
    Args:
        question: 사용자 질문
        authorization: 인증 헤더 (선택사항)
        x_request_id: 요청 ID (선택사항)
        
    Returns:
        AnswerResponse: 에이전트의 답변
        
    Raises:
        HTTPException: 에이전트 실행 실패 시
    """
    try:
        # 에이전트 실행
        init_state = {"messages": [HumanMessage(content=question)]}
        final_state = agent.invoke(init_state)

        # 최종 답변 추출
        if not final_state or "messages" not in final_state:
            raise HTTPException(
                status_code=500,
                detail="Agent did not return a valid response"
            )
        
        messages = final_state["messages"]
        if not messages:
            raise HTTPException(
                status_code=500,
                detail="Agent returned empty messages"
            )
        
        last_msg = messages[-1]
        
        # 메시지 내용 추출
        if hasattr(last_msg, "content"):
            answer_text = last_msg.content
        elif isinstance(last_msg, dict) and "content" in last_msg:
            answer_text = last_msg["content"]
        else:
            answer_text = str(last_msg)
        
        return AnswerResponse(
            answer=answer_text,
            model=model_name,
            rag_available=RAG_AVAILABLE
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


@app.post("/agent", response_model=AnswerResponse)
async def agent_post(
    question: str,
    authorization: str = Header(None, alias="Authorization"),
    x_request_id: str = Header(None, alias="X-NCP-CLOVASTUDIO-REQUEST-ID"),
):
    """
    주식 관련 질문에 대한 답변을 생성합니다 (POST 방식).
    
    Args:
        question: 사용자 질문
        authorization: 인증 헤더 (선택사항)
        x_request_id: 요청 ID (선택사항)
        
    Returns:
        AnswerResponse: 에이전트의 답변
    """
    return await agent_get(question, authorization, x_request_id)


@app.get("/info")
async def get_info():
    """
    시스템 정보를 반환합니다.
    """
    return {
        "system": "Naver Stock AI Agent",
        "version": "2.0.0",
        "model": model_name,
        "features": {
            "multi_agent": True,
            "rag": RAG_AVAILABLE,
            "langsmith": os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true",
            "supervisor": True,
            "query_interpreter": True,
        },
        "agents": [
            "Query_interpreter",
            "SQL_schema_agent",
            "SQL_gen_agent",
            "SQL_check_agent",
            "SQL_execute_agent",
            "RAG_agent" if RAG_AVAILABLE else None,
            "Final_answer_agent",
            "Supervisor"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("네이버 주식 AI 에이전트 API 서버 시작")
    print("=" * 60)
    print(f"모델: {model_name}")
    print(f"RAG 사용 가능: {RAG_AVAILABLE}")
    print(f"LangSmith: {os.getenv('LANGCHAIN_TRACING_V2', 'false')}")
    print("=" * 60)
    print("\n접속 URL: http://localhost:8000")
    print("API 문서: http://localhost:8000/docs")
    print("\n종료하려면 Ctrl+C를 누르세요.")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)


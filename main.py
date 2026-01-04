"""
한국 주식 AI 에이전트 - FastAPI + LangServe
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 에이전트 import
from src.agent import agent, model_name, RAG_AVAILABLE
from src.logger import logger

# FastAPI 앱 생성
app = FastAPI(
    title="Naver Stock AI Agent API",
    version="2.0.0",
    description="한국 주식 데이터를 자연어로 조회하는 AI 에이전트",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LangServe로 에이전트 라우트 추가
add_routes(
    app,
    agent,
    path="/agent",
    enabled_endpoints=["invoke", "stream", "batch", "playground"],
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
)

@app.get("/")
async def root():
    """API 정보"""
    return {
        "name": "Naver Stock AI Agent API",
        "version": "2.0.0",
        "model": model_name,
        "rag_available": RAG_AVAILABLE,
        "langsmith_enabled": os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true",
        "endpoints": {
            "invoke": "/agent/invoke",
            "stream": "/agent/stream",
            "batch": "/agent/batch",
            "playground": "/agent/playground",
            "docs": "/docs",
            "health": "/health"
        },
        "message": "LangServe 기반 API - Playground에서 테스트하세요!"
    }

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "model": model_name,
        "rag_available": RAG_AVAILABLE
    }

@app.get("/info")
async def get_info():
    """시스템 상세 정보"""
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
            "langserve": True,
            "streaming": True
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
    
    logger.info("=" * 60)
    logger.info("네이버 주식 AI 에이전트 API 서버 시작")
    logger.info("=" * 60)
    logger.info(f"모델: {model_name}")
    logger.info(f"RAG 사용 가능: {RAG_AVAILABLE}")
    logger.info(f"LangSmith: {os.getenv('LANGCHAIN_TRACING_V2', 'false')}")
    logger.info("=" * 60)
    logger.info("접속 URL: http://localhost:8000")
    logger.info("API 문서: http://localhost:8000/docs")
    logger.info("Playground: http://localhost:8000/agent/playground")
    logger.info("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

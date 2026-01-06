"""
에이전트 설정 및 초기화

모델, DB, 전역 상수 등을 초기화합니다.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_naver import ChatClovaX
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from ..db import set_db
from ..logger import logger

# 환경변수 로드
load_dotenv()

# LangSmith 추적 설정
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Naver-Stock-Agent")

# 전역 상수
MAX_RETRIES = 3

# API 키 확인
api_key_clova = os.getenv("CLOVASTUDIO_API_KEY")
api_key_openai = os.getenv("OPENAI_API_KEY")

if not api_key_clova and not api_key_openai:
    raise RuntimeError("CLOVASTUDIO_API_KEY 또는 OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# 모델 초기화
if api_key_openai:
    default_model = ChatOpenAI(
        model='gpt-5-nano', 
        api_key=api_key_openai, 
        temperature=0,
        timeout=30,
        max_retries=2
    )
    model_name = 'openai:gpt-5-nano'
    
    # SQL Gen Agent 전용 모델 (고성능)
    sql_gen_model = ChatOpenAI(
        model='gpt-5-mini',
        api_key=api_key_openai,
        temperature=0,
        timeout=30,
        max_retries=2
    )
elif api_key_clova:
    default_model = ChatClovaX(
        model='HCX-005', 
        api_key=api_key_clova, 
        max_tokens=4096, 
        temperature=0, 
        top_k=3,
        timeout=30
    )
    model_name = 'HCX-005'
    sql_gen_model = default_model  # Clova 사용시 동일 모델
else:
    raise RuntimeError("사용 가능한 LLM API 키가 없습니다.")

# DB 초기화
src_dir = Path(__file__).parent.parent
db_path = str(src_dir.parent / 'data' / 'stock_db.db')
db = set_db(db_path)

# SQLDatabaseToolkit 생성
sql_toolkit = SQLDatabaseToolkit(db=db, llm=default_model)
sql_tools = sql_toolkit.get_tools()

# SQL 도구 추출
list_tables_tool = next(tool for tool in sql_tools if tool.name == "sql_db_list_tables")
get_schema_tool = next(tool for tool in sql_tools if tool.name == "sql_db_schema")

# RAG 시스템 확인
try:
    from ..rag_setup import get_retriever_tool
    retriever_tool = get_retriever_tool()
    RAG_AVAILABLE = retriever_tool is not None
    if RAG_AVAILABLE:
        logger.info("RAG 시스템이 성공적으로 로드되었습니다.")
    else:
        logger.warning("RAG 시스템 로드 실패. RAG 기능 없이 실행됩니다.")
except ImportError:
    logger.warning("RAG 모듈을 찾을 수 없습니다. RAG 기능 없이 실행됩니다.")
    retriever_tool = None
    RAG_AVAILABLE = False
except Exception as e:
    logger.error(f"RAG 시스템 초기화 중 오류 발생: {e}")
    logger.warning("RAG 기능 없이 계속 진행합니다.")
    retriever_tool = None
    RAG_AVAILABLE = False


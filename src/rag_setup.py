"""
RAG (Retrieval Augmented Generation) 설정 모듈

이 모듈은 PDF 문서를 로드하고, 벡터스토어를 구축하며,
retriever tool을 생성하는 기능을 제공합니다.
"""

import os
from pathlib import Path
from typing import Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import create_retriever_tool
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 기본 설정
DEFAULT_PDF_PATH = "../docs/RAG_document_ver7.pdf"
CHROMA_DB_PATH = "../data/chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K = 10


def load_pdf_documents(pdf_path: str = DEFAULT_PDF_PATH) -> list:
    """
    PDF 파일을 로드하고 문서를 분할합니다.
    
    Args:
        pdf_path: PDF 파일 경로
        
    Returns:
        분할된 문서 리스트
    """
    # 현재 파일의 디렉토리를 기준으로 경로 계산
    current_dir = Path(__file__).parent
    
    # 여러 가능한 경로 시도
    possible_paths = [
        pdf_path,
        current_dir / pdf_path,
        current_dir.parent / "RAG_document_ver7.pdf",
        current_dir.parent / "RAG_document_ver6.pdf",
        current_dir.parent / "RAG_document_ver5.pdf",
    ]
    
    pdf_file = None
    for path in possible_paths:
        if Path(path).exists():
            pdf_file = str(path)
            print(f"✅ PDF 파일 발견: {pdf_file}")
            break
    
    if not pdf_file:
        raise FileNotFoundError(
            f"PDF 파일을 찾을 수 없습니다. 다음 경로를 확인하세요:\n" +
            "\n".join([f"  - {p}" for p in possible_paths])
        )
    
    # PDF 로드
    loader = PyPDFLoader(pdf_file)
    
    # 텍스트 분할기 생성
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    # 문서 로드 및 분할
    split_docs = loader.load_and_split(text_splitter)
    
    print(f"📄 총 {len(split_docs)}개의 문서 청크가 생성되었습니다.")
    
    return split_docs


def create_vectorstore(documents: list, persist_directory: str = CHROMA_DB_PATH) -> Chroma:
    """
    Chroma 벡터스토어를 생성합니다.
    
    Args:
        documents: 문서 리스트
        persist_directory: 벡터스토어 저장 경로
        
    Returns:
        Chroma 벡터스토어 인스턴스
    """
    # OpenAI API 키 확인
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY가 설정되지 않았습니다. "
            "RAG 기능을 사용하려면 .env 파일에 OpenAI API 키를 추가하세요."
        )
    
    # 임베딩 모델 생성
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    
    # Chroma 벡터스토어 생성
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"🗄️ 벡터스토어가 {persist_directory}에 생성되었습니다.")
    
    return vectorstore


def load_existing_vectorstore(persist_directory: str = CHROMA_DB_PATH) -> Optional[Chroma]:
    """
    기존 벡터스토어를 로드합니다.
    
    Args:
        persist_directory: 벡터스토어 저장 경로
        
    Returns:
        Chroma 벡터스토어 인스턴스 또는 None
    """
    if not Path(persist_directory).exists():
        return None
    
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            return None
        
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        print(f"✅ 기존 벡터스토어를 로드했습니다: {persist_directory}")
        return vectorstore
    except Exception as e:
        print(f"⚠️ 기존 벡터스토어 로드 실패: {e}")
        return None


def get_retriever_tool(force_rebuild: bool = False):
    """
    Retriever tool을 생성하고 반환합니다.
    
    Args:
        force_rebuild: True일 경우 기존 벡터스토어를 무시하고 새로 구축
        
    Returns:
        LangChain retriever tool
    """
    # 기존 벡터스토어 확인
    vectorstore = None
    if not force_rebuild:
        vectorstore = load_existing_vectorstore()
    
    # 벡터스토어가 없으면 새로 생성
    if vectorstore is None:
        print("📚 새로운 벡터스토어를 구축합니다...")
        try:
            documents = load_pdf_documents()
            vectorstore = create_vectorstore(documents)
        except FileNotFoundError as e:
            print(f"⚠️ {e}")
            print("⚠️ RAG 기능 없이 계속 진행합니다.")
            return None
        except RuntimeError as e:
            print(f"⚠️ {e}")
            return None
    
    # Retriever 생성
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    
    # Retriever Tool 생성
    retriever_tool = create_retriever_tool(
        retriever,
        name='pdf_search',
        description='Use this tool to search information from PDF document about technical analysis indicators and stock trading terms'
    )
    
    print("✅ Retriever tool이 성공적으로 생성되었습니다.")
    
    return retriever_tool


def rebuild_vectorstore():
    """벡터스토어를 강제로 재구축합니다."""
    print("🔄 벡터스토어를 재구축합니다...")
    
    # 기존 벡터스토어 삭제
    import shutil
    if Path(CHROMA_DB_PATH).exists():
        shutil.rmtree(CHROMA_DB_PATH)
        print(f"🗑️ 기존 벡터스토어를 삭제했습니다: {CHROMA_DB_PATH}")
    
    # 새로 구축
    return get_retriever_tool(force_rebuild=True)


if __name__ == "__main__":
    """
    직접 실행 시 벡터스토어를 구축합니다.
    
    사용법:
        python rag_setup.py
    """
    print("=" * 60)
    print("RAG 벡터스토어 구축 시작")
    print("=" * 60)
    
    try:
        # 벡터스토어 구축
        retriever_tool = rebuild_vectorstore()
        
        if retriever_tool:
            print("\n" + "=" * 60)
            print("✅ RAG 시스템이 성공적으로 구축되었습니다!")
            print("=" * 60)
            
            # 테스트 검색
            print("\n🧪 테스트 검색을 수행합니다...")
            test_query = "RSI가 뭐야?"
            result = retriever_tool.invoke(test_query)
            print(f"\n검색 결과 (상위 {min(3, len(result))}개):")
            for i, doc in enumerate(result[:3], 1):
                print(f"\n[문서 {i}]")
                print(doc[:200] + "..." if len(doc) > 200 else doc)
        else:
            print("\n⚠️ RAG 시스템 구축에 실패했습니다.")
            
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


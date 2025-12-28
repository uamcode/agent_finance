# 네이버 주식 AI 에이전트 v2.0

LangGraph 기반 멀티 에이전트 시스템으로 한국 주식(KOSPI/KOSDAQ) 데이터를 자연어로 조회할 수 있는 AI 에이전트입니다.

## 🎯 주요 기능

- ✅ **멀티 에이전트 아키텍처**: Supervisor 패턴으로 7개 전문 에이전트 협업
- ✅ **RAG 시스템**: PDF 문서 기반 기술적 분석 용어 설명
- ✅ **Query Interpreter**: 사용자 질문 자동 명확화
- ✅ **LangSmith 통합**: 실시간 에이전트 추적 및 모니터링
- ✅ **Streamlit UI**: 사용자 친화적인 웹 인터페이스
- ✅ **FastAPI 백엔드**: RESTful API 제공
- ✅ **멀티 LLM 지원**: OpenAI GPT / 네이버 ClovaX 선택 가능

## 🏗️ 시스템 아키텍처

```
[사용자] → [Streamlit UI / FastAPI]
            ↓
        [Supervisor Agent]
            ↓
    ┌───────────────────────┐
    │ Query Interpreter     │ → 질문 명확화
    │ SQL Schema Agent      │ → 스키마 조회 & 라우팅
    │ SQL Gen Agent         │ → SQL 쿼리 생성
    │ SQL Check Agent       │ → 쿼리 검증
    │ SQL Execute Agent     │ → 쿼리 실행
    │ RAG Agent            │ → 문서 검색
    │ Final Answer Agent    │ → 최종 답변 생성
    └───────────────────────┘
            ↓
    [LangSmith Dashboard] (모니터링)
```

## 📁 프로젝트 구조

```
Agent-yj/
├── main.py              # 메인 실행 파일 (Streamlit 런처)
├── streamlit_app.py     # Streamlit 웹 앱
├── api_server.py        # FastAPI 서버 (API 용도)
├── README.md            # 프로젝트 문서
├── requirements.txt     # 패키지 의존성
│
├── src/                 # 소스 코드
│   ├── agent.py         # 멀티 에이전트 시스템
│   ├── db.py            # 데이터베이스 생성/관리
│   └── rag_setup.py     # RAG 벡터스토어 설정
│
├── data/                # 데이터 저장소
│   ├── stock_db.db      # 주식 데이터 SQLite DB
│   └── chroma_db/       # RAG 벡터스토어
│
└── docs/                # 문서 및 참고 자료
    ├── env_example.txt  # 환경 변수 예시
    ├── image.png        # 프로젝트 이미지
    └── RAG_document_ver7.pdf  # RAG 참고 문서
```

## 📦 설치 방법

### 1. 가상환경 설정 (권장)

```bash
cd Agent-yj

# 가상환경 생성
python -m venv venv

# 가상환경 활성화
# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 환경 변수 설정

`docs/env_example.txt`를 `.env`로 복사하고 API 키를 입력하세요:

```bash
# Windows
copy docs\env_example.txt .env

# Linux/Mac
cp docs/env_example.txt .env
```

`.env` 파일 예시:
```
# 최소 하나 이상 필요
CLOVASTUDIO_API_KEY=your_clova_key_here
OPENAI_API_KEY=your_openai_key_here

# 선택사항
TAVILY_API_KEY=your_tavily_key_here

# LangSmith (선택사항 - 모니터링/디버깅)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=Naver-Stock-Agent
```

### 4. 데이터베이스 생성 (최초 1회, 중요!)

⚠️ **서버 시작 전에 미리 생성하세요** (시간 절약)

**방법 A: 기존 DB 파일 사용**
- `stock_db.db` 파일이 이미 있으면 이 단계 건너뛰기

**방법 B: 새로 생성**
```bash
python
```

Python 인터프리터에서:
```python
from src.db import make_db

# 기본값 사용 (2024-01-01 ~ 오늘, 약 5-10분)
make_db(market='ALL')

# 특정 기간 지정
make_db(market='ALL', date1='2024-12-01', date2='2024-12-27')

# 특정 날짜 하나만
make_db(market='ALL', date1='2024-12-27', date2='2024-12-27')

# KOSPI만
make_db(market='KOSPI', date1='2024-11-01')

exit()
```

### 5. RAG 벡터스토어 구축 (최초 1회)

RAG 시스템을 위한 벡터스토어를 구축합니다:

```bash
python -m src.rag_setup
```

완료되면 `data/chroma_db/` 폴더가 생성됩니다.

## 🚀 실행 방법

### 기본 실행 (권장 ⭐)

```bash
python main.py
```

브라우저가 자동으로 열리며 Streamlit 웹 앱이 시작됩니다!

### 또는 Streamlit 직접 실행

```bash
streamlit run streamlit_app.py
```

브라우저: `http://localhost:8501`

### FastAPI 서버 (API 용도)

```bash
python api_server.py
```

API 문서: `http://localhost:8000/docs`

### 에이전트 직접 테스트

```bash
python -m src.agent
```

### 다음 실행부터는

가상환경만 활성화하면 됩니다:

```bash
cd Agent-yj

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 이후 실행
python main.py
```

## 💬 사용 예시

### Streamlit UI
1. 웹 브라우저에서 앱 열기
2. 하단 입력창에 질문 입력:
   - "삼성전자의 최근 종가를 알려줘"
   - "거래량이 많은 상위 10개 종목은?"
   - "RSI가 30 이하인 종목을 찾아줘"
   - "골든크로스가 뭐야?" (RAG 시스템이 설명)

### FastAPI (curl)
```bash
curl "http://localhost:8000/agent?question=삼성전자의%20최근%20종가는?"
```

### Python 코드
```python
from src.agent import agent
from langchain.schema import HumanMessage

response = agent.invoke({
    "messages": [HumanMessage(content="삼성전자의 최근 종가는?")]
})

print(response["messages"][-1].content)
```

## 🗄️ 데이터베이스 구조

### Stock_Info 테이블
- `Stock_ticker`: 종목 코드 (예: 005930.KS)
- `Stock_Name`: 종목명 (예: 삼성전자)
- `Market`: 시장 (KOSPI/KOSDAQ)

### Stock_Prices 테이블
- `Stock_Name`: 종목명
- `date`: 날짜
- `open`, `high`, `low`, `close`: 시가/고가/저가/종가
- `volume`: 거래량
- `dividends`, `splits`: 배당/분할

## 🤖 에이전트 설명

| 에이전트 | 역할 |
|---------|------|
| **Query Interpreter** | 사용자 질문 분석 및 명확화 |
| **SQL Schema Agent** | DB 스키마 조회 및 RAG/SQL 라우팅 결정 |
| **SQL Gen Agent** | SQL 쿼리 생성 |
| **SQL Check Agent** | 쿼리 문법 및 로직 검증 |
| **SQL Execute Agent** | 검증된 쿼리 실행 |
| **RAG Agent** | PDF 문서에서 기술적 분석 용어 검색 |
| **Final Answer Agent** | 결과 포맷팅 및 한국어 답변 생성 |
| **Supervisor** | 에이전트 간 작업 조율 및 흐름 제어 |

## 📊 LangSmith 모니터링

LangSmith를 활성화하면 다음을 확인할 수 있습니다:

- 각 에이전트의 실행 순서 및 소요 시간
- LLM 호출 내역 및 토큰 사용량
- 쿼리 생성/검증/실행 과정
- 에러 발생 시 상세 디버깅 정보

**설정 방법:**
1. [LangSmith](https://smith.langchain.com/) 가입
2. API 키 발급
3. `.env` 파일에 추가:
   ```
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_key_here
   ```

## 📝 문제 해결

### 1. 가상환경 관련
```
'python' is not recognized / command not found
```
**해결:** 가상환경이 활성화되어 있는지 확인 (터미널에 `(venv)` 표시)
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. RAG 시스템 오류
```
⚠️ RAG 모듈을 찾을 수 없습니다
```
**해결:** `python rag_setup.py`를 실행하여 벡터스토어를 구축하세요.

### 3. API 키 오류
```
RuntimeError: CLOVASTUDIO_API_KEY 또는 OPENAI_API_KEY가 설정되지 않았습니다
```
**해결:** `.env` 파일에 최소 하나 이상의 LLM API 키를 입력하세요.

### 4. DB 파일 없음
```
FileNotFoundError: stock_db.db
```
**해결:** [4. 데이터베이스 생성](#4-데이터베이스-생성-최초-1회-중요) 단계를 진행하세요.
```python
from db import make_db
make_db(market='ALL', date1='2024-12-27')
```

### 5. Streamlit 포트 충돌
```
Port 8501 is already in use
```
**해결:** `streamlit run streamlit_app.py --server.port 8502`로 다른 포트 사용

### 6. 패키지 설치 오류
```
pip install 실패
```
**해결:** 
```bash
# pip 업그레이드
pip install --upgrade pip

# 캐시 삭제 후 재설치
pip cache purge
pip install -r requirements.txt
```

## 🔧 개발 정보

### 기술 스택
- **Backend**: Python 3.8+
- **LLM Framework**: LangChain, LangGraph
- **UI**: Streamlit
- **API**: FastAPI
- **DB**: SQLite
- **Vector DB**: ChromaDB
- **Monitoring**: LangSmith

### 주요 의존성
- `langchain==0.3.27`
- `langgraph==0.6.2`
- `streamlit>=1.31.0`
- `fastapi`
- `chromadb`
- `langchain-openai`
- `langchain-naver==0.1.0`

## 📄 라이선스

This project is for educational purposes.

## 🙏 감사의 말

- YahooQuery: 대용량 주식 데이터 제공
- LangChain/LangGraph: 에이전트 프레임워크
- 네이버 클로바 스튜디오: 한국어 LLM 지원

## 📧 문의

이슈나 질문이 있으시면 GitHub Issues를 이용해주세요.

---

**버전 히스토리:**
- v2.0.0 (2025-01): 멀티 에이전트 + RAG + LangSmith 통합
- v1.0.0 (2024-12): 초기 단일 에이전트 버전

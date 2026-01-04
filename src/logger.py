"""
구조화된 로깅 시스템
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
import json


# 로그 디렉토리 생성
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 로그 파일명 (날짜별)
LOG_FILE = LOG_DIR / f"agent_{datetime.now().strftime('%Y%m%d')}.log"


class StructuredFormatter(logging.Formatter):
    """JSON 형식 로깅"""
    
    def format(self, record):
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # 추가 필드
        if hasattr(record, 'agent_name'):
            log_data['agent_name'] = record.agent_name
        if hasattr(record, 'duration_ms'):
            log_data['duration_ms'] = record.duration_ms
        if hasattr(record, 'tokens_used'):
            log_data['tokens_used'] = record.tokens_used
        if hasattr(record, 'error_type'):
            log_data['error_type'] = record.error_type
        if hasattr(record, 'query'):
            log_data['query'] = record.query
        if hasattr(record, 'session_id'):
            log_data['session_id'] = record.session_id
            
        return json.dumps(log_data, ensure_ascii=False)


def setup_logger(name: str = "stock_agent", level: int = logging.INFO):
    """로거 설정"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 기존 핸들러 제거
    logger.handlers.clear()
    
    # 파일 핸들러 (JSON)
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(StructuredFormatter())
    logger.addHandler(file_handler)
    
    # 콘솔 핸들러 (사람이 읽기 쉬운 형식)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


# 전역 로거
logger = setup_logger()


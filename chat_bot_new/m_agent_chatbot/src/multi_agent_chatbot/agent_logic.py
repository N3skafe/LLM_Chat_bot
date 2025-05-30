from typing import List, Tuple, Annotated, Sequence, Literal, Dict, Any, Optional
from typing_extensions import TypedDict, NotRequired
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langgraph.graph import StateGraph, END
import operator
import json
import logging
import os
import sys
import gc
from datetime import datetime
from logging.handlers import RotatingFileHandler
import psutil
import stat
from functools import lru_cache
import requests
from requests.exceptions import Timeout, RequestException
import hashlib
from PIL import Image
import io
import re
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import time
import asyncio
import signal
import atexit
from pathlib import Path
import shutil
import tempfile
import traceback
from contextlib import contextmanager
import weakref

# 상수 정의
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
LOG_DIR = "logs"
HISTORY_FILE = "chat_history.json"
MAX_MEMORY_USAGE = 1000  # MB
SAVE_THRESHOLD = 10
CACHE_SIZE = 128
MAX_WORKERS = 4
REQUEST_TIMEOUT = 10  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
MAX_HISTORY_LENGTH = 1000
MAX_CONTEXT_LENGTH = 2000
CLEANUP_INTERVAL = 3600  # 1시간
TEMP_DIR = "temp"
MAX_TEMP_FILES = 100
MAX_TEMP_AGE = 86400  # 24시간
MAX_RESPONSE_TIME = 30  # seconds
MAX_QUEUE_SIZE = 1000
BACKUP_INTERVAL = 3600  # 1시간

# 전역 변수
# _executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
# _message_queue = Queue(maxsize=MAX_QUEUE_SIZE)
# _processing_thread = None
_event_loop = None
_cleanup_timer = None
_backup_timer = None
_instances = weakref.WeakSet()

@contextmanager
def timeout_context(seconds):
    """타임아웃 컨텍스트 매니저"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # 시그널 핸들러 설정
    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # 원래 핸들러로 복원
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)

def get_event_loop():
    """이벤트 루프를 가져오거나 생성합니다."""
    global _event_loop
    if _event_loop is None:
        _event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_event_loop)
    return _event_loop

def cleanup_resources():
    """리소스를 정리합니다."""
    try:
        logger.info("Cleaning up resources...")
        # 메시지 큐/스레드 관련 코드 제거
        # _stop_message_processor()
        if _event_loop:
            _event_loop.close()
        cleanup_temp_files()
        gc.collect()
        logger.info("Resource cleanup completed")
    except Exception as e:
        logger.error(f"Error during resource cleanup: {str(e)}")
        logger.error(traceback.format_exc())

def cleanup_temp_files():
    """임시 파일을 정리합니다."""
    try:
        temp_dir = Path(TEMP_DIR)
        if not temp_dir.exists():
            return

        current_time = time.time()
        files = list(temp_dir.glob("*"))
        
        # 파일 수 제한 확인
        if len(files) > MAX_TEMP_FILES:
            files.sort(key=lambda x: x.stat().st_mtime)
            for f in files[:-MAX_TEMP_FILES]:
                try:
                    f.unlink()
                except Exception as e:
                    logger.error(f"Error deleting temp file {f}: {str(e)}")
        
        # 오래된 파일 정리
        for f in files:
            try:
                if current_time - f.stat().st_mtime > MAX_TEMP_AGE:
                    f.unlink()
            except Exception as e:
                logger.error(f"Error deleting old temp file {f}: {str(e)}")
    except Exception as e:
        logger.error(f"Error during temp file cleanup: {str(e)}")
        logger.error(traceback.format_exc())

def on_shutdown(signum, frame):
    """종료 시 리소스를 정리합니다."""
    logger.info(f"Received signal {signum}, cleaning up...")
    cleanup_resources()
    sys.exit(0)

def setup_signal_handlers():
    """시그널 핸들러를 설정합니다."""
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, on_shutdown)
        signal.signal(signal.SIGTERM, on_shutdown)
    atexit.register(cleanup_resources)

# 시그널 핸들러 설정 함수 호출
setup_signal_handlers()

def backup_history():
    """대화 기록을 백업합니다."""
    try:
        history_file = Path(HISTORY_FILE)
        if not history_file.exists():
            return

        backup_dir = Path(LOG_DIR) / "backups"
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"history_{timestamp}.json"

        shutil.copy2(history_file, backup_file)
        logger.info(f"History backup created: {backup_file}")

        # 오래된 백업 파일 정리
        backup_files = list(backup_dir.glob("history_*.json"))
        if len(backup_files) > 5:  # 최대 5개의 백업 유지
            backup_files.sort(key=lambda x: x.stat().st_mtime)
            for f in backup_files[:-5]:
                f.unlink()
    except Exception as e:
        logger.error(f"Error during history backup: {str(e)}")
        logger.error(traceback.format_exc())

def schedule_backup():
    """주기적인 백업을 스케줄링합니다."""
    global _backup_timer
    if _backup_timer:
        _backup_timer.cancel()
    _backup_timer = threading.Timer(BACKUP_INTERVAL, backup_history)
    _backup_timer.daemon = True
    _backup_timer.start()

# 로깅 설정
def setup_logging():
    # 로그 디렉토리 생성 및 권한 설정
    log_dir = Path(LOG_DIR)
    if not log_dir.exists():
        log_dir.mkdir(mode=0o755, parents=True)
    
    # 로그 파일 권한 설정
    def set_log_file_permissions(filename):
        if os.path.exists(filename):
            os.chmod(filename, stat.S_IRUSR | stat.S_IWUSR)

    # 로그 포맷 설정
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 파일 핸들러 설정
    info_log = log_dir / 'info.log'
    error_log = log_dir / 'error.log'

    info_handler = RotatingFileHandler(
        str(info_log),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(log_format)
    set_log_file_permissions(str(info_log))

    error_handler = RotatingFileHandler(
        str(error_log),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(log_format)
    set_log_file_permissions(str(error_log))

    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)

    # 로거 설정
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(info_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging()

from .llm_config import AVAILABLE_MODELS, llm_general, llm_coding, llm_reasoning, llm_image
from .rag_handler import get_relevant_documents, query_pdf_content
from .image_handler import analyze_image_with_llm, extract_text_from_image
from .web_search import search_web, format_search_results
from PIL import Image


# --- Agent State ---
class AgentState(TypedDict):
    input_query: str
    chat_history: Annotated[Sequence[BaseMessage], operator.add]
    image_data: NotRequired[Optional[Image.Image]]
    image_analysis_result: NotRequired[Optional[str]]
    rag_context: NotRequired[Optional[str]]
    web_search_results: NotRequired[Optional[str]]
    selected_agent: Literal["coding_math", "reasoning", "general", "rag", "image_analysis_route", "web_search"]
    output_message: NotRequired[Optional[str]]
    intermediate_steps: NotRequired[List[str]]

# --- Nodes ---
def route_query_node(state: AgentState) -> AgentState:
    """쿼리 유형에 따라 다음 노드를 결정합니다."""
    query = state["input_query"].lower()
    image_data = state.get("image_data")
    
    logger.info(f"Routing query: {query}")
    logger.debug(f"Image data present: {bool(image_data)}")
    
    # 이미지 분석이 우선순위가 높을 경우
    if image_data is not None:
        logger.info("Routing to image analysis")
        return {"selected_agent": "image_analysis_route"}

    # RAG 사용 여부 판단 (PDF 문서 검색)
    if any(kw in query for kw in ["pdf", "문서", "내 파일", "내 자료", "찾아줘", "요약"]):
        logger.info("Routing to RAG")
        return {"selected_agent": "rag"}

    # 웹 검색이 필요한 경우 (최신 정보나 실시간 데이터가 필요한 경우)
    web_search_keywords = [
        # 시간 관련
        "현재", "지금", "요즘", "최근", "이번", "올해", "작년", "내년",
        # 상태/상황 관련
        "상태", "상황", "동향", "트렌드", "뉴스", "소식", "정보", "검색",
        # 특정 주제
        "가격", "시세", "환율", "주식", "날씨", "기후",
        # 영어 키워드
        "current", "latest", "news", "update"
    ]
    
    if any(kw in query for kw in web_search_keywords):
        logger.info("Routing to web search")
        return {"selected_agent": "web_search"}

    # 키워드 기반 라우팅
    if any(kw in query for kw in ["코드", "코딩", "프로그래밍", "알고리즘", "수학", "계산", "풀어줘"]):
        logger.info("Routing to coding/math agent")
        return {"selected_agent": "coding_math"}
    elif any(kw in query for kw in ["추론", "분석", "설명해줘", "왜", "어떻게 생각해"]):
        logger.info("Routing to reasoning agent")
        return {"selected_agent": "reasoning"}
    else:
        logger.info("Routing to general agent")
        return {"selected_agent": "general"}

def image_analysis_node(state: AgentState) -> AgentState:
    """이미지를 분석하고 결과를 상태에 저장합니다."""
    image = state.get("image_data")
    query = state["input_query"]
    
    logger.info("Processing image analysis request")
    logger.debug(f"Query: {query}")
    
    if not image:
        logger.warning("Image analysis requested but no image provided")
        return {"output_message": "이미지 분석을 요청했지만 이미지가 제공되지 않았습니다.", "image_analysis_result": None}

    analysis_prompt = query if query else "이 이미지에 대해 설명해주세요."
    
    logger.info(f"Analyzing image with prompt: {analysis_prompt}")
    try:
        # 이미지 텍스트 추출 시도
        extracted_text = extract_text_from_image(image)
        if extracted_text and extracted_text.strip():
            logger.info(f"Extracted text from image: {extracted_text[:200]}...")
            analysis_prompt = f"이미지에서 추출된 텍스트: {extracted_text}\n\n{analysis_prompt}"
        else:
            logger.info("No text extracted from image or OCR failed.")
        
        # 이미지 분석 수행
        analysis_result, error = analyze_image_with_llm(image, analysis_prompt)
        
        if error or not analysis_result or not analysis_result.strip():
            logger.error(f"Image analysis error: {error if error else '분석 결과 없음'}")
            return {
                "output_message": "이미지에서 텍스트를 추출하거나 분석하는 데 실패했습니다. 이미지가 명확한지 확인해 주세요.",
                "image_analysis_result": None,
                "intermediate_steps": state.get("intermediate_steps", []) + [f"Image analysis error: {error if error else '분석 결과 없음'}"]
            }
        
        logger.info("Image analysis completed successfully")
        logger.debug(f"Analysis result: {analysis_result[:200]}...")
        
        # 이미지 데이터 메모리 해제
        state["image_data"] = None
        gc.collect()
        
        return {
            "output_message": analysis_result,
            "image_analysis_result": analysis_result,
            "intermediate_steps": state.get("intermediate_steps", []) + [f"Image analysis result: {analysis_result[:200]}..."]
        }
    except Exception as e:
        error_msg = f"이미지 분석 중 오류가 발생했습니다: {str(e)}"
        logger.error(f"Image analysis error: {str(e)}")
        # 이미지 데이터 메모리 해제
        state["image_data"] = None
        gc.collect()
        return {
            "output_message": "이미지에서 텍스트를 추출하거나 분석하는 데 실패했습니다. 이미지가 명확한지 확인해 주세요.",
            "image_analysis_result": None,
            "intermediate_steps": state.get("intermediate_steps", []) + [f"Image analysis error: {str(e)}"]
        }

def rag_node(state: AgentState) -> AgentState:
    """RAG를 사용하여 컨텍스트를 검색하고 상태에 저장합니다."""
    query = state["input_query"]
    logger.info(f"Performing RAG search for: {query}")
    
    try:
        # get_relevant_documents 함수 사용
        relevant_docs = get_relevant_documents(query, k=3)
        
        if not relevant_docs:
            logger.warning("No relevant documents found")
            context = "관련 정보를 찾을 수 없습니다."
        else:
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            logger.info("RAG search completed successfully")
            logger.debug(f"RAG Context (first 200 chars): {context[:200]}")
        
        return {
            "rag_context": context,
            "intermediate_steps": state.get("intermediate_steps", []) + [f"RAG context retrieved: {context[:200]}..."]
        }
    except Exception as e:
        error_msg = f"RAG 검색 중 오류가 발생했습니다: {str(e)}"
        logger.error(f"RAG search error: {str(e)}")
        return {
            "rag_context": error_msg,
            "intermediate_steps": state.get("intermediate_steps", []) + [f"RAG error: {str(e)}"]
        }

def web_search_node(state: AgentState) -> AgentState:
    """웹 검색을 수행하고 결과를 상태에 저장합니다."""
    query = state["input_query"]
    logger.info(f"Performing web search for: {query}")
    
    try:
        search_results = search_web(query)  # timeout 매개변수 제거
        if not search_results:
            logger.warning("No web search results found")
            return {
                "output_message": "웹 검색 결과를 찾을 수 없습니다. 다른 키워드로 다시 시도해주세요.",
                "web_search_results": None,
                "intermediate_steps": state.get("intermediate_steps", []) + ["Web search: No results found"]
            }
        
        formatted_results = format_search_results(search_results)
        logger.info("Web search completed successfully")
        logger.debug(f"Web search results: {formatted_results[:200]}...")
        
        return {
            "web_search_results": formatted_results,
            "intermediate_steps": state.get("intermediate_steps", []) + [f"Web search results: {formatted_results[:200]}..."]
        }
    except Timeout:
        error_msg = "웹 검색 시간이 초과되었습니다. 잠시 후 다시 시도해주세요."
        logger.error("Web search timeout")
        return {
            "output_message": error_msg,
            "web_search_results": None,
            "intermediate_steps": state.get("intermediate_steps", []) + ["Web search: Timeout"]
        }
    except RequestException as e:
        error_msg = f"웹 검색 중 네트워크 오류가 발생했습니다: {str(e)}"
        logger.error(f"Web search network error: {str(e)}")
        return {
            "output_message": error_msg,
            "web_search_results": None,
            "intermediate_steps": state.get("intermediate_steps", []) + [f"Web search error: {str(e)}"]
        }
    except Exception as e:
        error_msg = f"웹 검색 중 오류가 발생했습니다: {str(e)}"
        logger.error(f"Web search error: {str(e)}")
        return {
            "output_message": error_msg,
            "web_search_results": None,
            "intermediate_steps": state.get("intermediate_steps", []) + [f"Web search error: {str(e)}"]
        }

# 유틸리티 함수
def retry_with_backoff(func, *args, **kwargs):
    """지수 백오프를 사용한 재시도 로직"""
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait_time = RETRY_DELAY * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time} seconds...")
            time.sleep(wait_time)

def schedule_cleanup():
    """주기적인 리소스 정리를 스케줄링합니다."""
    global _cleanup_timer
    if _cleanup_timer:
        _cleanup_timer.cancel()
    _cleanup_timer = threading.Timer(CLEANUP_INTERVAL, cleanup_resources)
    _cleanup_timer.daemon = True
    _cleanup_timer.start()

# 민감 정보 필터링
def filter_sensitive_info(text: str) -> str:
    """민감한 정보를 필터링합니다."""
    patterns = [
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
        (r'\b\d{3}[-.]?\d{3,4}[-.]?\d{4}\b', '[PHONE]'),
        (r'\b\d{6}[-]?\d{7}\b', '[ID_NUMBER]'),
    ]
    
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    return text

# 이미지 처리 유틸리티
def process_image(image: Image.Image) -> Optional[Image.Image]:
    """이미지를 처리하고 최적화합니다."""
    try:
        # 이미지 크기 확인
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format or 'JPEG')
        if img_byte_arr.tell() > MAX_IMAGE_SIZE:
            logger.warning("Image size exceeds limit, resizing...")
            # 이미지 크기 조정
            ratio = MAX_IMAGE_SIZE / img_byte_arr.tell()
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        logger.error(traceback.format_exc())
        return None

class ChatHistoryManager:
    def __init__(self):
        self.max_history_length = MAX_HISTORY_LENGTH
        self.history_file = Path(HISTORY_FILE)
        self.state = {
            "input_query": "",
            "chat_history": [],
            "image_data": None,
            "image_analysis_result": None,
            "rag_context": None,
            "web_search_results": None,
            "selected_agent": "general",
            "output_message": None,
            "intermediate_steps": []
        }
        self.save_counter = 0
        self.save_threshold = SAVE_THRESHOLD
        self._cache = {}
        self._lock = threading.Lock()
        logger.info("ChatHistoryManager initialized")
        # 메시지 큐/스레드 관련 코드 제거
        # _start_message_processor()
        schedule_cleanup()
        schedule_backup()
        _instances.add(self)
    
    def __del__(self):
        """소멸자에서 리소스 정리"""
        # 메시지 큐/스레드 관련 코드 제거
        # _stop_message_processor()
        self._cleanup_memory()
    
    @lru_cache(maxsize=CACHE_SIZE)
    def get_recent_history(self, n=6):
        """최근 대화 기록을 캐시와 함께 가져옵니다."""
        try:
            with self._lock:
                history = self.state["chat_history"][-n:]
                logger.debug(f"Retrieved {len(history)} recent messages")
                return history
        except IndexError as e:
            logger.error(f"Index error when getting recent history: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error getting recent history: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def _check_memory_usage(self):
        """메모리 사용량을 확인하고 필요한 경우 가비지 컬렉션을 수행합니다."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_usage = memory_info.rss / 1024 / 1024  # MB 단위
            
            if memory_usage > MAX_MEMORY_USAGE:
                logger.warning(f"High memory usage detected: {memory_usage:.2f}MB")
                self._cleanup_memory()
        except Exception as e:
            logger.error(f"Error checking memory usage: {str(e)}")
            logger.error(traceback.format_exc())

    def _cleanup_memory(self):
        """메모리 정리를 수행합니다."""
        try:
            with self._lock:
                # 캐시 정리
                self._cache.clear()
                self.get_recent_history.cache_clear()
                
                # 이미지 데이터 정리
                if self.state["image_data"]:
                    self.state["image_data"] = None
                
                # 가비지 컬렉션 수행
                gc.collect()
        except Exception as e:
            logger.error(f"Error during memory cleanup: {str(e)}")
            logger.error(traceback.format_exc())

    def add_message(self, role, content):
        """메시지를 히스토리에 추가합니다. (무한 반복 방지를 위해 큐에 다시 넣지 않음)"""
        try:
            with self._lock:
                # 히스토리 길이 체크 및 조정
                if len(self.state["chat_history"]) >= self.max_history_length:
                    remove_count = min(10, len(self.state["chat_history"]) - self.max_history_length + 1)
                    self.state["chat_history"] = self.state["chat_history"][remove_count:]
                    logger.debug(f"Removed {remove_count} oldest messages to maintain history length")
                
                message = {
                    "role": role,
                    "content": content,
                    "timestamp": datetime.now().isoformat()
                }
                self.state["chat_history"].append(message)
                logger.debug(f"Added message: {role} - {content[:100]}...")
                
                # 메시지를 큐에 추가하는 부분을 제거 (무한 반복 방지)
                # _message_queue.put((role, content))
                
                self._check_memory_usage()
        except Exception as e:
            logger.error(f"Error adding message to history: {str(e)}")
            logger.error(traceback.format_exc())

    def save_history(self):
        """대화 기록을 파일에 저장합니다."""
        if self.save_counter < self.save_threshold:
            self.save_counter += 1
            return

        try:
            with self._lock:
                # 임시 파일에 먼저 저장
                temp_file = self.history_file.with_suffix('.tmp')
                with temp_file.open('w', encoding='utf-8') as f:
                    json.dump(self.state["chat_history"], f, ensure_ascii=False, indent=2)
                
                # 임시 파일을 실제 파일로 이동
                temp_file.replace(self.history_file)
                
                logger.info(f"Saved {len(self.state['chat_history'])} messages to history")
                self.save_counter = 0
        except PermissionError as e:
            logger.error(f"Permission denied when saving history file: {str(e)}")
        except Exception as e:
            logger.error(f"Error saving chat history: {str(e)}")
            logger.error(traceback.format_exc())

    def clear_history(self):
        """대화 기록을 초기화합니다."""
        try:
            with self._lock:
                self.state["chat_history"] = []
                logger.info("Chat history cleared")
                self.save_history()
                self._cleanup_memory()
        except Exception as e:
            logger.error(f"Error clearing chat history: {str(e)}")
            logger.error(traceback.format_exc())

# ChatHistoryManager 인스턴스 생성
chat_history_manager = ChatHistoryManager()

def get_specialized_response(prompt: str, context: str) -> str:
    """특수 목적에 대한 전문적인 응답 생성"""
    try:
        # 시스템 프롬프트 생성
        system_prompt = f"""당신은 {context} 전문가입니다. 
        사용자의 요청에 대해 전문적이고 상세한 답변을 제공해주세요.
        필요한 경우 추가 정보를 요청하거나 단계별로 안내해주세요."""
        
        # 메시지 구성
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
        
        # LangChain LLM을 사용하여 응답 생성
        response = llm_reasoning.invoke(messages)
        
        return response.content.strip()
    except Exception as e:
        error_msg = f"특수 목적 응답 생성 중 오류 발생: {str(e)}"
        logger.error(error_msg)
        return "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다."

def handle_specialized_request(prompt: str, request_type: str) -> str:
    """특수 목적 요청 처리"""
    context_map = {
        "초안 작성": "문서 작성 및 편집",
        "여행 계획": "여행 계획 및 여행지 추천",
        "적금 상품": "금융 상품 및 투자",
        "번역": "다국어 번역",
        "PDF 분석": "문서 분석 및 요약",
        "웹 검색": "정보 검색 및 요약"
    }
    
    # 적금 상품 관련 요청인 경우 PDF 검색 우선 수행
    if request_type == "적금 상품":
        try:
            # PDF에서 관련 정보 검색
            relevant_docs = get_relevant_documents(prompt, k=3)
            if relevant_docs:
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                # PDF 검색 결과를 포함하여 응답 생성
                return get_specialized_response(f"다음은 PDF 문서에서 찾은 정보입니다:\n{context}\n\n이 정보를 바탕으로 {prompt}", "금융 상품 및 투자")
            else:
                logger.info("No relevant PDF documents found for financial product query")
        except Exception as e:
            logger.error(f"Error searching PDF for financial product: {str(e)}")
    
    # 일반적인 처리
    context = context_map.get(request_type, "일반 상담")
    return get_specialized_response(prompt, context)

def llm_call_node(state: AgentState) -> AgentState:
    """선택된 에이전트(LLM)를 호출하고 응답을 생성합니다."""
    try:
        agent_name = state["selected_agent"]
        query = state["input_query"]
        history = state["chat_history"]
        image_analysis_context = state.get("image_analysis_result")
        # 이미지 분석 결과가 있으면 rag_context는 사용하지 않음
        if image_analysis_context:
            rag_context = None
        else:
            rag_context = state.get("rag_context")
        web_search_context = state.get("web_search_results")

        # 현재 시간 포함
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        logger.info(f"Processing query with agent: {agent_name}")
        logger.debug(f"Query: {query}")
        logger.debug(f"Contexts - RAG: {bool(rag_context)}, Image: {bool(image_analysis_context)}, Web: {bool(web_search_context)}")

        # 민감 정보 필터링
        query = filter_sensitive_info(query)
        if rag_context:
            rag_context = filter_sensitive_info(rag_context)
        if web_search_context:
            web_search_context = filter_sensitive_info(web_search_context)

        # 시스템 프롬프트 설정 (현재 시간 포함)
        system_prompt = f"""현재 시간은 {now}입니다.\n당신은 사용자와 대화하는 AI 챗봇입니다. 다음 규칙을 따라주세요:\n\n1. 기본 응답 원칙:\n   - 사용자의 질문에 직접적으로 답변하세요\n   - 내부 생각이나 분석 과정을 출력하지 마세요\n   - 불필요한 주어(\"제가\", \"저는\" 등)를 사용하지 마세요\n   - 영어로 된 내부 생각을 출력하지 마세요\n\n2. 검색 결과 활용:\n   - 웹 검색이나 PDF 검색 결과가 있다면, 그 정보를 바탕으로 답변하세요\n   - 검색 결과를 자연스럽게 답변에 포함시키되, 출처를 명시하세요\n   - 예시: \"최근 뉴스에 따르면 [검색 결과 내용]입니다.\"\n   - 검색 결과가 없는 경우: \"관련 정보를 찾지 못했습니다.\"\n\n3. 정보 부족 시:\n   - 구체적으로 어떤 정보가 부족한지 알려주세요\n   - 예시: \"현재 날씨 정보가 필요합니다.\"\n   - 추가 정보를 요청할 때는 간단명료하게 하세요\n\n4. 오류 발생 시:\n   - 구체적인 오류 내용을 알려주세요\n   - 예시: \"이미지 분석 중 오류가 발생했습니다: 이미지 형식이 지원되지 않습니다.\"\n\n5. 인사 처리:\n   - 인사에는 간단한 인사로만 응답하세요\n   - 예시: \"안녕하세요\", \"반갑습니다\"\n\n6. 답변 형식:\n   - 검색 결과가 있는 경우:\n     * \"검색 결과에 따르면 [답변 내용]입니다.\"\n     * \"최근 정보에 의하면 [답변 내용]입니다.\"\n   - 일반 답변의 경우:\n     * 간단명료하게 직접 답변하세요\n   - 정보 부족 시:\n     * \"답변을 위해 [필요한 정보]가 필요합니다.\"\n   - 오류 발생 시:\n     * \"오류가 발생했습니다: [구체적인 오류 내용]\" """

        # 컨텍스트가 있는 경우 시스템 프롬프트에 추가
        if image_analysis_context or web_search_context or rag_context:
            contexts = []
            if image_analysis_context:
                contexts.append(f"이미지 분석: {image_analysis_context}")
            # 이미지 분석이 있으면 rag_context는 무시
            elif rag_context:
                contexts.append(f"문서 내용: {rag_context}")
            if web_search_context:
                contexts.append(f"웹 검색: {web_search_context}")
            system_prompt += f"\n\n참고할 정보:\n{' '.join(contexts)}"

        # 프롬프트 구성
        messages = [SystemMessage(content=system_prompt)]
        
        # 대화 기록 추가 (최근 3개만)
        if history:
            recent_history = chat_history_manager.get_recent_history(3)
            for msg in recent_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        
        # 현재 쿼리 추가
        messages.append(HumanMessage(content=query))

        # 디버깅을 위한 프롬프트 로깅
        logger.debug("\n=== Final prompt to LLM ===")
        for msg in messages:
            logger.debug(f"\n[{msg.type}]:\n{msg.content}")
        logger.debug("\n========================\n")

        # 모델 선택
        if web_search_context:
            llm = llm_reasoning
            model_name = "llama3.2:latest"
        else:
            llm = AVAILABLE_MODELS.get(agent_name, llm_general)
            if agent_name == "coding_math":
                model_name = "deepseek-r1:latest"
            elif agent_name == "reasoning":
                model_name = "llama3.2:latest"
            elif agent_name == "general":
                model_name = "qwen3:latest"
            elif agent_name == "image_analysis":
                model_name = "llava:7b"
            else:
                model_name = "qwen3:latest"

        logger.info(f"Selected model: {model_name}")

        # LLM 호출
        response = llm.invoke(messages)
        response_text = response.content.strip()
        
        # 응답 후처리
        # 내부 생각 태그 제거
        response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        response_text = re.sub(r'<thought>.*?</thought>', '', response_text, flags=re.DOTALL)
        response_text = re.sub(r'<reasoning>.*?</reasoning>', '', response_text, flags=re.DOTALL)
        
        # 내부 독백 패턴 제거
        patterns_to_remove = [
            # 영어 패턴
            r"^(Okay|Alright|Well|Let me|I need to|I'll|I will|I should|I must|I have to).*?[.!?]",
            r"^(I think|I believe|I would say|I can see|I understand).*?[.!?]",
            r"^(Based on|According to|Looking at|Considering).*?[.!?]",
            
            # 한국어 패턴
            r"^(제가|저는|내가|나는).*?[.!?]",
            r"^(생각해보니|살펴보니|확인해보니).*?[.!?]",
            r"^(먼저|우선|일단).*?[.!?]",
            r"^(그럼|자|이제).*?[.!?]",
            r"^(응답:|답변:|AI:|Assistant:|챗봇:).*?[.!?]",
            r"^(~라고 생각합니다|~라고 판단됩니다|~라고 보입니다).*?[.!?]",
            r"^(사용자가|사용자는|질문이|요청이).*?[.!?]",
            r"^(~에 대해|~에 대해서).*?[.!?]",
            r"^(~을|~를).*?[.!?]",
            r"^(~하겠습니다|~하겠어요).*?[.!?]"
        ]
        
        for pattern in patterns_to_remove:
            response_text = re.sub(pattern, "", response_text, flags=re.IGNORECASE)
        
        # 불필요한 공백 제거 및 정리
        response_text = re.sub(r'\s+', ' ', response_text).strip()
        
        # 빈 응답 처리
        if not response_text:
            response_text = "죄송합니다. 다시 한번 질문해주시겠어요?"
        
        # 응답 저장
        chat_history_manager.add_message("user", query)
        chat_history_manager.add_message("assistant", response_text)
        
        logger.info("Response generated successfully")
        logger.debug(f"Response: {response_text[:200]}...")
        
        return {
            "output_message": response_text,
            "intermediate_steps": state.get("intermediate_steps", []) + [f"LLM response: {response_text[:200]}..."]
        }
    except Exception as e:
        error_msg = f"응답 생성 중 오류가 발생했습니다: {str(e)}"
        logger.error(f"LLM 호출 중 오류 발생: {str(e)}")
        return {
            "output_message": error_msg,
            "intermediate_steps": state.get("intermediate_steps", []) + [f"LLM error: {str(e)}"]
        }


# --- Graph Definition ---
workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("query_router", route_query_node)
workflow.add_node("image_analyzer", image_analysis_node)
workflow.add_node("rag_retriever", rag_node)
workflow.add_node("web_searcher", web_search_node)
workflow.add_node("coding_math_agent", llm_call_node)
workflow.add_node("reasoning_agent", llm_call_node)
workflow.add_node("general_agent", llm_call_node)
workflow.add_node("final_llm_call", llm_call_node)

# 엣지 설정
workflow.set_entry_point("query_router")

def decide_next_step_after_routing(state: AgentState):
    if state["selected_agent"] == "image_analysis_route":
        return "image_analyzer"
    elif state["selected_agent"] == "rag":
        return "rag_retriever"
    elif state["selected_agent"] == "web_search":
        return "web_searcher"
    elif state["selected_agent"] == "coding_math":
        return "coding_math_agent"
    elif state["selected_agent"] == "reasoning":
        return "reasoning_agent"
    else:
        return "general_agent"

workflow.add_conditional_edges(
    "query_router",
    decide_next_step_after_routing,
    {
        "image_analyzer": "image_analyzer",
        "rag_retriever": "rag_retriever",
        "web_searcher": "web_searcher",
        "coding_math_agent": "coding_math_agent",
        "reasoning_agent": "reasoning_agent",
        "general_agent": "general_agent",
    }
)

def decide_after_preprocessing(state: AgentState):
    return "final_llm_call"

workflow.add_edge("image_analyzer", "final_llm_call")
workflow.add_edge("rag_retriever", "final_llm_call")
workflow.add_edge("web_searcher", "final_llm_call")

workflow.add_edge("coding_math_agent", END)
workflow.add_edge("reasoning_agent", END)
workflow.add_edge("general_agent", END)
workflow.add_edge("final_llm_call", END)

# 그래프 컴파일
app_graph = workflow.compile()

# 그래프 실행 함수
def run_graph(query: str, chat_history: List[Tuple[str, str]], image_pil: Optional[Image.Image] = None):
    lc_history = []
    for human, ai in chat_history:
        lc_history.append(HumanMessage(content=human))
        lc_history.append(AIMessage(content=ai))

    initial_state: AgentState = {
        "input_query": query,
        "chat_history": lc_history,
        "image_data": image_pil,
        "image_analysis_result": None,
        "rag_context": None,
        "web_search_results": None,
        "selected_agent": "general",
        "output_message": None,
        "intermediate_steps": []
    }
    
    final_state = app_graph.invoke(initial_state)
    
    # 응답 저장
    chat_history_manager.add_message("user", query)
    chat_history_manager.add_message("assistant", final_state.get("output_message", "죄송합니다. 답변을 생성하지 못했습니다."))
    
    return final_state.get("output_message", "죄송합니다. 답변을 생성하지 못했습니다.")
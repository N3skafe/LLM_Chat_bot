import sys
import os
from PIL import Image
from typing import List, Tuple, Optional
import streamlit as st
import tempfile
import time
import asyncio
import warnings
import logging
import hashlib
import signal
import atexit
import threading

# 로깅 설정
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

# 로거 초기화
logger = setup_logging()

# 모든 경고 메시지 무시
warnings.filterwarnings("ignore")

# 로깅 레벨 설정
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)

# Streamlit 설정
os.environ['STREAMLIT_SERVER_WATCH_DIRS'] = 'false'  # 파일 감시 비활성화
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'     # 헤드리스 모드 활성화

# 현재 파일의 절대 경로를 기준으로 상위 디렉토리를 sys.path에 추가
current_file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_file_dir)
sys.path.insert(0, parent_dir)

from multi_agent_chatbot.agent_logic import (
    run_graph,
    get_specialized_response,
    handle_specialized_request
)
from multi_agent_chatbot.rag_handler import (
    process_and_embed_pdf, 
    PDF_STORAGE_PATH,
    get_initialized_vectorstore,
    validate_pdf,
    cleanup_old_databases,
    RAGManager,
    rag_manager,
    query_pdf_content,
    get_processed_pdfs
)

# 이미지 캐싱을 위한 함수
@st.cache_data
def load_image(image_file):
    return Image.open(image_file)

@st.cache_data
def get_image_hash(image):
    return hashlib.md5(image.tobytes()).hexdigest()

# 비동기 이벤트 루프 설정
def setup_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

# 이벤트 루프 설정
loop = setup_event_loop()

# 페이지 설정 (반드시 첫 번째 Streamlit 명령어여야 함)
st.set_page_config(
    page_title="멀티 에이전트 AI OING",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일 정의
st.markdown("""
<style>
    /* 전체 페이지 스타일 */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0;
        transition: all 0.3s ease;
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%) !important;
        min-height: 100vh;
    }

    /* 테마별 스타일. 실제 색은 선택한 테마의 동적 CSS가 덮어씁니다. */
    .theme-oing-purple .chat-message.user .message,
    .theme-oing-purple .chat-message.user .message::before {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
    }

    .theme-oing-purple .chat-message.assistant .message {
        background-color: #f8fafc;
        color: #1e293b;
    }

    .theme-flame-red .chat-message.user .message,
    .theme-flame-red .chat-message.user .message::before {
        background: linear-gradient(135deg, #b91c1c 0%, #ea580c 100%);
        color: white;
    }

    .theme-flame-red .chat-message.assistant .message {
        background-color: #f8fafc;
        color: #1e293b;
    }

    .theme-amber .chat-message.user .message,
    .theme-amber .chat-message.user .message::before {
        background: linear-gradient(90deg, #d97706 0%, #f59e0b 100%);
        color: #1f2937;
    }

    .theme-amber .chat-message.assistant .message {
        background-color: #ffffff;
        color: #1f2937;
    }

    .theme-forest-green .chat-message.user .message,
    .theme-forest-green .chat-message.user .message::before {
        background: linear-gradient(90deg, #166534 0%, #15803d 100%);
        color: white;
    }

    .theme-forest-green .chat-message.assistant .message {
        background-color: #ffffff;
        color: #1f2937;
    }

    .theme-ocean-blue .chat-message.user .message,
    .theme-ocean-blue .chat-message.user .message::before {
        background: linear-gradient(90deg, #0369a1 0%, #0284c7 100%);
        color: white;
    }

    .theme-ocean-blue .chat-message.assistant .message {
        background-color: #f0f9ff;
        color: #0f172a;
    }

    /* 공통 스타일 */
    .chat-message {
        display: flex;
        align-items: flex-start;
        margin: 12px 0;
        max-width: 70%;
        position: relative;
        width: 100%;
        animation: fadeIn 0.3s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* 사용자 메시지 스타일 */
    .chat-message.user {
        margin-left: auto;
        flex-direction: row-reverse;
        justify-content: flex-start;
        padding-right: 0;
        width: 100%;
        gap: 8px;
    }
    
    .chat-message.user .message {
        border-radius: 20px;
        border-bottom-right-radius: 4px;
        padding: 14px 24px;
        margin-left: 0;
        font-size: 15px;
        line-height: 1.6;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        order: 2;
        max-width: calc(100% - 48px);
        transition: all 0.3s ease;
        position: relative;
    }
    
    .chat-message.user .message::before {
        content: '';
        position: absolute;
        right: -8px;
        bottom: 0;
        width: 20px;
        height: 20px;
        clip-path: polygon(0 0, 100% 100%, 0 100%);
    }
    
    .chat-message.user .avatar {
        order: 1;
        margin-right: 0;
        z-index: 1;
    }
    
    .chat-message.user .message:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
    }
    
    /* AI 메시지 스타일 */
    .chat-message.assistant {
        margin-right: auto;
        flex-direction: row;
        justify-content: flex-start;
        padding-left: 0;
        width: 100%;
    }
    
    .chat-message.assistant .message {
        border-radius: 20px;
        border-bottom-left-radius: 4px;
        padding: 14px 24px;
        margin-left: 12px;
        font-size: 15px;
        line-height: 1.6;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        order: 2;
        max-width: calc(100% - 48px);
        transition: all 0.3s ease;
    }
    
    .chat-message.assistant .message:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    }
    
    /* 아바타 스타일 */
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        background-color: white;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .chat-message .avatar:hover {
        transform: scale(1.1);
    }
    
    /* 입력 영역 스타일 */
    .bottom-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-top: 1px solid rgba(226, 232, 240, 0.8);
        z-index: 1000;
        box-shadow: 0 -4px 16px rgba(0, 0, 0, 0.05);
    }

    .bottom-content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 1.5rem;
    }

    /* 입력 필드 스타일 */
    .stTextInput > div > div > input {
        border-radius: 16px;
        padding: 16px 24px;
        border: 2px solid rgba(226, 232, 240, 0.8);
        background-color: rgba(255, 255, 255, 0.9);
        color: #1e293b;
        font-size: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
    }

    .stTextInput > div > div > input:focus {
        border-color: #6366f1;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2);
        outline: none;
    }

    /* 버튼 스타일 */
    .stButton > button {
        border-radius: 16px;
        padding: 12px 24px;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        transition: all 0.3s ease;
        font-size: 15px;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(99, 102, 241, 0.3);
    }

    /* 사이드바 스타일 */
    .css-1d391kg {
        padding: 2rem 1.5rem;
        background-color: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(226, 232, 240, 0.8);
        width: 300px !important;
    }

    /* 사이드바 헤더 스타일 */
    .sidebar-header {
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        border-bottom: 1px solid rgba(226, 232, 240, 0.8);
    }

    .sidebar-header img {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }

    .sidebar-header img:hover {
        transform: scale(1.05);
    }

    .sidebar-header h1 {
        font-size: 1.8rem;
        color: #1e293b;
        margin: 0;
        font-weight: 600;
    }

    /* 사이드바 섹션 스타일 */
    .sidebar-section {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
    }

    .sidebar-section:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    }

    .sidebar-section h2 {
        font-size: 1.2rem;
        color: #1e293b;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }

    /* 파일 업로더 스타일 */
    .stFileUploader > div {
        border-radius: 16px;
        border: 2px dashed rgba(226, 232, 240, 0.8);
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
    }

    .stFileUploader > div:hover {
        border-color: #6366f1;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.1);
    }

    /* 모델 정보 스타일 */
    .model-info {
        font-size: 1rem;
        color: #475569;
        line-height: 1.6;
    }

    .model-info strong {
        color: #1e293b;
    }

    /* 사용 팁 스타일 */
    .usage-tips {
        font-size: 1rem;
        color: #475569;
        line-height: 1.6;
    }

    .usage-tips li {
        margin-bottom: 0.8rem;
        position: relative;
        padding-left: 1.5rem;
    }

    .usage-tips li::before {
        content: "•";
        color: #6366f1;
        position: absolute;
        left: 0;
        font-size: 1.2rem;
    }

    /* 스크롤바 스타일 */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(226, 232, 240, 0.5);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(99, 102, 241, 0.5);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(99, 102, 241, 0.7);
    }

    /* 반응형 스타일 */
    @media (max-width: 1200px) {
        .stApp {
            max-width: 100%;
            padding: 0 1rem;
        }
        
        .main-container {
            max-width: 100%;
            border-radius: 0;
        }
        
        .bottom-content {
            max-width: 100%;
        }
        
        .chat-container {
            padding: 1.5rem;
        }
        
        .chat-message {
            max-width: 80%;
        }
    }

    @media (max-width: 768px) {
        .chat-container {
            padding: 1rem;
        }
        
        .chat-message {
            max-width: 90%;
        }
        
        .bottom-content {
            padding: 1rem;
        }
        
        .sidebar-header img {
            width: 80px;
            height: 80px;
        }
        
        .sidebar-header h1 {
            font-size: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# 데이터베이스 초기화
if not get_initialized_vectorstore():
    st.error("데이터베이스 초기화에 실패했습니다.")
    st.stop()

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "streaming" not in st.session_state:
    st.session_state.streaming = False
if "theme" not in st.session_state:
    st.session_state.theme = "OING PURPLE(기본 색상)"

def start_new_chat():
    """새로운 채팅을 시작하는 함수"""
    st.session_state.messages = []
    st.session_state.streaming = False

def cleanup_resources():
    """앱 종료 시 리소스를 정리합니다."""
    try:
        # RAG 시스템 정리
        cleanup_old_databases()
        
        # 세션 상태 정리
        if "messages" in st.session_state:
            del st.session_state.messages
        if "streaming" in st.session_state:
            del st.session_state.streaming
        if "theme" in st.session_state:
            del st.session_state.theme
        if "conversation_started" in st.session_state:
            del st.session_state.conversation_started
        
        print("리소스가 성공적으로 정리되었습니다.")
    except Exception as e:
        print(f"리소스 정리 중 오류 발생: {str(e)}")

def on_shutdown():
    """Streamlit 앱 종료 시 호출되는 콜백 함수"""
    cleanup_resources()

# Streamlit 종료 콜백 등록
st.runtime.scriptrunner.add_script_run_ctx().on_shutdown = on_shutdown

def process_pdf_upload(pdf_file):
    """PDF 파일 업로드 처리 함수"""
    if pdf_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            temp_file_path = tmp_file.name

        try:
            # PDF 파일 검증
            is_valid, error_message = validate_pdf(temp_file_path)
            if not is_valid:
                return f"'{pdf_file.name}' 파일 검증 실패: {error_message}"

            # PDF 처리
            success = process_and_embed_pdf(temp_file_path)
            if success:
                return f"'{pdf_file.name}' 파일이 성공적으로 처리되어 RAG DB에 추가되었습니다."
            else:
                # 실패 원인 확인
                pdf_id = None
                for pid, info in rag_manager.pdf_metadata.items():
                    if info["filename"] == pdf_file.name and info["status"] == "failed":
                        pdf_id = pid
                        break
                
                if pdf_id and "error" in rag_manager.pdf_metadata[pdf_id]:
                    return f"'{pdf_file.name}' 파일 처리 실패: {rag_manager.pdf_metadata[pdf_id]['error']}"
                else:
                    return f"'{pdf_file.name}' 파일 처리 중 알 수 없는 오류가 발생했습니다."
        except Exception as e:
            return f"'{pdf_file.name}' 파일 처리 중 오류가 발생했습니다: {str(e)}"
        finally:
            try:
                os.unlink(temp_file_path)
            except:
                pass  # 임시 파일 삭제 실패는 무시
    return "PDF 파일이 업로드되지 않았습니다."

def stream_response(response_text):
    """응답을 스트리밍하는 함수"""
    response_container = st.empty()
    full_response = ""
    
    for chunk in response_text.split():
        full_response += chunk + " "
        response_container.markdown(full_response + "▌")
        time.sleep(0.05)  # 스트리밍 효과를 위한 지연
    
    response_container.markdown(full_response)
    return full_response

def get_theme_colors(theme):
    """테마별 전체 색상 반환"""
    colors = {
        "OING PURPLE(기본 색상)": {
            "primary": "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
            "primary_solid": "#6366f1",
            "user_message": "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
            "assistant_message": "#f8fafc",
            "user_text": "white",
            "assistant_text": "#1e293b",
            "background": "linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%)",
            "container_bg": "rgba(255, 255, 255, 0.95)",
            "chat_area_bg": "rgba(248, 250, 252, 0.8)",
            "button": "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
            "button_text": "white",
            "sidebar_bg": "rgba(255, 255, 255, 0.95)",
            "border": "rgba(226, 232, 240, 0.8)"
        },
        "FLAME RED": {
            "primary": "linear-gradient(135deg, #b91c1c 0%, #ea580c 100%)",
            "primary_solid": "#b91c1c",
            "user_message": "linear-gradient(135deg, #b91c1c 0%, #ea580c 100%)",
            "assistant_message": "#f8fafc",
            "user_text": "white",
            "assistant_text": "#1e293b",
            "background": "linear-gradient(135deg, #fafafa 0%, #e4e4e4 100%)",
            "container_bg": "rgba(255, 255, 255, 0.95)",
            "chat_area_bg": "rgba(248, 249, 250, 0.8)",
            "button": "linear-gradient(135deg, #b91c1c 0%, #ea580c 100%)",
            "button_text": "white",
            "sidebar_bg": "linear-gradient(135deg, #b91c1c 0%, #ea580c 100%)",
            "border": "rgba(226, 232, 240, 0.8)"
        },
        "AMBER": {
            "primary": "linear-gradient(90deg, #d97706 0%, #f59e0b 100%)",
            "primary_solid": "#d97706",
            "user_message": "linear-gradient(90deg, #d97706 0%, #f59e0b 100%)",
            "assistant_message": "#ffffff",
            "user_text": "#1f2937",
            "assistant_text": "#1f2937",
            "background": "linear-gradient(135deg, #fffbeb 0%, #f5f5f4 100%)",
            "container_bg": "rgba(255, 255, 255, 0.95)",
            "chat_area_bg": "rgba(255, 255, 255, 0.8)",
            "button": "linear-gradient(90deg, #d97706 0%, #f59e0b 100%)",
            "button_text": "#1f2937",
            "sidebar_bg": "linear-gradient(135deg, #d97706 0%, #f59e0b 100%)",
            "border": "rgba(226, 232, 240, 0.8)"
        },
        "FOREST GREEN": {
            "primary": "linear-gradient(90deg, #166534 0%, #15803d 100%)",
            "primary_solid": "#166534",
            "user_message": "linear-gradient(90deg, #166534 0%, #15803d 100%)",
            "assistant_message": "#ffffff",
            "user_text": "white",
            "assistant_text": "#1f2937",
            "background": "linear-gradient(135deg, #f0fdf4 0%, #e5e7eb 100%)",
            "container_bg": "rgba(255, 255, 255, 0.95)",
            "chat_area_bg": "rgba(255, 255, 255, 0.8)",
            "button": "linear-gradient(90deg, #166534 0%, #15803d 100%)",
            "button_text": "white",
            "sidebar_bg": "linear-gradient(135deg, #166534 0%, #15803d 100%)",
            "border": "rgba(226, 232, 240, 0.8)"
        },
        "OCEAN BLUE": {
            "primary": "linear-gradient(90deg, #0369a1 0%, #0284c7 100%)",
            "primary_solid": "#0369a1",
            "user_message": "linear-gradient(90deg, #0369a1 0%, #0284c7 100%)",
            "assistant_message": "#f0f9ff",
            "user_text": "white",
            "assistant_text": "#0f172a",
            "background": "linear-gradient(135deg, #f0f9ff 0%, #e2e8f0 100%)",
            "container_bg": "rgba(255, 255, 255, 0.95)",
            "chat_area_bg": "rgba(240, 249, 255, 0.8)",
            "button": "linear-gradient(90deg, #0369a1 0%, #0284c7 100%)",
            "button_text": "white",
            "sidebar_bg": "linear-gradient(135deg, #0369a1 0%, #0284c7 100%)",
            "border": "rgba(226, 232, 240, 0.8)"
        }
    }
    return colors.get(theme, colors["OING PURPLE(기본 색상)"])

def get_theme_class(theme: str) -> str:
    """테마 이름을 CSS 클래스에 안전한 식별자로 바꿉니다."""
    return {
        "OING PURPLE(기본 색상)": "oing-purple",
        "FLAME RED": "flame-red",
        "AMBER": "amber",
        "FOREST GREEN": "forest-green",
        "OCEAN BLUE": "ocean-blue",
    }.get(theme, "oing-purple")

def initialize_session_state():
    """세션 상태 초기화"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "pdf_metadata" not in st.session_state:
        st.session_state.pdf_metadata = {}
    if "conversation_started" not in st.session_state:
        st.session_state.conversation_started = False
    if "streaming" not in st.session_state:
        st.session_state.streaming = False
    if "theme" not in st.session_state:
        st.session_state.theme = "OING PURPLE(기본 색상)"

def get_conversation_starters():
    """대화 스타터 목록 반환"""
    return {
        "📝 초안 작성하기": "안녕하세요! 어떤 주제의 초안을 작성하시겠습니까? 목적과 주요 내용을 알려주시면 도와드리겠습니다.",
        "✈️ 여행 계획 세우기": "어떤 여행을 계획하고 계신가요? 목적지, 기간, 예산 등을 알려주시면 맞춤형 여행 계획을 제안해드리겠습니다.",
        "💰 맞춤 적금 알아보기": "적금 상품을 찾고 계신가요? 목표 금액, 기간, 월 저축 가능 금액을 알려주시면 최적의 적금 상품을 추천해드리겠습니다.",
        "🌐 언어 번역하기": "어떤 언어로 번역이 필요하신가요? 원본 텍스트를 입력해주시면 정확한 번역을 제공해드리겠습니다.",
        "📚 PDF 내용 분석하기": "PDF 문서의 내용을 분석하고 싶으신가요? PDF를 업로드해주시면 주요 내용을 요약하고 질문에 답변해드리겠습니다.",
        "🔍 웹 검색 도우미": "어떤 정보를 찾고 계신가요? 검색어를 입력해주시면 관련 정보를 찾아드리겠습니다."
    }

def handle_conversation_starter(starter_text):
    """대화 스타터 처리"""
    st.session_state.conversation_started = True
    st.session_state.messages.append({"role": "assistant", "content": starter_text})
    return starter_text

def get_ai_response(prompt: str, image: Optional[Image.Image] = None) -> str:
    """AI 응답 생성"""
    try:
        # 대화 스타터 관련 키워드 확인
        starter_keywords = {
            "초안": "초안 작성",
            "여행": "여행 계획",
            "적금": "적금 상품",
            "번역": "번역",
            "PDF": "PDF 분석",
            "검색": "웹 검색"
        }
        
        # 키워드 기반으로 요청 유형 결정
        request_type = None
        for keyword, req_type in starter_keywords.items():
            if keyword in prompt:
                request_type = req_type
                break
        
        if request_type:
            # 특수 목적 요청 처리
            return handle_specialized_request(prompt, request_type)
        
        # PDF 관련 질문인 경우
        if hasattr(st.session_state, 'vector_store') and st.session_state.vector_store is not None:
            response = query_pdf_content(prompt)
            if response and "관련 정보를 찾지 못했습니다" not in response:
                return response
        
        # 일반 대화 처리
        return run_graph(prompt, [(m["content"], "") for m in st.session_state.messages if m["role"] == "user"], image_pil=image)
        
    except Exception as e:
        error_msg = f"AI 응답 생성 중 오류 발생: {str(e)}"
        logger.error(error_msg)
        return f"죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다. 오류 내용: {str(e)}"

def process_multiple_pdfs(files_to_process):
    """여러 PDF 파일을 처리합니다."""
    results = []
    for filename, content in files_to_process:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(content)
            temp_file_path = tmp_file.name

        try:
            # PDF 파일 검증
            is_valid, error_message = validate_pdf(temp_file_path)
            if not is_valid:
                results.append(f"'{filename}' 파일 검증 실패: {error_message}")
                continue

            # PDF 처리
            success = process_and_embed_pdf(temp_file_path, filename)
            if success:
                results.append(f"'{filename}' 파일이 성공적으로 처리되어 RAG DB에 추가되었습니다.")
            else:
                # 실패 원인 확인
                pdf_id = None
                for pid, info in rag_manager.pdf_metadata.items():
                    if info["filename"] == filename and info["status"] == "failed":
                        pdf_id = pid
                        break
                
                if pdf_id and "error" in rag_manager.pdf_metadata[pdf_id]:
                    results.append(f"'{filename}' 파일 처리 실패: {rag_manager.pdf_metadata[pdf_id]['error']}")
                else:
                    results.append(f"'{filename}' 파일 처리 중 알 수 없는 오류가 발생했습니다.")
        except Exception as e:
            results.append(f"'{filename}' 파일 처리 중 오류가 발생했습니다: {str(e)}")
        finally:
            try:
                os.unlink(temp_file_path)
            except:
                pass  # 임시 파일 삭제 실패는 무시
    return results

def main():
    try:
        # 세션 상태 초기화
        initialize_session_state()
        
        # 사이드바 설정
        with st.sidebar:
            st.title("Multi-Agency AI Secretary")
            
            # 사이드바 헤더
            st.markdown("""
            <div class="sidebar-header">
                <img src="https://i.postimg.cc/y8Jckyhh/big2.png" alt="Logo">
                <h1>인공지능 "오잉"</h1>
            </div>
            """, unsafe_allow_html=True)
            
            # 새로운 채팅 시작 버튼
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown('<h2>💬 채팅 관리</h2>', unsafe_allow_html=True)
            if st.button("새로운 채팅 시작", key="new_chat", use_container_width=True):
                start_new_chat()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            
            # RAG 설정
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown('<h2>📚 RAG 설정</h2>', unsafe_allow_html=True)
            
            # PDF 파일 업로드 (여러 파일 지원)
            pdf_files = st.file_uploader("PDF 파일 업로드", type=['pdf'], accept_multiple_files=True)
            if pdf_files:
                with st.spinner("PDF 처리 중..."):
                    # 파일 내용 수집
                    files_to_process = [(f.name, f.getvalue()) for f in pdf_files]
                    
                    # 여러 PDF 처리
                    results = process_multiple_pdfs(files_to_process)
                    
                    # 결과 표시
                    for result in results:
                        if "성공" in result:
                            st.success(result)
                        else:
                            st.error(result)
            
            # 처리된 PDF 목록 표시
            processed_pdfs = get_processed_pdfs()
            if processed_pdfs:
                st.markdown("### 처리된 PDF 목록")
                for pdf in processed_pdfs:
                    status_color = "green" if pdf["status"] == "processed" else "red"
                    st.markdown(f"- {pdf['filename']} <span style='color:{status_color}'>({pdf['status']})</span>", unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

            # 테마 선택
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown('<h2>🎨 테마 선택</h2>', unsafe_allow_html=True)
            themes = ["OING PURPLE(기본 색상)", "FLAME RED", "AMBER", "FOREST GREEN", "OCEAN BLUE"]
            if st.session_state.theme not in themes:
                st.session_state.theme = themes[0]
            theme = st.radio(
                "채팅 테마를 선택하세요",
                themes,
                index=themes.index(st.session_state.theme)
            )
            
            # 테마 변경 시 세션 상태 업데이트
            if theme != st.session_state.theme:
                st.session_state.theme = theme
                st.rerun()
                
            st.markdown('</div>', unsafe_allow_html=True)

            # 모델 정보
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown('<h2>🤖 모델 정보</h2>', unsafe_allow_html=True)
            st.markdown("""
            <div class="model-info">
                <p><strong>일반 질문</strong>: qwen3:latest</p>
                <p><strong>이미지/PDF</strong>: llava:7b</p>
                <p><strong>복잡한 추론/이미지</strong>: llama3.2:latest</p>
                <p><strong>코딩/수학</strong>: deepseek-r1:latest</p>
                <p><strong>임베딩</strong>: nomic-embed-text:latest</p>
                <p><strong>벡터DB</strong>: ChromaDB</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # 사용 팁
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown('<h2>💡 사용 팁</h2>', unsafe_allow_html=True)
            st.markdown("""
            <div class="usage-tips">
                <ul>
                    <li>PDF를 업로드하면 해당 내용 기반으로 답변합니다.</li>
                    <li>이미지와 함께 질문하면 이미지를 분석하여 답변에 활용합니다.</li>
                    <li>'코드 짜줘', '수학 문제 풀어줘' 등으로 특정 에이전트를 유도할 수 있습니다.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # 종료 버튼 추가
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown('<h2>⚙️ 시스템</h2>', unsafe_allow_html=True)
            if st.button("앱 종료", key="exit_app", use_container_width=True):
                cleanup_resources()
                st.stop()
            st.markdown('</div>', unsafe_allow_html=True)

        # 메인 컨테이너
        theme_class = get_theme_class(st.session_state.theme)
        theme_colors = get_theme_colors(st.session_state.theme)
        
        # 테마별 스타일 동적 적용
        st.markdown(f"""
        <style>
            /* 전체 페이지 스타일 */
            .stApp {{
                background: {theme_colors['background']} !important;
            }}

            /* 메인 컨테이너 스타일 */
            .main-container {{
                background-color: {theme_colors['container_bg']};
                border: 1px solid {theme_colors['border']};
            }}

            /* 채팅 영역 스타일 */
            .chat-container {{
                background-color: {theme_colors['chat_area_bg']};
                border-radius: 16px;
                padding: 20px;
                margin: 20px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            }}

            /* 대화 스타터 버튼 스타일 */
            .conversation-starter-button {{
                background: {theme_colors['button']};
                color: {theme_colors['button_text']};
                border: none;
                border-radius: 12px;
                padding: 12px 20px;
                margin: 8px;
                font-size: 14px;
                cursor: pointer;
                transition: all 0.3s ease;
                width: calc(50% - 16px);
                text-align: center;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            }}

            .conversation-starter-button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
            }}

            /* 채팅 메시지 스타일 */
            .chat-message.user .message {{
                background: {theme_colors['user_message']};
                color: {theme_colors['user_text']};
            }}

            .chat-message.user .message::before {{
                background: {theme_colors['user_message']};
            }}

            .chat-message.assistant .message {{
                background: {theme_colors['assistant_message']};
                color: {theme_colors['assistant_text']};
            }}

            /* 버튼 스타일 */
            .stButton > button {{
                background: {theme_colors['button']};
                color: {theme_colors['button_text']};
            }}

            /* 사이드바 스타일 */
            .css-1d391kg {{
                background-color: {theme_colors['sidebar_bg']};
                border-right: 1px solid {theme_colors['border']};
            }}

            /* 입력 필드 스타일 */
            .stTextInput > div > div > input:focus {{
                border-color: {theme_colors['primary_solid']};
                box-shadow: 0 4px 12px {theme_colors['primary_solid']}33;
            }}

            /* 파일 업로더 스타일 */
            .stFileUploader > div:hover {{
                border-color: {theme_colors['primary_solid']};
                box-shadow: 0 4px 12px {theme_colors['primary_solid']}1A;
            }}

            /* 스크롤바 스타일 */
            ::-webkit-scrollbar-thumb {{
                background: {theme_colors['primary_solid']}80;
            }}

            ::-webkit-scrollbar-thumb:hover {{
                background: {theme_colors['primary_solid']};
            }}
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown(f'<div class="main-container theme-{theme_class}">', unsafe_allow_html=True)
        
        # 채팅 메시지 표시 영역 (상단)
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # 메시지 표시를 위한 컨테이너
        messages_container = st.container()
        
        # 메시지 표시
        with messages_container:
            if not st.session_state.messages:
                st.markdown(f"""
                <div class="chat-message assistant">
                    <div class="avatar"><img src="https://i.postimg.cc/y8Jckyhh/big2.png" alt="Logo"></div>
                    <div class="message">
                        안녕하세요! 저는 멀티 에이전트 AI "OING"입니다. 🤖<br><br>
                        저는 다음과 같은 도움을 드릴 수 있습니다:<br>
                        • PDF 문서 기반 질문 답변 📚<br>
                        • 이미지 분석 및 설명 🖼️<br>
                        • 코딩 및 수학 문제 해결 💻<br>
                        • 일반적인 대화 및 질문 답변 💬<br><br>
                        무엇을 도와드릴까요?
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # 대화 스타터 버튼 표시
                st.markdown('<div class="conversation-starters">', unsafe_allow_html=True)
                conversation_starters = get_conversation_starters()
                
                # 2열 그리드로 버튼 배치
                col1, col2 = st.columns(2)
                for i, (title, response) in enumerate(conversation_starters.items()):
                    if i % 2 == 0:
                        if col1.button(title, key=f"starter_{i}", use_container_width=True):
                            handle_conversation_starter(response)
                    else:
                        if col2.button(title, key=f"starter_{i}", use_container_width=True):
                            handle_conversation_starter(response)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                for message in st.session_state.messages:
                    if message["role"] == "user":
                        st.markdown(f"""
                        <div class="chat-message user">
                            <div class="message">{message["content"]}</div>
                            <div class="avatar">👤</div>
                        </div>
                        """, unsafe_allow_html=True)
                        if "image" in message:
                            st.image(message["image"], width=300)
                    else:
                        st.markdown(f"""
                        <div class="chat-message assistant">
                            <div class="avatar">🤖</div>
                            <div class="message">{message["content"]}</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

        # 하단 고정 영역
        st.markdown('<div class="bottom-container">', unsafe_allow_html=True)
        st.markdown('<div class="bottom-content">', unsafe_allow_html=True)

        # 이미지 업로드
        st.markdown('<div class="image-uploader">', unsafe_allow_html=True)
        uploaded_image = st.file_uploader("이미지 업로드", type=['png', 'jpg', 'jpeg'])
        image = None
        if uploaded_image:
            image = load_image(uploaded_image)  # 캐시된 이미지 로드 함수 사용
            st.image(image, width=200)
        st.markdown('</div>', unsafe_allow_html=True)

        # 사용자 입력
        st.markdown('<div class="chat-input">', unsafe_allow_html=True)
        if prompt := st.chat_input("여기에 질문을 입력하세요..."):
            # 사용자 메시지 추가
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.conversation_started = True  # 대화 시작 상태로 변경
            
            # 메시지 표시 업데이트
            with messages_container:
                st.markdown(f"""
                <div class="chat-message user">
                    <div class="message">{prompt}</div>
                    <div class="avatar">👤</div>
                </div>
                """, unsafe_allow_html=True)
                if image:
                    st.session_state.messages[-1]["image"] = image
                    st.image(image, width=300)

            # 챗봇 응답 생성
            with st.spinner("생각 중..."):
                try:
                    response = get_ai_response(prompt, image=image)
                    # 스트리밍 응답
                    full_response = stream_response(response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                    # AI 응답 표시
                    with messages_container:
                        st.markdown(f"""
                        <div class="chat-message assistant">
                            <div class="avatar">🤖</div>
                            <div class="message">{full_response}</div>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"오류가 발생했습니다: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # bottom-content 닫기
        st.markdown('</div>', unsafe_allow_html=True)  # bottom-container 닫기
        st.markdown('</div>', unsafe_allow_html=True)  # main-container 닫기

    except Exception as e:
        st.error(f"오류가 발생했습니다: {str(e)}")
        cleanup_resources()
        st.stop()

if __name__ == "__main__":
    main()
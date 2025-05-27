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

from multi_agent_chatbot.agent_logic import run_graph
from multi_agent_chatbot.rag_handler import process_and_embed_pdf, PDF_STORAGE_PATH

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
    page_title="멀티 에이전트 AI 챗봇",
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

    /* Streamlit 기본 여백 제거 */
    .main .block-container {
        padding-top: 0 !important;
    }

    /* 빈 컨테이너 숨기기 */
    div[data-testid="stVerticalBlock"] > div:empty,
    div[data-testid="stVerticalBlock"] > div > div:empty,
    div[data-testid="stVerticalBlock"] > div > div > div:empty,
    div[data-testid="stVerticalBlock"] > div > div > div > div:empty,
    div[data-testid="stVerticalBlock"] > div > div > div > div > div:empty,
    div[data-testid="stVerticalBlock"] > div > div > div > div > div > div:empty {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        min-height: 0 !important;
        max-height: 0 !important;
        overflow: hidden !important;
    }

    /* 메인 컨테이너 스타일 */
    .main-container {
        display: flex;
        flex-direction: column;
        height: 100vh;
        max-width: 1200px;
        margin: 0 auto;
        position: relative;
        background-color: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        overflow: hidden;
    }

    /* 채팅 컨테이너 스타일 */
    .chat-container {
        flex: 1;
        overflow-y: auto;
        padding: 2rem;
        margin-bottom: 200px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        width: 100%;
        max-width: 1200px;
        margin-left: auto;
        margin-right: auto;
        scroll-behavior: smooth;
    }

    /* 채팅 메시지 스타일 */
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
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border-radius: 20px;
        border-bottom-right-radius: 4px;
        padding: 14px 24px;
        margin-left: 0;
        font-size: 15px;
        line-height: 1.6;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2);
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
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        clip-path: polygon(0 0, 100% 100%, 0 100%);
    }
    
    .chat-message.user .avatar {
        order: 1;
        margin-right: 0;
        z-index: 1;
    }
    
    .chat-message.user .message:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(99, 102, 241, 0.3);
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
        background-color: #f8fafc;
        color: #1e293b;
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

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "streaming" not in st.session_state:
    st.session_state.streaming = False

def start_new_chat():
    """새로운 채팅을 시작하는 함수"""
    st.session_state.messages = []
    st.session_state.streaming = False

def process_pdf_upload(pdf_file):
    """PDF 파일 업로드 처리 함수"""
    if pdf_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            temp_file_path = tmp_file.name

        try:
            success = process_and_embed_pdf(temp_file_path, original_filename=pdf_file.name)
            if success:
                return f"'{pdf_file.name}' 파일이 성공적으로 처리되어 RAG DB에 추가되었습니다."
            else:
                # 실패 원인 확인
                pdf_id = None
                for pid, info in pdf_metadata.items():
                    if info["filename"] == pdf_file.name and info["status"] == "failed":
                        pdf_id = pid
                        break
                
                if pdf_id and "error" in pdf_metadata[pdf_id]:
                    return f"'{pdf_file.name}' 파일 처리 실패: {pdf_metadata[pdf_id]['error']}"
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

def main():
    # 사이드바 설정
    with st.sidebar:
        # 사이드바 헤더
        st.markdown("""
        <div class="sidebar-header">
            <img src="https://i.postimg.cc/y8Jckyhh/big2.png" alt="Logo">
            <h1>인공지능 에이젼트 "오잉"</h1>
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
        pdf_file = st.file_uploader("PDF 파일 업로드", type=['pdf'])
        if pdf_file:
            with st.spinner("PDF 처리 중..."):
                status = process_pdf_upload(pdf_file)
                st.info(status)
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

    # 메인 컨테이너
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # 채팅 메시지 표시 영역 (상단)
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # 메시지 표시를 위한 컨테이너
    messages_container = st.container()
    
    # 메시지 표시
    with messages_container:
        if not st.session_state.messages:
            st.markdown(f"""
            <div class="chat-message assistant">
                <div class="avatar">🤖</div>
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
                response = run_graph(
                    prompt,
                    [(m["content"], "") for m in st.session_state.messages if m["role"] == "user"],
                    image
                )
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

if __name__ == "__main__":
    main()
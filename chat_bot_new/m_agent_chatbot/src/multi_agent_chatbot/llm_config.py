from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
# from langchain_ollama import ChatOllama, OllamaEmbeddings


# Ollama 서버 주소 (필요시 변경)
OLLAMA_BASE_URL = "http://localhost:11434"
TOKEN_CONTEXT_LENGTH = 4096

# 모델 이름 정의 (Ollama에 pull된 모델명과 일치해야 함)
MODEL_DEEPSEEK = "deepseek-r1:latest" # 코딩, 수학
MODEL_LLAMA3 = "llama3.2:latest" # 복잡한 추론, 이미지 분석 (llava 사용 고려)
MODEL_QWEN3 = "qwen3:latest" # 일반 질문
MODEL_EMBEDDING = "nomic-embed-text" # 임베딩용
MODEL_IMAGE_ANALYSIS = "llava:7b" # 이미지 분석 전용 모델 (llama3가 미지원시)

def get_llm(model_name: str, temperature: float = 0.1):
    """지정된 모델 이름으로 ChatOllama 인스턴스를 반환합니다."""
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=model_name,
        temperature=temperature,
        num_ctx=TOKEN_CONTEXT_LENGTH, # 컨텍스트 길이 설정
        # top_k=10, # 필요시 추가 파라미터 설정
        # top_p=0.9,
        # mirostat=1,
        # mirostat_tau=0.1,
        # mirostat_eta=0.1
    )

llm_coding = get_llm(MODEL_DEEPSEEK)
llm_reasoning = get_llm(MODEL_LLAMA3) # 이미지 분석도 이 모델로 시도
llm_general = get_llm(MODEL_QWEN3)
llm_image = get_llm(MODEL_IMAGE_ANALYSIS) # 별도 이미지 분석 모델

# 임베딩 모델
embeddings = OllamaEmbeddings(
    base_url=OLLAMA_BASE_URL,
    model=MODEL_EMBEDDING
)

# 사용 가능한 모델 목록 (라우팅 로직에서 사용)
AVAILABLE_MODELS = {
    "coding_math": llm_coding,
    "reasoning": llm_reasoning,
    "general": llm_general,
    "image_analysis": llm_image # 이미지 분석용
}

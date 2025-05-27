# 멀티 에이전트 AI 챗봇

Gradio, Ollama, LangChain, LangGraph, ChromaDB를 사용하여 구축된 멀티 에이전트 AI 챗봇입니다.

## 주요 기능

-   **멀티 LLM 에이전트**:
    -   코딩/수학 문제: `deepseek-coder` (또는 유사 모델)
    -   복잡한 추론: `llama3.2` (또는 유사 모델)
    -   일반 질문: `gemma:2b`
-   **Ollama 연동**: 로컬에서 LLM 모델 실행
-   **RAG (Retrieval Augmented Generation)**:
    -   PDF 파일 업로드 및 ChromaDB를 사용한 Vector DB 구축
    -   LangChain을 활용한 문서 검색 및 컨텍스트 기반 답변 생성
    -   PDF 내 JavaScript 코드를 Python 코드로 변환 시도 (LLM 활용)
-   **이미지 분석**: 사용자가 업로드한 이미지를 LLM(예: LLaVA)이 분석하고 답변에 활용
-   **대화 히스토리 기억**: 이전 대화 내용을 기억하고 이어감
-   **Gradio UI**: 사용하기 쉬운 웹 인터페이스
-   **Poetry 환경**: 의존성 관리 및 패키징

## 설치 및 실행

1.  **Poetry 설치**: [Poetry 공식 문서](https://python-poetry.org/docs/#installation) 참고
2.  **Ollama 설치 및 실행**: [Ollama 공식 사이트](https://ollama.com/) 참고
3.  **필요한 LLM 모델 다운로드**:
    ```bash
    ollama pull deepseek-coder:6.7b-instruct
    ollama pull llama3:8b
    ollama pull gemma:2b
    ollama pull nomic-embed-text # 임베딩용
    ollama pull llava:7b # 이미지 분석용 (llama3가 멀티모달 미지원 시)
    ```
    *참고: 모델명은 사용 가능한 최신 버전으로 확인하고 필요시 `llm_config.py`를 수정하세요.*
4.  **프로젝트 클론 및 의존성 설치**:
    ```bash
    git clone <repository_url>
    cd multi-agent-chatbot
    poetry install
    ```
5.  **애플리케이션 실행**:
    ```bash
    poetry run python src/multi_agent_chatbot/app.py
    ```
    또는
    ```bash
    poetry shell
    python src/multi_agent_chatbot/app.py
    ```
    웹 브라우저에서 `http://localhost:7860` (또는 터미널에 표시된 주소)로 접속합니다.

## 사용 방법

-   **일반 대화**: 질문을 입력하고 "전송" 버튼을 누릅니다.
-   **RAG 활용**:
    1.  "PDF 파일 업로드" 섹션에서 PDF 파일을 업로드합니다.
    2.  "PDF 처리 상태"에 성공 메시지가 표시되면, 해당 PDF 내용과 관련된 질문을 할 수 있습니다. (예: "업로드한 PDF 요약해줘")
-   **이미지 분석**:
    1.  "이미지 업로드" 섹션에서 이미지를 업로드합니다.
    2.  이미지와 관련된 질문을 함께 입력하거나, 이미지만 업로드하고 일반적인 분석을 요청할 수 있습니다. (예: "이 이미지에 뭐가 보여?")

## 프로젝트 구조

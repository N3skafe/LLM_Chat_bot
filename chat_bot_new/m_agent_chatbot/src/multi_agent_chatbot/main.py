import sys
import os
from PIL import Image # PIL.Image is used, so ensure it's imported
from typing import List, Tuple, Optional

# 현재 파일(app.py)의 절대 경로를 기준으로 상위 디렉토리(src)를 sys.path에 추가
# __file__ 은 현재 실행 중인 스크립트의 경로입니다.
# 가정: app.py가 project_root/app_interface/app.py 에 있고,
# multi_agent_chatbot 모듈이 project_root/multi_agent_chatbot/ 에 있는 경우
# 혹은 app.py가 project_root/src/app_interface/app.py 에 있고,
# multi_agent_chatbot 모듈이 project_root/src/multi_agent_chatbot/ 에 있는 경우
# 아래 코드는 project_root/ (또는 project_root/src/)를 sys.path에 추가합니다.
# 이렇게 하면 `from multi_agent_chatbot...` 임포트가 가능해집니다.
current_file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_file_dir)
sys.path.insert(0, parent_dir) # 상위 디렉토리를 sys.path에 추가

# 이제 multi_agent_chatbot 모듈을 임포트할 수 있습니다.
from multi_agent_chatbot.agent_logic import run_graph
from multi_agent_chatbot.rag_handler import process_and_embed_pdf, PDF_STORAGE_PATH

import gradio as gr
# from PIL import Image # 이미 위에서 임포트 함
# from typing import List, Tuple, Optional # 이미 위에서 임포트 함


# --- Gradio 인터페이스 ---
def chat_interface(message: str, history: List[Tuple[str, str]], image_upload: Optional[Image.Image]):
    """Gradio 챗봇 인터페이스 함수"""
    print(f"User query: {message}")
    if image_upload:
        # PIL.Image 객체는 .filename 속성이 없을 수 있습니다. 로깅 시 주의.
        print(f"Image uploaded: type={type(image_upload)}, size={image_upload.size if image_upload else 'N/A'}")

    # run_graph 함수는 PIL Image 객체를 직접 받도록 수정됨
    # 에러 핸들링을 추가하면 더 견고해집니다.
    try:
        response_text = run_graph(message, history, image_upload)
    except Exception as e:
        print(f"Error in run_graph: {e}")
        response_text = "죄송합니다, 요청을 처리하는 중 오류가 발생했습니다."
        # 개발 중에는 더 자세한 에러 메시지를 반환할 수도 있습니다.
        # import traceback
        # response_text = f"Error: {e}\n{traceback.format_exc()}"

    history.append((message, response_text))
    return "", history, None # 입력창 비우기, 업데이트된 히스토리, 이미지 업로드 초기화

def process_pdf_upload(pdf_file_obj): # 파라미터 이름을 명확히 (Gradio File 객체)
    """Gradio PDF 업로드 처리 함수"""
    if pdf_file_obj is not None:
        # Gradio File 컴포넌트는 임시 파일 객체를 반환하며, .name 속성으로 임시 파일 경로를 가집니다.
        temp_file_path = pdf_file_obj.name
        original_filename = os.path.basename(temp_file_path) # Gradio가 임시 파일에 원래 이름을 유지하지 않을 수 있음
                                                            # pdf_file_obj.orig_name 이나 다른 속성을 확인해야 할 수도 있음
                                                            # 보통은 temp_file_path의 basename이 임시 이름임.
                                                            # 사용자가 업로드한 실제 파일 이름을 얻고 싶다면,
                                                            # Gradio File 객체의 다른 속성을 확인하거나,
                                                            # 파일 이름을 별도로 입력받는 UI를 고려해야 할 수 있습니다.
                                                            # 여기서는 임시 파일 경로의 basename을 사용합니다.
        
        print(f"Processing PDF: {temp_file_path}")
        
        # (선택적) 업로드된 파일을 영구적인 위치로 복사/이동
        # if PDF_STORAGE_PATH: # PDF_STORAGE_PATH가 설정되어 있고, 파일을 영구 저장하고 싶을 때
        #     if not os.path.exists(PDF_STORAGE_PATH):
        #         os.makedirs(PDF_STORAGE_PATH)
        #     permanent_file_path = os.path.join(PDF_STORAGE_PATH, os.path.basename(pdf_file_obj.name)) # 원본 파일 이름 사용 시 주의
        #     try:
        #         import shutil
        #         shutil.copy(temp_file_path, permanent_file_path)
        #         print(f"Copied PDF to {permanent_file_path}")
        #         file_path_for_processing = permanent_file_path # 복사된 파일로 처리
        #     except Exception as e:
        #         print(f"Error copying PDF: {e}")
        #         return f"'{original_filename}' 파일 복사 중 오류 발생: {e}"
        # else:
        #     file_path_for_processing = temp_file_path # 임시 파일 경로 직접 사용

        file_path_for_processing = temp_file_path # 현재 코드는 임시 파일 경로를 직접 사용

        success = process_and_embed_pdf(file_path_for_processing)
        
        # 업로드된 파일의 실제 이름을 표시하기 위해 노력 (Gradio는 임시 이름을 줄 수 있음)
        # Gradio File 객체의 `name`은 임시 경로이고, `orig_name`은 원본 파일 이름일 수 있습니다.
        # 확인이 필요합니다. Gradio 버전에 따라 다를 수 있습니다.
        # 여기서는 `os.path.basename(temp_file_path)`가 임시파일의 이름이므로,
        # 사용자에게 보여줄 파일 이름은 `original_filename`이 더 적절할 수 있습니다.
        # 다만, `pdf_file_obj.orig_name`이 있다면 그것을 사용하는 것이 가장 좋습니다.
        display_filename = getattr(pdf_file_obj, 'orig_name', original_filename)


        if success:
            return f"'{display_filename}' 파일이 성공적으로 처리되어 RAG DB에 추가되었습니다."
        else:
            return f"'{display_filename}' 파일 처리 중 오류가 발생했습니다."
    return "PDF 파일이 업로드되지 않았습니다."

# Gradio 앱 구성
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 멀티 에이전트 AI 챗봇 (Ollama & RAG)")
    gr.Markdown("코딩/수학 문제는 `deepseek-r1`, 복잡한 추론은 `llama3`, 일반 질문은 `gemma`가 처리합니다. PDF를 업로드하여 RAG 기능을 사용할 수 있고, 이미지도 업로드하여 분석할 수 있습니다.")

    with gr.Row():
        with gr.Column(scale=3): # 채팅창 영역을 조금 더 넓게 조정 (예시)
            chatbot = gr.Chatbot(
                label="대화 내용",
                height=600,
                bubble_full_width=False,
                #avatar_images=(None, "https://gradio.app/images/logo.png") # (user, bot) 아바타 이미지 예시
            )
            
            with gr.Row():
                # 이미지 업로드 UI 개선: sources에 "clipboard" 추가하여 붙여넣기 지원
                image_input = gr.Image(type="pil", label="이미지 업로드 (선택)", #sources=["upload", "clipboard"],
                                       height=150, width=150, interactive=False)
                user_input = gr.Textbox(
                    label="질문 입력",
                    placeholder="여기에 질문을 입력하고 Enter를 누르거나 전송 버튼을 클릭하세요.",
                    scale=4 # Textbox가 이미지 옆에서 더 많은 공간을 차지하도록
                )
            
            submit_button = gr.Button("전송", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("## RAG 설정")
            # file_count="single" (기본값) 또는 "multiple"로 설정 가능
            pdf_upload = gr.File(label="PDF 파일 업로드 (RAG 학습용)", file_types=[".pdf"])
            pdf_status = gr.Textbox(label="PDF 처리 상태", interactive=False, lines=3) # 여러 줄 표시 가능하도록
            
            # .upload 이벤트는 파일 업로드가 "완료"되었을 때 트리거됩니다.
            pdf_upload.upload(fn=process_pdf_upload, inputs=pdf_upload, outputs=pdf_status)
            
            gr.Markdown("## 모델 정보")
            gr.Markdown(
                """
                - **라우팅 및 에이전트 관리**: LangGraph
                - **코딩/수학**: `deepseek-r1:lastest` (예시)
                - **복잡한 추론/이미지**: `llama3:8b` (예시, 이미지 분석은 `llava` 또는 multimodal `llama3` 변형)
                - **일반 질문**: `gemma:2b` (예시)
                - **임베딩**: `nomic-embed-text` (예시)
                - **벡터DB**: ChromaDB (예시, `rag_handler.py`에 따라 다름)
                - **토큰 컨텍스트**: 모델별 상이 (예: 4096 ~ 8192+)
                """
            )
            gr.Markdown("---")
            gr.Markdown("### 사용 팁:\n"
                        "- PDF를 업로드하면 해당 내용 기반으로 답변합니다.\n"
                        "- 이미지와 함께 질문하면 이미지를 분석하여 답변에 활용합니다.\n"
                        "- '코드 짜줘', '수학 문제 풀어줘' 등으로 특정 에이전트를 유도할 수 있습니다.\n"
                        "- 이미지 업로드 후에는 자동으로 질문과 함께 전송됩니다. 이미지를 제거하려면 'X' 버튼을 누르세요.")


    # 이벤트 핸들러 연결
    # Textbox에서 엔터키 입력 시
    user_input.submit(
        fn=chat_interface,
        inputs=[user_input, chatbot, image_input],
        outputs=[user_input, chatbot, image_input] # 이미지 입력창도 초기화
    )
    # 버튼 클릭 시
    submit_button.click(
        fn=chat_interface,
        inputs=[user_input, chatbot, image_input],
        outputs=[user_input, chatbot, image_input] # 이미지 입력창도 초기화
    )

def main():
    # server_name="0.0.0.0"으로 설정하면 Docker 내부 또는 외부 네트워크에서 접근 가능
    # share=True는 임시 공개 링크를 생성 (디버깅 및 간단한 공유에 유용)
    demo.launch(server_name="0.0.0.0", share=True, debug=True) # debug=True 추가 시 유용

if __name__ == "__main__":
    # PDF_STORAGE_PATH 디렉토리 존재 확인 및 생성 (선택적 파일 복사 기능을 사용할 경우)
    # if PDF_STORAGE_PATH and not os.path.exists(PDF_STORAGE_PATH):
    # try:
    # os.makedirs(PDF_STORAGE_PATH)
    # print(f"Created PDF storage directory: {PDF_STORAGE_PATH}")
    # except OSError as e:
    # print(f"Error creating PDF storage directory {PDF_STORAGE_PATH}: {e}")
    # sys.exit(1) # 디렉토리 생성 실패 시 종료할 수 있음

    main()
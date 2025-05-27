from PIL import Image
from typing import Optional, List, Tuple, Union
from langchain_core.messages import HumanMessage, SystemMessage
import base64
import io
import logging
import pytesseract
from PIL import ImageEnhance, ImageFilter
import hashlib
from functools import lru_cache
import os
from pathlib import Path
import requests
import json
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .llm_config import llm_image

# 로깅 설정
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# 이미지 캐시
image_cache = {}

# 이미지 저장 경로
IMAGE_STORAGE_PATH = str(Path(__file__).parent.parent.parent / "data" / "images")
os.makedirs(IMAGE_STORAGE_PATH, exist_ok=True)

@lru_cache(maxsize=100)
def get_image_hash(image_bytes: bytes) -> str:
    """이미지의 해시값을 생성합니다."""
    return hashlib.md5(image_bytes).hexdigest()

def convert_image_format(image: Image.Image) -> Image.Image:
    """
    이미지 형식을 변환하여 처리 가능한 상태로 만듭니다.
    """
    try:
        # RGBA 이미지를 RGB로 변환
        if image.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            return background
        # 그레이스케일 이미지를 RGB로 변환
        elif image.mode in ('L', '1'):
            return image.convert('RGB')
        # 이미 RGB인 경우 그대로 반환
        elif image.mode == 'RGB':
            return image
        # 기타 형식은 RGB로 변환
        else:
            return image.convert('RGB')
    except Exception as e:
        logger.error(f"이미지 형식 변환 중 오류 발생: {str(e)}")
        return image.convert('RGB')

def optimize_image(image: Image.Image) -> Image.Image:
    """
    이미지를 최적화하여 처리 속도를 향상시킵니다.
    """
    try:
        # 이미지 형식 변환
        image = convert_image_format(image)
        
        # 이미지 크기 최적화
        max_size = 600
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    except Exception as e:
        logger.error(f"이미지 최적화 중 오류 발생: {str(e)}")
        return image

def extract_text_from_image(image: Image.Image) -> str:
    """
    이미지에서 텍스트를 추출합니다.
    """
    try:
        # 이미지 최적화
        optimized_image = optimize_image(image)
        
        # OCR 설정 (빠른 모드)
        custom_config = r'--oem 1 --psm 6 -l kor+eng'
        
        # OCR 수행
        text = pytesseract.image_to_string(optimized_image, config=custom_config)
        return text.strip()
    except Exception as e:
        logger.error(f"텍스트 추출 중 오류 발생: {str(e)}")
        return ""

def encode_image_to_base64(image_path: str) -> str:
    """이미지 파일을 base64로 인코딩합니다."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        return ""

def save_uploaded_image(image_data: bytes, filename: str) -> str:
    """업로드된 이미지를 저장하고 경로를 반환합니다."""
    try:
        # 파일명에서 확장자 추출
        _, ext = os.path.splitext(filename)
        if not ext:
            ext = '.png'  # 기본 확장자
        
        # 고유한 파일명 생성
        timestamp = int(time.time())
        unique_filename = f"{timestamp}{ext}"
        file_path = os.path.join(IMAGE_STORAGE_PATH, unique_filename)
        
        # 이미지 저장
        with open(file_path, 'wb') as f:
            f.write(image_data)
        
        return file_path
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        return ""

def analyze_image_with_llm(image_path_or_image: Union[str, Image.Image], query: Optional[str] = None) -> Tuple[str, str]:
    """
    LLM을 사용하여 이미지를 분석합니다.
    Args:
        image_path_or_image: 이미지 파일 경로(str) 또는 PIL Image 객체
        query: 이미지 분석을 위한 질문 (선택사항)
    Returns:
        Tuple[str, str]: (분석 결과, 오류 메시지)
    """
    try:
        # PIL Image 객체인 경우 base64로 직접 인코딩
        if isinstance(image_path_or_image, Image.Image):
            buffered = io.BytesIO()
            image_path_or_image.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        else:
            # 이미지 파일 경로인 경우 파일에서 읽어서 인코딩
            base64_image = encode_image_to_base64(image_path_or_image)
            
        if not base64_image:
            return "", "이미지 인코딩에 실패했습니다."

        # 시스템 프롬프트 설정
        system_prompt = """당신은 이미지 분석 전문가입니다. 주어진 이미지를 자세히 분석하고 설명해주세요.
        이미지에 있는 텍스트, 객체, 사람, 장면 등을 모두 포함하여 설명해주세요.
        가능한 구체적이고 상세하게 설명해주세요."""

        # 사용자 프롬프트 생성
        if query:
            user_prompt = f"다음 질문에 대해 이미지를 분석해주세요: {query}"
        else:
            user_prompt = "이 이미지를 자세히 분석해주세요."

        # 메시지 생성
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
            ])
        ]

        # LLM 호출
        response = llm_image.invoke(messages)
        
        # 응답 파싱
        if hasattr(response, 'content'):
            return response.content, ""
        else:
            return str(response), ""

    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        return "", f"이미지 분석 중 오류가 발생했습니다: {str(e)}"

def get_image_analysis_prompt(image_path: str, query: Optional[str] = None) -> str:
    """
    이미지 분석을 위한 프롬프트를 생성합니다.
    """
    try:
        # 이미지를 base64로 인코딩
        base64_image = encode_image_to_base64(image_path)
        if not base64_image:
            return ""

        # 프롬프트 생성
        if query:
            prompt = f"다음 질문에 대해 이미지를 분석해주세요: {query}\n\n이미지: data:image/jpeg;base64,{base64_image}"
        else:
            prompt = f"이 이미지를 자세히 분석해주세요.\n\n이미지: data:image/jpeg;base64,{base64_image}"

        return prompt

    except Exception as e:
        logger.error(f"Error creating image analysis prompt: {e}")
        return ""
import base64
from PIL import Image
import io
import re
import logging
import time
import hashlib
import os
from typing import Any, Dict, Optional, Callable
from functools import lru_cache
from datetime import datetime

def pil_to_base64(image: Image.Image) -> str:
    """PIL Image 객체를 Base64 문자열로 변환합니다."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG") # 또는 JPEG
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_javascript_from_text(text: str) -> list[str]:
    """텍스트에서 JavaScript 코드를 추출합니다."""
    # 간단한 <script> 태그 기반 추출. 더 복잡한 패턴이 필요할 수 있음.
    js_blocks = re.findall(r'<script.*?>(.*?)</script>', text, re.DOTALL)
    
    # 추가적으로 ```javascript ... ``` 와 같은 마크다운 코드 블록도 고려
    markdown_js_blocks = re.findall(r'```javascript\s*\n(.*?)\n```', text, re.DOTALL)
    
    return js_blocks + markdown_js_blocks

def convert_js_to_python_code(js_code: str, llm) -> str:
    """LLM을 사용하여 JavaScript 코드를 Python 코드로 변환합니다."""
    prompt = f"""
    You are an expert JavaScript to Python code converter.
    Convert the following JavaScript code to Python.
    Provide only the Python code as output, without any explanations or surrounding text.

    JavaScript Code:
    ```javascript
    {js_code}
    ```

    Python Code:
    """
    try:
        response = llm.invoke(prompt)
        # response가 AIMessage 객체일 경우 content 속성 사용
        python_code = response.content if hasattr(response, 'content') else str(response)
        # Python 코드만 추출 (```python ... ``` 형식 제거)
        match = re.search(r'```python\s*\n(.*?)\n```', python_code, re.DOTALL)
        if match:
            return match.group(1).strip()
        return python_code.strip() # LLM이 코드만 반환했을 경우
    except Exception as e:
        print(f"Error converting JS to Python: {e}")
        return f"# Error converting JavaScript to Python: {e}\n# Original JavaScript:\n# {js_code}"

# 로깅 설정
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

# 전역 로거 생성
logger = setup_logging()

class CacheManager:
    def __init__(self, max_size: int = 1000, cache_duration: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.cache_duration = cache_duration
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.cache_duration:
                return entry["result"]
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]
        
        self.cache[key] = {
            "result": value,
            "timestamp": time.time()
        }

class FileManager:
    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def ensure_directory_exists(self, directory_path: str):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
    
    def cleanup_old_files(self, max_age_days: int = 1):
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        for filename in os.listdir(self.base_path):
            filepath = os.path.join(self.base_path, filename)
            if os.path.getmtime(filepath) < current_time - max_age_seconds:
                try:
                    os.remove(filepath)
                    logger.info(f"Removed old file: {filename}")
                except Exception as e:
                    logger.error(f"Error removing file {filename}: {e}")

class ErrorHandler:
    @staticmethod
    def handle_error(error: Exception, context: str = "") -> str:
        error_msg = f"{context}: {str(error)}" if context else str(error)
        logger.error(error_msg)
        return error_msg
    
    @staticmethod
    def retry_operation(operation: Callable, max_retries: int = 3, delay: int = 1):
        for attempt in range(max_retries):
            try:
                return operation()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay)

# 유틸리티 함수들
@lru_cache(maxsize=100)
def get_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()

def format_timestamp() -> str:
    return datetime.now().isoformat()

def validate_result(result: str, min_length: int = 10) -> bool:
    if not result or not result.strip():
        return False
    if len(result) < min_length:
        return False
    return True

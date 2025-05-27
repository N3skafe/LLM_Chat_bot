import base64
from PIL import Image
import io
import re

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

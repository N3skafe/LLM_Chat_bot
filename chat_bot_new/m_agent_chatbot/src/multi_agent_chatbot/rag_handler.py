import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import time
import tempfile
import shutil
import logging
from datetime import datetime
import chromadb
import json
import uuid
import atexit
import hashlib
import base64
from io import BytesIO
import struct
from PIL import Image, ImageFile, ImageEnhance, ImageOps, ImageFilter
import warnings
from pdfminer.pdfpage import PDFPage
from pdfminer.pdftypes import PDFObjRef
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader, PDFPlumberLoader
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from PIL import Image
import pytesseract
import io
import pdfplumber

# PyMuPDF import with fallback
try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None
        logger.warning("PyMuPDF (fitz) is not available. Some PDF processing features will be limited.")

from .llm_config import embeddings, llm_general, llm_coding
from .utils import extract_javascript_from_text, convert_js_to_python_code

# 데이터 디렉토리 설정
BASE_DIR = Path(__file__).parent.parent.parent.parent / "m_agent_chatbot"
DATA_DIR = BASE_DIR / "data"
BASE_CHROMA_DB_PATH = str(DATA_DIR / "chroma_db")
PDF_STORAGE_PATH = str(DATA_DIR / "pdfs")
PDF_METADATA_PATH = str(DATA_DIR / "pdf_metadata.json")
PDF_INDEX_PATH = str(DATA_DIR / "pdf_index.json")

# 디렉토리 생성
for path in [DATA_DIR, Path(BASE_CHROMA_DB_PATH), Path(PDF_STORAGE_PATH)]:
    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created directory: {path}")

# 메타데이터 파일 초기화
for path in [PDF_METADATA_PATH, PDF_INDEX_PATH]:
    if not os.path.exists(path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({}, f)
        logger.info(f"Created empty metadata file: {path}")

# ChromaDB 초기화
vectorstore = None

# PDF 메타데이터 관리
pdf_metadata = {}
pdf_index = {}  # PDF 파일 경로와 ID 매핑

# 정리할 데이터베이스 목록
databases_to_cleanup = set()

# 텍스트 분할기 초기화
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

# 전역 설정
MAX_PROCESSING_TIME = 6000  # 최대 처리 시간을 100분으로 증가
PROCESSING_CHECK_INTERVAL = 5  # 처리 상태 확인 간격 (초)
BATCH_SIZE = 50  # 한 번에 처리할 청크 수100>>50

class ProcessingTimeout(Exception):
    """PDF 처리 시간 초과 예외"""
    pass

def save_pdf_metadata():
    """PDF 메타데이터를 파일에 저장합니다."""
    with open(PDF_METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(pdf_metadata, f, ensure_ascii=False, indent=2)

def load_pdf_metadata():
    """PDF 메타데이터를 파일에서 로드합니다."""
    global pdf_metadata
    try:
        if os.path.exists(PDF_METADATA_PATH):
            with open(PDF_METADATA_PATH, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:  # 파일이 비어있지 않은 경우에만 JSON 파싱 시도
                    pdf_metadata = json.loads(content)
                else:
                    pdf_metadata = {}
        else:
            pdf_metadata = {}
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON in {PDF_METADATA_PATH}. Creating new metadata file.")
        pdf_metadata = {}
        save_pdf_metadata()  # 새로운 메타데이터 파일 생성
    except Exception as e:
        print(f"Error loading PDF metadata: {str(e)}")
        pdf_metadata = {}

def save_pdf_index():
    """PDF 인덱스를 파일에 저장합니다."""
    with open(PDF_INDEX_PATH, 'w', encoding='utf-8') as f:
        json.dump(pdf_index, f, ensure_ascii=False, indent=2)

def load_pdf_index():
    """PDF 인덱스를 파일에서 로드합니다."""
    global pdf_index
    try:
        if os.path.exists(PDF_INDEX_PATH):
            with open(PDF_INDEX_PATH, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:  # 파일이 비어있지 않은 경우에만 JSON 파싱 시도
                    pdf_index = json.loads(content)
                else:
                    pdf_index = {}
        else:
            pdf_index = {}
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON in {PDF_INDEX_PATH}. Creating new index file.")
        pdf_index = {}
        save_pdf_index()  # 새로운 인덱스 파일 생성
    except Exception as e:
        print(f"Error loading PDF index: {str(e)}")
        pdf_index = {}

# 메타데이터와 인덱스 로드
load_pdf_metadata()
load_pdf_index()

def get_new_db_path():
    """Generate a new unique database path."""
    return f"{BASE_CHROMA_DB_PATH}_{uuid.uuid4().hex[:8]}"

def cleanup_database(db_path):
    """Clean up a single database directory."""
    if not os.path.exists(db_path):
        return
    
    try:
        # 먼저 SQLite 파일을 삭제
        sqlite_file = os.path.join(db_path, "chroma.sqlite3")
        if os.path.exists(sqlite_file):
            try:
                os.remove(sqlite_file)
            except Exception as e:
                print(f"Warning: Could not remove SQLite file {sqlite_file}: {e}")
                return False
        
        # 나머지 파일들 삭제
        for root, dirs, files in os.walk(db_path, topdown=False):
            for name in files:
                try:
                    os.remove(os.path.join(root, name))
                except Exception as e:
                    print(f"Warning: Could not remove file {name}: {e}")
            for name in dirs:
                try:
                    os.rmdir(os.path.join(root, name))
                except Exception as e:
                    print(f"Warning: Could not remove directory {name}: {e}")
        
        # 마지막으로 디렉토리 삭제
        try:
            os.rmdir(db_path)
            return True
        except Exception as e:
            print(f"Warning: Could not remove directory {db_path}: {e}")
            return False
    except Exception as e:
        print(f"Error cleaning up database {db_path}: {e}")
        return False

def cleanup_old_databases():
    """Clean up old database directories."""
    global databases_to_cleanup
    
    for db_path in list(databases_to_cleanup):
        if cleanup_database(db_path):
            databases_to_cleanup.remove(db_path)

def register_cleanup(db_path):
    """Register a database for cleanup."""
    global databases_to_cleanup
    databases_to_cleanup.add(db_path)

# 프로그램 종료 시 정리 작업 등록
atexit.register(cleanup_old_databases)

def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def is_duplicate_file(file_path: str) -> Optional[str]:
    """Check if file is a duplicate based on content hash."""
    file_hash = calculate_file_hash(file_path)
    for pdf_id, info in pdf_index.items():
        if os.path.exists(info["path"]):
            existing_hash = calculate_file_hash(info["path"])
            if existing_hash == file_hash:
                return pdf_id
    return None

def update_pdf_status(pdf_id: str, status: str, error_message: Optional[str] = None):
    """Update PDF processing status and save metadata."""
    if pdf_id in pdf_metadata:
        pdf_metadata[pdf_id]["status"] = status
        if error_message:
            pdf_metadata[pdf_id]["error"] = error_message
        pdf_metadata[pdf_id]["last_updated"] = datetime.now().isoformat()
        save_pdf_metadata()

def retry_failed_pdfs():
    """Retry processing of failed PDFs."""
    # 현재 실패한 PDF 목록을 복사하여 사용
    failed_pdfs = [(pdf_id, info) for pdf_id, info in pdf_metadata.items() 
                  if info["status"] == "failed" and pdf_id in pdf_index]
    
    for pdf_id, info in failed_pdfs:
            pdf_path = pdf_index[pdf_id]["path"]
            if os.path.exists(pdf_path):
                print(f"Retrying failed PDF: {pdf_path}")
            try:
                # 원본 파일명 추출 (UUID 제거)
                original_filename = os.path.basename(pdf_path).split("_", 1)[1] if "_" in os.path.basename(pdf_path) else os.path.basename(pdf_path)
                
                # 파일이 이미 성공적으로 처리되었는지 확인
                file_hash = calculate_file_hash(pdf_path)
                duplicate_exists = False
                for other_id, other_info in pdf_metadata.items():
                    if (other_id != pdf_id and 
                        other_info["status"] == "processed" and 
                        other_id in pdf_index and 
                        os.path.exists(pdf_index[other_id]["path"]) and
                        calculate_file_hash(pdf_index[other_id]["path"]) == file_hash):
                        duplicate_exists = True
                        # 중복 파일이 이미 처리되었으므로 현재 실패한 파일 정리
                        cleanup_failed_files(original_filename)
                        break
                
                if not duplicate_exists:
                    update_pdf_status(pdf_id, "processing")
                    if process_and_embed_pdf(pdf_path, original_filename=original_filename):
                        update_pdf_status(pdf_id, "processed")
                    else:
                        update_pdf_status(pdf_id, "failed", "Retry failed")
            except Exception as e:
                update_pdf_status(pdf_id, "failed", str(e))

def get_image_info(data):
    """이미지 데이터의 형식과 크기를 감지합니다."""
    if isinstance(data, (bytes, bytearray)):
        # JPEG
        if data.startswith(b'\xFF\xD8'):
            return 'JPEG'
        # PNG
        elif data.startswith(b'\x89PNG\r\n\x1a\n'):
            return 'PNG'
        # GIF
        elif data.startswith(b'GIF87a') or data.startswith(b'GIF89a'):
            return 'GIF'
        # TIFF
        elif data.startswith(b'II*\x00') or data.startswith(b'MM\x00*'):
            return 'TIFF'
        # SGI
        elif data.startswith(b'\x01\xDA'):
            return 'SGI'
    return None

def decode_sgi_image(data):
    """SGI 이미지 포맷을 디코딩합니다."""
    try:
        # SGI 이미지 헤더 파싱
        magic, storage, bpc, dimension, xsize, ysize, zsize = struct.unpack('>h4B3h', data[:12])
        if magic != 474:  # SGI 매직 넘버
            return None
            
        # Raw 데이터를 RGB 형식으로 변환
        if bpc == 1 and zsize in [1, 3, 4]:
            if zsize == 1:  # Grayscale
                mode = 'L'
            elif zsize == 3:  # RGB
                mode = 'RGB'
            else:  # RGBA
                mode = 'RGBA'
                
            try:
                # 이미지 데이터 추출 (헤더 이후)
                image_data = data[512:]  # SGI 헤더 크기는 512 bytes
                return Image.frombytes(mode, (xsize, ysize), image_data)
            except:
                return None
    except:
        return None
    return None

def decode_image_stream(stream_data):
    """PDF 스트림에서 이미지 데이터를 디코딩합니다."""
    try:
        # PDFMiner의 PDFStream 객체 처리
        if hasattr(stream_data, '__class__') and stream_data.__class__.__name__ == 'PDFStream':
            try:
                # PDFStream에서 raw 데이터 추출
                stream_data = stream_data.get_data()
            except:
                try:
                    # 압축된 데이터일 경우 압축 해제 시도
                    stream_data = stream_data.get_raw_data()
                except:
                    return None

        # 이미지 데이터가 base64로 인코딩되어 있을 수 있음
        try:
            decoded = base64.b64decode(stream_data)
            return decoded
        except:
            pass
        
        # 직접 바이트 데이터일 수 있음
        if isinstance(stream_data, (bytes, bytearray)):
            return stream_data
            
        # stream 객체일 수 있음
        if hasattr(stream_data, 'get_data'):
            return stream_data.get_data()
            
        return stream_data
    except Exception as e:
        logger.warning(f"Failed to decode image stream: {str(e)}")
        return None

class ImageProcessor:
    def __init__(self):
        self.ocr_config = r'--oem 1 --psm 6 -l kor+eng'
    
    def process_image(self, image_data):
        """이미지 데이터를 처리하고 텍스트를 추출합니다."""
        try:
            # PIL Image로 변환
            if not isinstance(image_data, Image.Image):
                try:
                    ImageFile.LOAD_TRUNCATED_IMAGES = True
                    decoded_data = decode_image_stream(image_data)
                    if decoded_data is None:
                        return ""
                    
                    image_format = get_image_info(decoded_data)
                    image = None
                    
                    if image_format:
                        try:
                            if image_format == 'SGI':
                                image = decode_sgi_image(decoded_data)
                            else:
                                image = Image.open(BytesIO(decoded_data))
                        except Exception as e:
                            logger.warning(f"Failed to decode {image_format} image: {str(e)}")
                    
                    if image is None:
                        return ""
                    
                    if image.mode not in ['RGB', 'L']:
                        image = image.convert('RGB')
                except Exception as e:
                    logger.warning(f"Failed to process image data: {str(e)}")
                    return ""
            else:
                image = image_data

            # 이미지 전처리
            try:
                max_size = 2000
                if max(image.size) > max_size:
                    ratio = max_size / max(image.size)
                    new_size = tuple(int(dim * ratio) for dim in image.size)
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                
                image = image.convert('L')
                image = ImageOps.autocontrast(image)
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1.5)
                image = image.filter(ImageFilter.MedianFilter(size=3))
            except Exception as e:
                logger.warning(f"Image preprocessing failed: {str(e)}")
            
            # OCR 수행
            try:
                tesseract_cmd = pytesseract.get_tesseract_version()
            except Exception as e:
                logger.error("Tesseract is not properly installed.")
                return ""
            
            text = pytesseract.image_to_string(image, config=self.ocr_config)
            return text.strip()
        except Exception as e:
            logger.error(f"이미지 텍스트 추출 중 오류 발생: {str(e)}")
            return ""

# 이미지 프로세서 인스턴스 생성
image_processor = ImageProcessor()

# extract_text_from_image 함수를 image_processor.process_image로 대체
def extract_text_from_image(image_data):
    """이미지에서 텍스트를 추출합니다. (ImageProcessor의 process_image 메서드를 사용)"""
    return image_processor.process_image(image_data)

def ensure_directory_exists(directory_path: str):
    """디렉토리가 존재하지 않으면 생성합니다."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def copy_to_permanent_storage(temp_file_path: str, filename: str) -> str:
    """임시 파일을 영구 저장소로 복사합니다."""
    ensure_directory_exists(PDF_STORAGE_PATH)
    
    # 이미 UUID가 포함된 파일명인지 확인
    if any(c for c in filename if not c.isalnum() and c not in '.-_ '):
        # UUID가 이미 포함된 것으로 보이는 경우, 원본 파일명만 추출
        try:
            original_filename = filename.split("_", 1)[1] if "_" in filename else filename
        except:
            original_filename = filename
    else:
        original_filename = filename
    
    # 고유한 파일명 생성
    file_id = str(uuid.uuid4())
    permanent_path = os.path.join(PDF_STORAGE_PATH, f"{file_id}_{original_filename}")
    
    # 파일 복사
    shutil.copy2(temp_file_path, permanent_path)
    return permanent_path

def process_with_timeout(func, *args, timeout=MAX_PROCESSING_TIME):
    """함수 실행을 타임아웃과 함께 처리합니다."""
    result = None
    error = None
    
    def worker():
        nonlocal result, error
        try:
            result = func(*args)
        except Exception as e:
            error = e
    
    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        raise ProcessingTimeout(f"Processing timed out after {timeout} seconds")
    
    if error:
        raise error
    
    return result

class PDFProcessor:
    def __init__(self):
        # 로더 순서 최적화: 가장 안정적인 로더를 먼저 시도
        self.loaders = [
            (PDFPlumberLoader, self._process_with_pdfplumber),  # 커스텀 처리
            (PyMuPDFLoader, self._process_with_pymupdf) if fitz else None,  # PyMuPDF 사용 가능한 경우
            (UnstructuredPDFLoader, None),  # 기본 처리
            (PyPDFLoader, None)  # 기본 처리
        ]
        self.loaders = [loader for loader in self.loaders if loader is not None]
        
        self.image_processor = ImageProcessor()
        
        # 텍스트 분할 설정 최적화
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # 청크 크기 감소
            chunk_overlap=100,  # 중복 감소
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    def _try_loader(self, loader_class, file_path: str) -> List[Document]:
        """일반적인 PDF 로더를 사용하여 문서를 추출합니다."""
        try:
            docs = loader_class(file_path).load()
            
            # 추출된 문서 검증
            valid_docs = []
            for doc in docs:
                if doc.page_content and doc.page_content.strip():
                    valid_docs.append(doc)
                else:
                    logger.warning(f"Empty document content from {loader_class.__name__}")
            
            if not valid_docs:
                logger.warning(f"{loader_class.__name__} extracted {len(docs)} documents but all were empty")
                return []
            
            return valid_docs
        except Exception as e:
            logger.error(f"Error in {loader_class.__name__}: {str(e)}")
            return []

    def _process_with_pdfplumber(self, file_path: str) -> List[Document]:
        """PDFPlumber를 사용하여 PDF를 처리합니다."""
        docs = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    try:
                        # 텍스트 추출 시도
                        text = page.extract_text() or ""
                        
                        # 이미지에서 텍스트 추출
                        image_texts = []
                        if hasattr(page, 'images'):
                            for image in page.images:
                                try:
                                    image_bytes = None
                                    for key in ['stream', 'bytes', 'data']:
                                        if key in image:
                                            image_bytes = image[key]
                                            break
                                    
                                    if image_bytes:
                                        logger.info(f"Processing image on page {i + 1}")
                                        ocr_text = self.image_processor.process_image(image_bytes)
                                        if ocr_text:
                                            image_texts.append(ocr_text)
                                except Exception as e:
                                    logger.warning(f"Failed to process image on page {i + 1}: {str(e)}")
                        
                        # 텍스트와 이미지 텍스트 결합
                        combined_text = text
                        if image_texts:
                            combined_text += "\n[Image Text]\n" + "\n".join(image_texts)
                        
                        if combined_text.strip():
                            docs.append(Document(
                                page_content=combined_text,
                                metadata={"page": i + 1, "source": file_path}
                            ))
                        else:
                            logger.warning(f"No text or image text extracted from page {i + 1}")
                    except Exception as e:
                        logger.error(f"Error processing page {i + 1}: {str(e)}")
        except Exception as e:
            logger.error(f"Error opening PDF with PDFPlumber: {str(e)}")
        return docs

    def _process_with_pymupdf(self, file_path: str) -> List[Document]:
        """PyMuPDF를 사용하여 PDF를 처리합니다."""
        docs = []
        try:
            doc = fitz.open(file_path)
            for i, page in enumerate(doc):
                try:
                    # 텍스트 추출
                    text = page.get_text() or ""
                    
                    # 이미지에서 텍스트 추출
                    image_texts = []
                    for img_index, img in enumerate(page.get_images()):
                        try:
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            if base_image and "image" in base_image:
                                logger.info(f"Processing image {img_index + 1} on page {i + 1}")
                                ocr_text = self.image_processor.process_image(base_image["image"])
                                if ocr_text:
                                    image_texts.append(ocr_text)
                        except Exception as e:
                            logger.warning(f"Failed to process image {img_index + 1} on page {i + 1}: {str(e)}")
                    
                    # 텍스트와 이미지 텍스트 결합
                    combined_text = text
                    if image_texts:
                        combined_text += "\n[Image Text]\n" + "\n".join(image_texts)
                    
                    if combined_text.strip():
                        docs.append(Document(
                            page_content=combined_text,
                            metadata={"page": i + 1, "source": file_path}
                        ))
                    else:
                        logger.warning(f"No text or image text extracted from page {i + 1}")
                except Exception as e:
                    logger.error(f"Error processing page {i + 1}: {str(e)}")
            doc.close()
        except Exception as e:
            logger.error(f"Error opening PDF with PyMuPDF: {str(e)}")
        return docs

    def _is_duplicate_file(self, file_path: str, original_filename: str) -> bool:
        """중복 파일인지 확인합니다."""
        file_hash = calculate_file_hash(file_path)
        for existing_id, info in pdf_index.items():
            if os.path.exists(info["path"]):
                if calculate_file_hash(info["path"]) == file_hash:
                    logger.info(f"Duplicate file detected. Reusing existing data for {original_filename}")
                    if existing_id in pdf_metadata and pdf_metadata[existing_id]["status"] == "failed":
                        pdf_metadata[existing_id]["status"] = "processed"
                        pdf_metadata[existing_id]["last_updated"] = datetime.now().isoformat()
                        save_pdf_metadata()
                    return True
        return False

    def _update_pdf_metadata(self, pdf_id: str, original_filename: str, permanent_file_path: str):
        """PDF 메타데이터를 업데이트합니다."""
        pdf_index[pdf_id] = {
            "id": pdf_id,
            "filename": original_filename,
            "path": permanent_file_path
        }
        save_pdf_index()
        
        pdf_metadata[pdf_id] = {
            "filename": original_filename,
            "upload_time": datetime.now().isoformat(),
            "status": "processing",
            "progress": 0
        }
        save_pdf_metadata()

    def _update_progress(self, pdf_id: str, progress: int):
        """처리 진행률을 업데이트합니다."""
        pdf_metadata[pdf_id]["progress"] = progress
        pdf_metadata[pdf_id]["last_updated"] = datetime.now().isoformat()
        save_pdf_metadata()

    def process_pdf(self, file_path: str, original_filename: str = None) -> bool:
        """PDF 파일을 처리하고 벡터 DB에 저장합니다."""
        try:
            logger.info(f"Starting PDF processing: {file_path}")
            
            if not original_filename:
                original_filename = os.path.basename(file_path)
            
            # 중복 파일 체크
            if self._is_duplicate_file(file_path, original_filename):
                return True
            
            # 영구 저장소로 파일 복사
            permanent_file_path = copy_to_permanent_storage(file_path, original_filename)
            logger.info(f"Copied to permanent storage: {permanent_file_path}")
            
            # 기존 실패한 파일 정리
            cleanup_failed_files(original_filename)
            
            # PDF ID 생성 및 메타데이터 업데이트
            pdf_id = str(uuid.uuid4())
            self._update_pdf_metadata(pdf_id, original_filename, permanent_file_path)
            
            def process_pdf_content():
                all_texts = []
                warnings.filterwarnings('ignore', category=UserWarning, module='pdfminer.pdfpage')
                
                self._update_progress(pdf_id, 10)
                
                # 각 로더로 시도
                for loader_class, custom_processor in self.loaders:
                    try:
                        logger.info(f"Trying loader: {loader_class.__name__}")
                        
                        if custom_processor:
                            docs = custom_processor(permanent_file_path)
                        else:
                            docs = self._try_loader(loader_class, permanent_file_path)
                        
                        if docs and len(docs) > 0:
                            logger.info(f"Successfully extracted {len(docs)} documents using {loader_class.__name__}")
                            
                            # 문서 유효성 검사
                            valid_docs = []
                            for doc in docs:
                                if len(doc.page_content.strip()) >= 50:  # 최소 길이 검사
                                    doc.metadata["loader"] = loader_class.__name__
                                    valid_docs.append(doc)
                                else:
                                    logger.warning(f"Skipping short document: {len(doc.page_content)} chars")
                            
                            if valid_docs:
                                all_texts.extend(valid_docs)
                                break  # 성공적으로 처리된 경우 다음 로더 시도하지 않음
                    except Exception as e:
                        logger.error(f"{loader_class.__name__} failed: {str(e)}")
                        continue
                
                if not all_texts:
                    raise Exception("No valid text could be extracted from the PDF using any loader")
                
                logger.info(f"Total valid documents extracted: {len(all_texts)}")
                self._update_progress(pdf_id, 70)
                
                # 텍스트 분할
                split_docs = []
                for doc in all_texts:
                    chunks = self.text_splitter.split_documents([doc])
                    if chunks:
                        for chunk in chunks:
                            chunk.metadata.update(doc.metadata)
                            # 청크 품질 검사
                            if len(chunk.page_content.strip()) >= 50:
                                split_docs.append(chunk)
                
                if not split_docs:
                    raise Exception("No valid text chunks generated after splitting")
                
                logger.info(f"Generated {len(split_docs)} valid chunks")
                
                # 청크를 배치로 나누어 처리
                total_chunks = len(split_docs)
                total_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE
                
                processed_chunks = 0
                for i in range(0, total_chunks, BATCH_SIZE):
                    batch = split_docs[i:i + BATCH_SIZE]
                    current_batch = i // BATCH_SIZE + 1
                    
                    try:
                        # RAGManager를 통해 문서 추가
                        rag_manager.add_documents(batch, {"pdf_id": pdf_id})
                        processed_chunks += len(batch)
                        progress = min(100, 70 + (processed_chunks * 30 // total_chunks))
                        self._update_progress(pdf_id, progress)
                        logger.info(f"Processed batch {current_batch}/{total_batches}")
                    except Exception as e:
                        logger.error(f"Error processing batch {current_batch}: {str(e)}")
                        raise
                
                logger.info("PDF processing completed successfully")
                return True
            
            # 타임아웃과 함께 처리
            process_with_timeout(process_pdf_content, timeout=MAX_PROCESSING_TIME)
            
            # 성공적으로 처리됨
            update_pdf_status(pdf_id, "processed")
            return True
            
        except ProcessingTimeout:
            error_msg = f"PDF processing timed out after {MAX_PROCESSING_TIME} seconds"
            logger.error(error_msg)
            update_pdf_status(pdf_id, "failed", error_msg)
            return False
        except Exception as e:
            error_msg = f"PDF processing failed: {str(e)}"
            logger.error(error_msg)
            update_pdf_status(pdf_id, "failed", error_msg)
            return False

# PDF 프로세서 인스턴스 생성
pdf_processor = PDFProcessor()

# process_and_embed_pdf 함수를 pdf_processor.process_pdf로 대체
def process_and_embed_pdf(temp_file_path: str, original_filename: str = None) -> bool:
    """PDF 파일을 처리하고 벡터 DB에 저장합니다. (PDFProcessor의 process_pdf 메서드를 사용)"""
    return pdf_processor.process_pdf(temp_file_path, original_filename)

def list_available_collections():
    """ 사용 가능한 ChromaDB 컬렉션 목록 (디버깅용) """
    client = Chroma(persist_directory=BASE_CHROMA_DB_PATH)
    chromadb_client = client._client
    collection = chromadb_client.list_collections()
    return [col.name for col in collection]

class RAGManager:
    def __init__(self):
        self.vectorstore = None
        self.collection_name = "rag_collection"
        self.pdf_metadata = {}
        self.pdf_index = {}
        self.initialize_vectorstore()
        
        # 메타데이터와 인덱스 로드
        self.load_metadata()
        self.load_index()
    
    def load_metadata(self):
        """PDF 메타데이터를 로드합니다."""
        try:
            if os.path.exists(PDF_METADATA_PATH):
                with open(PDF_METADATA_PATH, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        self.pdf_metadata = json.loads(content)
                    else:
                        self.pdf_metadata = {}
            logger.info(f"Loaded {len(self.pdf_metadata)} PDF metadata entries")
        except Exception as e:
            logger.error(f"Error loading PDF metadata: {e}")
            self.pdf_metadata = {}
    
    def load_index(self):
        """PDF 인덱스를 로드합니다."""
        try:
            if os.path.exists(PDF_INDEX_PATH):
                with open(PDF_INDEX_PATH, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        self.pdf_index = json.loads(content)
                    else:
                        self.pdf_index = {}
            logger.info(f"Loaded {len(self.pdf_index)} PDF index entries")
        except Exception as e:
            logger.error(f"Error loading PDF index: {e}")
            self.pdf_index = {}
    
    def initialize_vectorstore(self):
        """벡터 저장소를 초기화합니다."""
        try:
            if not self.vectorstore:
                self.vectorstore = Chroma(
                    persist_directory=BASE_CHROMA_DB_PATH,
                    embedding_function=embeddings,
                    collection_name=self.collection_name
                )
                count = self.vectorstore._collection.count()
                logger.info(f"ChromaDB initialized with {count} documents in collection: {self.collection_name}")
                
                if count > 0:
                    sample = self.vectorstore.similarity_search("", k=1)
                    logger.info(f"Sample document: {sample[0].page_content[:200]}...")
                else:
                    logger.warning("No documents found in the collection")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise
    
    def add_documents(self, documents, metadata=None):
        """문서를 벡터 저장소에 추가합니다."""
        try:
            if not self.vectorstore:
                self.initialize_vectorstore()
            
            # 기존 문서 수 확인
            before_count = self.vectorstore._collection.count()
            
            # 메타데이터 병합
            if metadata:
                for doc in documents:
                    doc.metadata.update(metadata)
            
            # 새 문서 추가
            self.vectorstore.add_documents(documents)
            self.vectorstore.persist()
            
            # 추가된 문서 수 확인
            after_count = self.vectorstore._collection.count()
            logger.info(f"Added {after_count - before_count} new documents")
            
            return True
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    def get_relevant_documents(self, query, k=5):
        """쿼리와 관련된 문서를 검색합니다."""
        try:
            if not self.vectorstore:
                self.initialize_vectorstore()
            
            total_docs = self.vectorstore._collection.count()
            logger.info(f"Searching in {total_docs} documents for query: {query}")
            
            if total_docs == 0:
                logger.warning("No documents in the collection")
                return []
            
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
            relevant_docs = retriever.invoke(query)
            
            logger.info(f"Found {len(relevant_docs)} relevant documents")
            for i, doc in enumerate(relevant_docs):
                logger.info(f"Document {i+1} preview: {doc.page_content[:200]}...")
            
            return relevant_docs
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

# RAGManager 인스턴스 생성
rag_manager = RAGManager()

def get_relevant_documents(query: str, k: int = 5) -> List[Document]:
    """쿼리와 관련된 문서를 검색합니다."""
    return rag_manager.get_relevant_documents(query, k)

def query_pdf_content(query: str, k: int = 5) -> str:
    """PDF 내용을 검색합니다."""
    try:
        logger.info(f"Querying PDF content for: {query}")
        docs = rag_manager.get_relevant_documents(query, k)
        
        if not docs:
            logger.warning("No relevant documents found")
            return "관련된 PDF 내용을 찾을 수 없습니다."
        
        content = "\n\n".join([doc.page_content for doc in docs])
        logger.info(f"Found content with {len(content)} characters")
        return content
    except Exception as e:
        logger.error(f"Error querying PDF content: {e}")
        return "PDF 내용을 검색하는 중 오류가 발생했습니다."

# 초기화 시 기존 DB 로드 확인
print(f"ChromaDB initialized. Available collections: {list_available_collections()}")
if not any(col == "rag_collection" for col in list_available_collections()):
    print("Warning: 'rag_collection' not found. PDFs might need to be re-uploaded.")

def get_processed_pdfs() -> List[Dict]:
    """
    처리된 PDF 파일 목록을 반환합니다.
    """
    return [
        {
            "filename": info["filename"],
            "id": info["id"],
            "status": pdf_metadata[info["id"]]["status"]
        }
        for info in pdf_index.values()
    ]

def get_stored_data_info():
    """저장된 데이터의 정보를 반환합니다."""
    try:
        # ChromaDB 컬렉션 정보
        collections = list_available_collections()
        collection_info = {}
        for collection in collections:
            client = Chroma(persist_directory=BASE_CHROMA_DB_PATH, collection_name=collection)
            count = client._collection.count()
            collection_info[collection] = count

        # PDF 파일 정보
        pdf_files = []
        for pdf_id, info in pdf_index.items():
            if os.path.exists(info["path"]):
                status = pdf_metadata.get(pdf_id, {}).get("status", "unknown")
                pdf_files.append({
                    "filename": info["filename"],
                    "status": status,
                    "path": info["path"]
                })

        return {
            "collections": collection_info,
            "pdf_files": pdf_files,
            "total_pdfs": len(pdf_files),
            "chroma_db_path": BASE_CHROMA_DB_PATH,
            "pdf_storage_path": PDF_STORAGE_PATH
        }
    except Exception as e:
        print(f"Error getting stored data info: {str(e)}")
        return None

def print_stored_data_info():
    """저장된 데이터의 정보를 출력합니다."""
    info = get_stored_data_info()
    if info:
        print("\n=== Stored Data Information ===")
        print("\nChromaDB Collections:")
        for collection, count in info["collections"].items():
            print(f"- {collection}: {count} documents")
        
        print("\nPDF Files:")
        for pdf in info["pdf_files"]:
            print(f"- {pdf['filename']} (Status: {pdf['status']})")
        
        print(f"\nTotal PDFs: {info['total_pdfs']}")
        print(f"\nChromaDB Path: {info['chroma_db_path']}")
        print(f"PDF Storage Path: {info['pdf_storage_path']}")
        print("=============================\n")
    else:
        print("Failed to get stored data information")

def cleanup_failed_files(original_filename: str):
    """실패한 상태의 동일 파일들을 정리합니다."""
    # 실패한 파일들의 ID 목록 수집
    failed_ids = []
    for pdf_id, info in pdf_metadata.items():
        if (info["status"] == "failed" and 
            info["filename"] == original_filename and 
            pdf_id in pdf_index):
            failed_ids.append(pdf_id)
    
    # 실패한 파일들 정리
    for failed_id in failed_ids:
        try:
            # 파일 삭제
            failed_path = pdf_index[failed_id]["path"]
            if os.path.exists(failed_path):
                os.remove(failed_path)
            
            # 메타데이터와 인덱스에서 제거
            del pdf_metadata[failed_id]
            del pdf_index[failed_id]
        except Exception as e:
            logger.warning(f"Failed to cleanup file {failed_id}: {str(e)}")
    
    # 변경사항 저장
    save_pdf_metadata()
    save_pdf_index()

# 주기적으로 실패한 PDF 재처리 시도
def start_pdf_retry_scheduler():
    """Start a background thread to periodically retry failed PDFs."""
    import threading
    import time

    def retry_worker():
        while True:
            retry_failed_pdfs()
            time.sleep(300)  # 5분마다 재시도

    thread = threading.Thread(target=retry_worker, daemon=True)
    thread.start()

# 초기화 시 실패한 PDF 재처리 시작
start_pdf_retry_scheduler()

def cleanup_all_data():
    """모든 PDF 데이터와 벡터 저장소를 정리합니다."""
    try:
        # ChromaDB 컬렉션 삭제
        client = Chroma(persist_directory=BASE_CHROMA_DB_PATH)
        collections = client._client.list_collections()
        for collection in collections:
            try:
                client._client.delete_collection(collection.name)
                logger.info(f"Deleted collection: {collection.name}")
            except Exception as e:
                logger.error(f"Error deleting collection {collection.name}: {e}")

        # PDF 파일 삭제
        if os.path.exists(PDF_STORAGE_PATH):
            for filename in os.listdir(PDF_STORAGE_PATH):
                try:
                    file_path = os.path.join(PDF_STORAGE_PATH, filename)
                    os.remove(file_path)
                    logger.info(f"Deleted PDF file: {filename}")
                except Exception as e:
                    logger.error(f"Error deleting file {filename}: {e}")

        # 메타데이터 파일 초기화
        for path in [PDF_METADATA_PATH, PDF_INDEX_PATH]:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump({}, f)
                logger.info(f"Reset metadata file: {path}")
            except Exception as e:
                logger.error(f"Error resetting {path}: {e}")

        # 전역 변수 초기화
        global pdf_metadata, pdf_index
        pdf_metadata = {}
        pdf_index = {}

        logger.info("All data has been cleaned up successfully")
        return True
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return False 
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
from pypdf import PdfReader  # PyPDF2 대신 pypdf 사용

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
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
BASE_CHROMA_DB_PATH = str(DATA_DIR / "chroma_db")
PDF_STORAGE_PATH = str(DATA_DIR / "pdfs")
PDF_METADATA_PATH = str(DATA_DIR / "pdf_metadata.json")
PDF_INDEX_PATH = str(DATA_DIR / "pdf_index.json")

def list_available_collections():
    """사용 가능한 ChromaDB 컬렉션 목록을 반환합니다."""
    try:
        client = Chroma(persist_directory=BASE_CHROMA_DB_PATH)
        return [col.name for col in client._client.list_collections()]
    except Exception as e:
        logger.error(f"Failed to list collections: {str(e)}")
        return []

# 디렉토리 생성
for path in [DATA_DIR, Path(BASE_CHROMA_DB_PATH), Path(PDF_STORAGE_PATH)]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {str(e)}")
        raise

# 메타데이터 파일 초기화
for path in [PDF_METADATA_PATH, PDF_INDEX_PATH]:
    try:
        if not os.path.exists(path):
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({}, f)
            logger.info(f"Created empty metadata file: {path}")
    except Exception as e:
        logger.error(f"Failed to initialize metadata file {path}: {str(e)}")
        raise

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
MAX_PROCESSING_TIME = 6000  # 최대 처리 시간 (초)
PROCESSING_CHECK_INTERVAL = 5  # 처리 상태 확인 간격 (초)
BATCH_SIZE = 50  # 한 번에 처리할 청크 수

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
            (PDFPlumberLoader, None),  # 기본 처리
            (PyMuPDFLoader, None) if fitz else None,  # PyMuPDF 사용 가능한 경우
            (UnstructuredPDFLoader, None),  # 기본 처리
            (PyPDFLoader, None)  # 기본 처리
        ]
        self.loaders = [loader for loader in self.loaders if loader is not None]
        
        # 텍스트 분할 설정
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

    def process_pdf(self, file_path: str, original_filename: str = None) -> bool:
        """PDF 파일을 처리하고 벡터 저장소에 추가합니다."""
        try:
            if not os.path.exists(file_path):
                logger.error(f"PDF file not found: {file_path}")
                return False

            # 파일 유효성 검사
            is_valid, error_message = validate_pdf(file_path)
            if not is_valid:
                logger.error(f"Invalid PDF file: {error_message}")
                return False

            # 중복 파일 검사
            if self._is_duplicate_file(file_path, original_filename):
                logger.info(f"Duplicate file detected: {original_filename}")
                return True

            # PDF ID 생성
            pdf_id = str(uuid.uuid4())
            
            # 메타데이터 업데이트
            self._update_pdf_metadata(pdf_id, original_filename, file_path)
            
            # 진행 상황 업데이트
            self._update_progress(pdf_id, 0)

            def process_pdf_content():
                try:
                    # PDF 처리 시도
                    documents = []
                    for loader_class, _ in self.loaders:
                        try:
                            loader = loader_class(file_path)
                            docs = loader.load()
                            if docs and len(docs) > 0:
                                documents = docs
                                break
                        except Exception as e:
                            logger.warning(f"Loader {loader_class.__name__} failed: {str(e)}")
                            continue

                    if not documents:
                        raise Exception("All PDF loaders failed")

                    # 문서 분할
                    split_docs = self.text_splitter.split_documents(documents)
                    
                    # 벡터 저장소에 추가
                    if vectorstore is None:
                        raise Exception("Vector store not initialized")
                    
                    # 배치 처리
                    for i in range(0, len(split_docs), BATCH_SIZE):
                        batch = split_docs[i:i + BATCH_SIZE]
                        vectorstore.add_documents(batch)
                        progress = min(100, (i + len(batch)) * 100 // len(split_docs))
                        self._update_progress(pdf_id, progress)
                    
                    update_pdf_status(pdf_id, "completed")
                    return True
                except Exception as e:
                    error_message = f"PDF processing failed: {str(e)}"
                    logger.error(error_message)
                    update_pdf_status(pdf_id, "failed", error_message)
                    return False

            # 타임아웃 처리
            return process_with_timeout(process_pdf_content, timeout=MAX_PROCESSING_TIME)
        except Exception as e:
            logger.error(f"Unexpected error in process_pdf: {str(e)}")
            return False

    def _is_duplicate_file(self, file_path: str, original_filename: str) -> bool:
        """중복 파일인지 확인합니다."""
        file_hash = calculate_file_hash(file_path)
        for existing_id, info in pdf_index.items():
            if os.path.exists(info["path"]):
                if calculate_file_hash(info["path"]) == file_hash:
                    logger.info(f"Duplicate file detected: {original_filename}")
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
        if pdf_id in pdf_metadata:
            pdf_metadata[pdf_id]["progress"] = progress
            pdf_metadata[pdf_id]["last_updated"] = datetime.now().isoformat()
            save_pdf_metadata()

# PDF 프로세서 인스턴스 생성
pdf_processor = PDFProcessor()

def process_and_embed_pdf(temp_file_path: str, original_filename: str = None) -> bool:
    """PDF 파일을 처리하고 벡터 저장소에 추가합니다."""
    return pdf_processor.process_pdf(temp_file_path, original_filename)

class RAGManager:
    def __init__(self):
        self.vectorstore = None
        self.embeddings = embeddings
        self.initialize_vectorstore()

    def initialize_vectorstore(self):
        """벡터 저장소를 초기화합니다."""
        try:
            if not os.path.exists(BASE_CHROMA_DB_PATH):
                os.makedirs(BASE_CHROMA_DB_PATH, exist_ok=True)
            
            self.vectorstore = Chroma(
                persist_directory=BASE_CHROMA_DB_PATH,
                embedding_function=self.embeddings,
                collection_name="rag_collection"
            )
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise

    def add_documents(self, documents, metadata=None):
        """문서를 벡터 저장소에 추가합니다."""
        try:
            if not self.vectorstore:
                self.initialize_vectorstore()
            
            if metadata:
                for doc in documents:
                    doc.metadata.update(metadata)
            
            # 배치 처리
            for i in range(0, len(documents), BATCH_SIZE):
                batch = documents[i:i + BATCH_SIZE]
                self.vectorstore.add_documents(batch)
            
            logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise

    def get_relevant_documents(self, query, k=5):
        """쿼리와 관련된 문서를 검색합니다."""
        try:
            if not self.vectorstore:
                self.initialize_vectorstore()
            
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {str(e)}")
            raise

# 전역 RAGManager 인스턴스 생성
rag_manager = RAGManager()

def get_relevant_documents(query: str, k: int = 5) -> List[Document]:
    """쿼리와 관련된 문서를 검색합니다."""
    return rag_manager.get_relevant_documents(query, k)

def query_pdf_content(query: str, k: int = 5) -> str:
    """PDF 내용을 쿼리하고 결과를 반환합니다."""
    try:
        documents = get_relevant_documents(query, k)
        if not documents:
            return "관련 문서를 찾을 수 없습니다."
        
        # 문서 내용 결합
        content = "\n\n".join(doc.page_content for doc in documents)
        return content
    except Exception as e:
        logger.error(f"Failed to query PDF content: {str(e)}")
        return f"문서 검색 중 오류가 발생했습니다: {str(e)}"

def get_processed_pdfs() -> List[Dict]:
    """처리된 PDF 파일 목록을 반환합니다."""
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
        collections = list_available_collections()
        collection_info = {}
        for collection in collections:
            client = Chroma(persist_directory=BASE_CHROMA_DB_PATH, collection_name=collection)
            count = client._collection.count()
            collection_info[collection] = count

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
        logger.error(f"Error getting stored data info: {str(e)}")
        return None

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

# 초기화 시 기존 DB 로드 확인
collections = list_available_collections()
if not any(col == "rag_collection" for col in collections):
    logger.warning("'rag_collection' not found. PDFs might need to be re-uploaded.")

def validate_pdf(file_path: str) -> Tuple[bool, str]:
    """
    PDF 파일의 유효성을 검사합니다.
    
    Args:
        file_path (str): 검사할 PDF 파일의 경로
        
    Returns:
        Tuple[bool, str]: (유효성 여부, 오류 메시지)
    """
    try:
        # 파일 존재 여부 확인
        if not os.path.exists(file_path):
            return False, "파일이 존재하지 않습니다."
            
        # 파일 크기 확인 (최대 100MB)
        file_size = os.path.getsize(file_path)
        if file_size > 100 * 1024 * 1024:  # 100MB
            return False, "파일 크기가 너무 큽니다. (최대 100MB)"
            
        # PDF 파일 형식 확인
        try:
            with open(file_path, 'rb') as f:
                header = f.read(5)
                if not header.startswith(b'%PDF-'):
                    return False, "유효하지 않은 PDF 파일입니다."
        except Exception as e:
            return False, f"PDF 파일 읽기 오류: {str(e)}"
            
        # PDF 파일 열기 테스트
        try:
            with open(file_path, 'rb') as f:
                PdfReader(f)
        except Exception as e:
            return False, f"PDF 파일 형식 오류: {str(e)}"
            
        return True, ""
        
    except Exception as e:
        return False, f"PDF 검증 중 오류 발생: {str(e)}" 

def verify_data_persistence() -> bool:
    """데이터 지속성을 확인합니다."""
    try:
        # ChromaDB 디렉토리 확인
        if not os.path.exists(BASE_CHROMA_DB_PATH):
            return False
        
        # PDF 저장소 확인
        if not os.path.exists(PDF_STORAGE_PATH):
            return False
        
        # 메타데이터 파일 확인
        if not os.path.exists(PDF_METADATA_PATH) or not os.path.exists(PDF_INDEX_PATH):
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error verifying data persistence: {str(e)}")
        return False

def get_database_status() -> Dict:
    """데이터베이스 상태를 반환합니다."""
    try:
        status = {
            "chroma_db_exists": os.path.exists(BASE_CHROMA_DB_PATH),
            "pdf_storage_exists": os.path.exists(PDF_STORAGE_PATH),
            "metadata_exists": os.path.exists(PDF_METADATA_PATH),
            "index_exists": os.path.exists(PDF_INDEX_PATH),
            "total_pdfs": len(pdf_metadata),
            "collections": list_available_collections()
        }
        return status
    except Exception as e:
        logger.error(f"Error getting database status: {str(e)}")
        return {}

def initialize_data() -> bool:
    """데이터 초기화를 수행합니다."""
    try:
        # 디렉토리 생성
        for path in [DATA_DIR, Path(BASE_CHROMA_DB_PATH), Path(PDF_STORAGE_PATH)]:
            path.mkdir(parents=True, exist_ok=True)
        
        # 메타데이터 파일 초기화
        for path in [PDF_METADATA_PATH, PDF_INDEX_PATH]:
            if not os.path.exists(path):
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump({}, f)
        
        # 메타데이터와 인덱스 로드
        load_pdf_metadata()
        load_pdf_index()
        
        return True
    except Exception as e:
        logger.error(f"Error initializing data: {str(e)}")
        return False

def get_initialized_vectorstore() -> Optional[Chroma]:
    """초기화된 벡터 저장소를 반환합니다."""
    global vectorstore
    try:
        if vectorstore is None:
            vectorstore = Chroma(
                persist_directory=BASE_CHROMA_DB_PATH,
                embedding_function=embeddings
            )
        return vectorstore
    except Exception as e:
        logger.error(f"Error getting initialized vectorstore: {str(e)}")
        return None

def cleanup_failed_files(original_filename: str):
    """실패한 상태의 동일 파일들을 정리합니다."""
    try:
        # 실패한 파일 찾기
        failed_files = []
        for pdf_id, info in pdf_metadata.items():
            if (info.get("status") == "failed" and 
                info.get("filename") == original_filename and 
                pdf_id in pdf_index):
                failed_files.append((pdf_id, info))
        
        # 실패한 파일 정리
        for pdf_id, info in failed_files:
            try:
                # 파일 삭제
                if os.path.exists(pdf_index[pdf_id]["path"]):
                    os.remove(pdf_index[pdf_id]["path"])
                
                # 메타데이터에서 제거
                if pdf_id in pdf_metadata:
                    del pdf_metadata[pdf_id]
                if pdf_id in pdf_index:
                    del pdf_index[pdf_id]
                
                logger.info(f"Cleaned up failed file: {original_filename} (ID: {pdf_id})")
            except Exception as e:
                logger.error(f"Error cleaning up file {pdf_id}: {str(e)}")
        
        # 메타데이터 저장
        save_pdf_metadata()
        save_pdf_index()
        
        return len(failed_files)
    except Exception as e:
        logger.error(f"Error in cleanup_failed_files: {str(e)}")
        return 0 

def initialize_rag_system():
    """RAG 시스템을 초기화합니다."""
    global vectorstore
    try:
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

        # 메타데이터와 인덱스 로드
        load_pdf_metadata()
        load_pdf_index()

        # ChromaDB 초기화
        try:
            vectorstore = Chroma(
                persist_directory=BASE_CHROMA_DB_PATH,
                embedding_function=embeddings,
                collection_name="rag_collection"
            )
            logger.info("ChromaDB initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            # ChromaDB 디렉토리가 손상되었을 수 있으므로 재생성
            if os.path.exists(BASE_CHROMA_DB_PATH):
                shutil.rmtree(BASE_CHROMA_DB_PATH)
            os.makedirs(BASE_CHROMA_DB_PATH, exist_ok=True)
            # 다시 초기화 시도
            vectorstore = Chroma(
                persist_directory=BASE_CHROMA_DB_PATH,
                embedding_function=embeddings,
                collection_name="rag_collection"
            )
            logger.info("ChromaDB reinitialized successfully")

        # RAGManager 초기화
        global rag_manager
        rag_manager = RAGManager()
        rag_manager.vectorstore = vectorstore
        logger.info("RAGManager initialized successfully")

        return True
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        return False

# 초기화 코드를 함수로 이동하고 호출
if not initialize_rag_system():
    raise RuntimeError("데이터베이스 초기화에 실패했습니다.") 
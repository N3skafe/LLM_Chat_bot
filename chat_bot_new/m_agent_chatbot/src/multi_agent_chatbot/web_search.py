from langchain_community.tools import DuckDuckGoSearchRun
from typing import List, Dict, Optional
import logging
import time
import re
from datetime import datetime
import hashlib
from .utils import (
    logger, CacheManager, ErrorHandler,
    get_hash, format_timestamp, validate_result
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DuckDuckGo 검색 도구 초기화
search = DuckDuckGoSearchRun()

def enhance_search_query(query: str) -> str:
    """
    검색 쿼리를 개선하여 더 정확한 결과를 얻습니다.
    """
    # 현재 연도 추가
    current_year = datetime.now().year
    enhanced_query = f"{query} {current_year}"
    
    # 특정 키워드에 대한 처리
    if "대통령" in query and "미국" in query:
        enhanced_query = f"현재 미국 대통령 {current_year} 공식 정보"
    
    logger.info(f"검색 쿼리 개선: '{query}' -> '{enhanced_query}'")
    return enhanced_query

def filter_search_result(result: str) -> str:
    """
    검색 결과를 필터링하여 관련성 높은 정보만 추출합니다.
    """
    # 불필요한 텍스트 제거
    result = re.sub(r'존재하지 않는 이미지입니다\.', '', result)
    result = re.sub(r'\.{3,}', '...', result)
    
    # 현재 연도 이전의 정보 제거
    current_year = datetime.now().year
    lines = result.split('\n')
    filtered_lines = []
    
    for line in lines:
        # 미래 시점이나 현재 시점의 정보만 포함
        if any(str(year) in line for year in range(current_year-1, current_year+2)):
            filtered_lines.append(line)
    
    filtered_result = '\n'.join(filtered_lines)
    
    if not filtered_result.strip():
        return result  # 필터링된 결과가 없으면 원본 반환
    
    return filtered_result

class WebSearchManager:
    def __init__(self):
        self.cache_manager = CacheManager()
        self.error_handler = ErrorHandler()
        self.search = DuckDuckGoSearchRun()
    
    def search_with_cache(self, query):
        try:
            cache_key = get_hash(query.encode())
            
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                logger.info(f"Using cached result for query: {query}")
                return cached_result
            
            enhanced_query = enhance_search_query(query)
            
            def search_operation():
                return self.search.run(enhanced_query)
            
            result = self.error_handler.retry_operation(search_operation)
            filtered_result = filter_search_result(result)
            
            self.cache_manager.set(cache_key, filtered_result)
            return filtered_result
        except Exception as e:
            return self.error_handler.handle_error(e, "Error in web search")
    
    def validate_result(self, result):
        return validate_result(result)

# WebSearchManager 인스턴스 생성
web_search_manager = WebSearchManager()

# 기존 함수들을 WebSearchManager 메서드로 대체
def search_web(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    result = web_search_manager.search_with_cache(query)
    if web_search_manager.validate_result(result):
        return [{
            'title': query,
            'link': '',
            'body': result
        }]
    return []

def format_search_results(results: List[Dict[str, str]]) -> str:
    """
    검색 결과를 문자열로 포맷팅합니다.
    
    Args:
        results (List[Dict[str, str]]): 검색 결과 리스트
        
    Returns:
        str: 포맷팅된 검색 결과 문자열
    """
    logger.info(f"검색 결과 포맷팅 시작: {len(results)}개 결과")
    
    if not results:
        logger.warning("포맷팅할 검색 결과가 없습니다.")
        return "검색 결과가 없습니다."
    
    try:
        formatted_results = []
        for i, result in enumerate(results, 1):
            body = result.get('body', '설명 없음')
            
            formatted_result = (
                f"[{i}] 검색 결과:\n"
                f"{body}\n"
            )
            formatted_results.append(formatted_result)
            
            logger.debug(f"결과 {i} 포맷팅 완료")
        
        final_result = "\n".join(formatted_results)
        logger.info("검색 결과 포맷팅 완료")
        return final_result
        
    except Exception as e:
        logger.error(f"검색 결과 포맷팅 중 오류 발생: {str(e)}", exc_info=True)
        return "검색 결과를 포맷팅하는 중 오류가 발생했습니다." 
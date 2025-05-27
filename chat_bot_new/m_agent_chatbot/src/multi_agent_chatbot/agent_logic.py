from typing import List, Tuple, TypedDict, Annotated, Sequence, Literal, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langgraph.graph import StateGraph, END
import operator

from .llm_config import AVAILABLE_MODELS, llm_general, llm_coding, llm_reasoning, llm_image
from .rag_handler import get_relevant_documents, query_pdf_content
from .image_handler import analyze_image_with_llm
from .web_search import search_web, format_search_results
from PIL import Image


# --- Agent State ---
class AgentState(TypedDict):
    input_query: str
    chat_history: Annotated[Sequence[BaseMessage], operator.add]
    image_data: Optional[Image.Image] # PIL Image
    image_analysis_result: Optional[str]
    rag_context: Optional[str]
    web_search_results: Optional[str]
    selected_agent: Literal["coding_math", "reasoning", "general", "rag", "image_analysis_route", "web_search"]
    output_message: Optional[str]
    intermediate_steps: list # 디버깅용

# --- Nodes ---
def route_query_node(state: AgentState) -> AgentState:
    """쿼리 유형에 따라 다음 노드를 결정합니다."""
    query = state["input_query"].lower()
    image_data = state["image_data"]
    
    # 이미지 분석이 우선순위가 높을 경우
    if image_data:
        return {"selected_agent": "image_analysis_route"}

    # RAG 사용 여부 판단 (PDF 문서 검색)
    if any(kw in query for kw in ["pdf", "문서", "내 파일", "내 자료", "찾아줘", "검색", "요약"]):
        return {"selected_agent": "rag"}

    # 웹 검색이 필요한 경우 (최신 정보나 실시간 데이터가 필요한 경우)
    web_search_keywords = [
        # 시간 관련
        "현재", "지금", "요즘", "최근", "이번", "올해", "작년", "내년",
        # 상태/상황 관련
        "상태", "상황", "동향", "트렌드", "뉴스", "소식", "정보",
        # 특정 주제
        "가격", "시세", "환율", "주식", "날씨", "기후",
        # 영어 키워드
        "current", "latest", "news", "update"
    ]
    
    if any(kw in query for kw in web_search_keywords):
        return {"selected_agent": "web_search"}

    # 키워드 기반 라우팅
    if any(kw in query for kw in ["코드", "코딩", "프로그래밍", "알고리즘", "수학", "계산", "풀어줘"]):
        return {"selected_agent": "coding_math"}
    elif any(kw in query for kw in ["추론", "분석", "설명해줘", "왜", "어떻게 생각해"]):
        return {"selected_agent": "reasoning"}
    else:
        return {"selected_agent": "general"}

def image_analysis_node(state: AgentState) -> AgentState:
    """이미지를 분석하고 결과를 상태에 저장합니다."""
    image = state["image_data"]
    query = state["input_query"]
    
    if not image:
        return {"output_message": "이미지 분석을 요청했지만 이미지가 제공되지 않았습니다.", "image_analysis_result": None}

    analysis_prompt = query if query else "이 이미지에 대해 설명해주세요."
    
    print(f"Analyzing image with prompt: {analysis_prompt}")
    try:
        analysis_result = analyze_image_with_llm(image, analysis_prompt)
        return {
            "output_message": analysis_result,
            "image_analysis_result": analysis_result,
            "intermediate_steps": state.get("intermediate_steps", []) + [f"Image analysis result: {analysis_result[:200]}..."]
        }
    except Exception as e:
        error_msg = f"이미지 분석 중 오류가 발생했습니다: {str(e)}"
        print(error_msg)
        return {
            "output_message": error_msg,
            "image_analysis_result": None,
            "intermediate_steps": state.get("intermediate_steps", []) + [f"Image analysis error: {str(e)}"]
        }

def rag_node(state: AgentState) -> AgentState:
    """RAG를 사용하여 컨텍스트를 검색하고 상태에 저장합니다."""
    query = state["input_query"]
    print(f"Performing RAG search for: {query}")
    
    # get_relevant_documents 함수 사용
    relevant_docs = get_relevant_documents(query, k=3)
    
    if not relevant_docs:
        context = "관련 정보를 찾을 수 없습니다."
    else:
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    print(f"RAG Context (first 200 chars): {context[:200]}")
    return {
        "rag_context": context,
        "intermediate_steps": state.get("intermediate_steps", []) + [f"RAG context retrieved: {context[:200]}..."]
    }

def web_search_node(state: AgentState) -> AgentState:
    """웹 검색을 수행하고 결과를 상태에 저장합니다."""
    query = state["input_query"]
    
    print(f"Performing web search for: {query}")
    try:
        search_results = search_web(query)
        if not search_results:
            return {
                "output_message": "웹 검색 결과를 찾을 수 없습니다. 다른 키워드로 다시 시도해주세요.",
                "web_search_results": None,
                "intermediate_steps": state.get("intermediate_steps", []) + ["Web search: No results found"]
            }
        
        formatted_results = format_search_results(search_results)
        print(f"Web search results: {formatted_results[:200]}...")
        
        return {
            "web_search_results": formatted_results,
            "intermediate_steps": state.get("intermediate_steps", []) + [f"Web search results: {formatted_results[:200]}..."]
        }
    except Exception as e:
        error_msg = f"웹 검색 중 오류가 발생했습니다: {str(e)}"
        print(error_msg)
        return {
            "output_message": error_msg,
            "web_search_results": None,
            "intermediate_steps": state.get("intermediate_steps", []) + [f"Web search error: {str(e)}"]
        }

def llm_call_node(state: AgentState) -> AgentState:
    """선택된 에이전트(LLM)를 호출하고 응답을 생성합니다."""
    agent_name = state["selected_agent"]
    query = state["input_query"]
    history = state["chat_history"]
    rag_context = state.get("rag_context")
    image_analysis_context = state.get("image_analysis_result")
    web_search_context = state.get("web_search_results")

    # 시스템 프롬프트 설정
    system_prompt = """당신은 사용자와 대화하는 AI 챗봇입니다. 다음 규칙을 따라주세요:

1. 기본 응답 원칙:
   - 사용자의 질문에 직접적으로 답변하세요
   - 내부 생각이나 분석 과정을 출력하지 마세요
   - 불필요한 주어("제가", "저는" 등)를 사용하지 마세요
   - 영어로 된 내부 생각을 출력하지 마세요

2. 검색 결과 활용:
   - 웹 검색이나 PDF 검색 결과가 있다면, 그 정보를 바탕으로 답변하세요
   - 검색 결과를 자연스럽게 답변에 포함시키되, 출처를 명시하세요
   - 예시: "최근 뉴스에 따르면 [검색 결과 내용]입니다."
   - 검색 결과가 없는 경우: "관련 정보를 찾지 못했습니다."

3. 정보 부족 시:
   - 구체적으로 어떤 정보가 부족한지 알려주세요
   - 예시: "현재 날씨 정보가 필요합니다."
   - 추가 정보를 요청할 때는 간단명료하게 하세요

4. 오류 발생 시:
   - 구체적인 오류 내용을 알려주세요
   - 예시: "이미지 분석 중 오류가 발생했습니다: 이미지 형식이 지원되지 않습니다."

5. 인사 처리:
   - 인사에는 간단한 인사로만 응답하세요
   - 예시: "안녕하세요", "반갑습니다"

6. 답변 형식:
   - 검색 결과가 있는 경우:
     * "검색 결과에 따르면 [답변 내용]입니다."
     * "최근 정보에 의하면 [답변 내용]입니다."
   - 일반 답변의 경우:
     * 간단명료하게 직접 답변하세요
   - 정보 부족 시:
     * "답변을 위해 [필요한 정보]가 필요합니다."
   - 오류 발생 시:
     * "오류가 발생했습니다: [구체적인 오류 내용]" """

    # 컨텍스트가 있는 경우 시스템 프롬프트에 추가
    if rag_context or image_analysis_context or web_search_context:
        contexts = []
        if image_analysis_context:
            contexts.append(f"이미지 분석: {image_analysis_context}")
        if rag_context:
            contexts.append(f"문서 내용: {rag_context}")
        if web_search_context:
            contexts.append(f"웹 검색: {web_search_context}")
        
        system_prompt += f"\n\n참고할 정보:\n{' '.join(contexts)}"

    # 프롬프트 구성
    messages = [SystemMessage(content=system_prompt)]
    
    # 대화 기록 추가 (최근 3개만)
    if history:
        for msg in history[-6:]:
            if isinstance(msg, HumanMessage):
                messages.append(HumanMessage(content=msg.content))
            elif isinstance(msg, AIMessage):
                messages.append(AIMessage(content=msg.content))
    
    # 현재 쿼리 추가
    messages.append(HumanMessage(content=query))

    # 디버깅을 위한 프롬프트 로깅
    print("\n=== Final prompt to LLM ===")
    for msg in messages:
        print(f"\n[{msg.type}]:\n{msg.content}")
    print("\n========================\n")

    # 모델 선택
    if web_search_context:
        llm = llm_reasoning
        model_name = "llama3.2:latest"
    else:
        llm = AVAILABLE_MODELS.get(agent_name, llm_general)
        if agent_name == "coding_math":
            model_name = "deepseek-r1:latest"
        elif agent_name == "reasoning":
            model_name = "llama3.2:latest"
        elif agent_name == "general":
            model_name = "qwen3:latest"
        elif agent_name == "image_analysis":
            model_name = "llava:7b"
        else:
            model_name = "qwen3:latest"

    # LLM 호출
    try:
        response = llm.invoke(messages)
        response_text = response.content.strip()
        
        # 응답 후처리
        import re
        
        # 내부 생각 태그 제거
        response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        response_text = re.sub(r'<thought>.*?</thought>', '', response_text, flags=re.DOTALL)
        response_text = re.sub(r'<reasoning>.*?</reasoning>', '', response_text, flags=re.DOTALL)
        
        # 내부 독백 패턴 제거
        patterns_to_remove = [
            # 영어 패턴
            r"^(Okay|Alright|Well|Let me|I need to|I'll|I will|I should|I must|I have to).*?[.!?]",
            r"^(I think|I believe|I would say|I can see|I understand).*?[.!?]",
            r"^(Based on|According to|Looking at|Considering).*?[.!?]",
            
            # 한국어 패턴
            r"^(제가|저는|내가|나는).*?[.!?]",
            r"^(생각해보니|살펴보니|확인해보니).*?[.!?]",
            r"^(먼저|우선|일단).*?[.!?]",
            r"^(그럼|자|이제).*?[.!?]",
            r"^(응답:|답변:|AI:|Assistant:|챗봇:).*?[.!?]",
            r"^(~라고 생각합니다|~라고 판단됩니다|~라고 보입니다).*?[.!?]",
            r"^(사용자가|사용자는|질문이|요청이).*?[.!?]",
            r"^(~에 대해|~에 대해서).*?[.!?]",
            r"^(~을|~를).*?[.!?]",
            r"^(~하겠습니다|~하겠어요).*?[.!?]"
        ]
        
        for pattern in patterns_to_remove:
            response_text = re.sub(pattern, "", response_text, flags=re.IGNORECASE)
        
        # 불필요한 공백 제거 및 정리
        response_text = re.sub(r'\s+', ' ', response_text).strip()
        
        # 빈 응답 처리
        if not response_text:
            response_text = "죄송합니다. 다시 한번 질문해주시겠어요?"
        
        return {
            "output_message": response_text,
            "intermediate_steps": state.get("intermediate_steps", []) + [f"LLM response: {response_text[:200]}..."]
        }
    except Exception as e:
        error_msg = f"응답 생성 중 오류가 발생했습니다: {str(e)}"
        print(f"LLM 호출 중 오류 발생: {str(e)}")
        return {
            "output_message": error_msg,
            "intermediate_steps": state.get("intermediate_steps", []) + [f"LLM error: {str(e)}"]
        }


# --- Graph Definition ---
workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("query_router", route_query_node)
workflow.add_node("image_analyzer", image_analysis_node)
workflow.add_node("rag_retriever", rag_node)
workflow.add_node("web_searcher", web_search_node)
workflow.add_node("coding_math_agent", llm_call_node)
workflow.add_node("reasoning_agent", llm_call_node)
workflow.add_node("general_agent", llm_call_node)
workflow.add_node("final_llm_call", llm_call_node)

# 엣지 설정
workflow.set_entry_point("query_router")

def decide_next_step_after_routing(state: AgentState):
    if state["selected_agent"] == "image_analysis_route":
        return "image_analyzer"
    elif state["selected_agent"] == "rag":
        return "rag_retriever"
    elif state["selected_agent"] == "web_search":
        return "web_searcher"
    elif state["selected_agent"] == "coding_math":
        return "coding_math_agent"
    elif state["selected_agent"] == "reasoning":
        return "reasoning_agent"
    else:
        return "general_agent"

workflow.add_conditional_edges(
    "query_router",
    decide_next_step_after_routing,
    {
        "image_analyzer": "image_analyzer",
        "rag_retriever": "rag_retriever",
        "web_searcher": "web_searcher",
        "coding_math_agent": "coding_math_agent",
        "reasoning_agent": "reasoning_agent",
        "general_agent": "general_agent",
    }
)

def decide_after_preprocessing(state: AgentState):
    return "final_llm_call"

workflow.add_edge("image_analyzer", "final_llm_call")
workflow.add_edge("rag_retriever", "final_llm_call")
workflow.add_edge("web_searcher", "final_llm_call")

workflow.add_edge("coding_math_agent", END)
workflow.add_edge("reasoning_agent", END)
workflow.add_edge("general_agent", END)
workflow.add_edge("final_llm_call", END)

# 그래프 컴파일
app_graph = workflow.compile()

# 그래프 실행 함수
def run_graph(query: str, chat_history: List[Tuple[str, str]], image_pil: Optional[Image.Image] = None):
    lc_history = []
    for human, ai in chat_history:
        lc_history.append(HumanMessage(content=human))
        lc_history.append(AIMessage(content=ai))

    initial_state: AgentState = {
        "input_query": query,
        "chat_history": lc_history,
        "image_data": image_pil,
        "image_analysis_result": None,
        "rag_context": None,
        "web_search_results": None,
        "selected_agent": "general",
        "output_message": None,
        "intermediate_steps": []
    }
    
    final_state = app_graph.invoke(initial_state)
    
    return final_state.get("output_message", "죄송합니다. 답변을 생성하지 못했습니다.")
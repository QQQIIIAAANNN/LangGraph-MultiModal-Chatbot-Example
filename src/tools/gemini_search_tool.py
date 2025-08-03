"""Tool for interacting with Gemini models for text generation with grounding."""
from langchain.tools import tool
from typing import Dict, Any, Optional, List
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearchRetrieval, GoogleSearch
import os
from dotenv import load_dotenv
import traceback
import uuid # Import uuid for potential future use within the tool if needed, though saving is handled by ToolAgent

# Load environment variables
load_dotenv()

# Configure Gemini API key globally
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("需要設定 GEMINI_API_KEY 環境變數")
# genai.configure(api_key=GOOGLE_API_KEY) # Client creation handles configuration
client = genai.Client(api_key=GOOGLE_API_KEY)

@tool("perform_grounded_search")
def perform_grounded_search(query: str, model_name: str = "gemini-2.5-flash") -> Dict[str, Any]:
    """
    使用 Gemini API 搭配 Google 搜尋進行基於 grounding 的搜尋，並返回
    - 文字內容 (text_content)
    - 圖片 (images)
    - 影片 (videos)
    - Grounding 來源 (grounding_sources)
    - 搜尋建議 (search_suggestions)
    
    請確保已設定正確的 API 金鑰及相關依賴。
    """
    print(f"--- 開始執行 Gemini Grounded Search (模型: {model_name}) ---")
    print(f"查詢: {query}")

    try:
        # 建立 Google 搜尋工具 (使用 Retrieval 功能以啟動 Grounding)
        google_search_tool = Tool(
            google_search=GoogleSearch()
        )
        
        # 呼叫 Gemini API 的新版方法：client.models.generate_content
        response = client.models.generate_content(
            model=model_name,
            contents=query,
            config=GenerateContentConfig(
                tools=[google_search_tool],
                # 同時請求文字、圖片與影片模態 (若有支援影片則會返回)
                response_modalities=["TEXT"],
            )
        )
        print("--- Gemini API 回應 (部分) ---")
        
        # 初始化回傳結果字典
        result = {
            "text_content": "",
            "grounding_sources": [],
            "search_suggestions": [],
            "error": None
        }

        if not response.candidates or len(response.candidates) == 0:
            print("警告: Gemini API 回應不包含候選項目。")
            result["error"] = "Gemini API 回應無效 (缺少候選項目)"
            return result

        candidate = response.candidates[0]

        # 處理候選項目的 content parts (包括文字、圖片、影片)
        if candidate.content and candidate.content.parts:
            all_text_parts = []
            for part in candidate.content.parts:
                if hasattr(part, 'text') and part.text:
                    all_text_parts.append(part.text)
                elif hasattr(part, 'mime_type'):
                    if part.mime_type.startswith('image/'):
                        if hasattr(part, 'data') and isinstance(part.data, bytes):
                            print(f"提取到圖片: {part.mime_type}")
                            result["images"].append({
                                "mime_type": part.mime_type,
                                "data": part.data,  # 請依需求處理 raw bytes
                                "description": f"Gemini 搜尋結果圖片 (查詢: {query[:30]}...)"
                            })
                    elif part.mime_type.startswith('video/'):
                        if hasattr(part, 'data') and isinstance(part.data, bytes):
                            print(f"提取到影片: {part.mime_type}")
                            result["videos"].append({
                                "mime_type": part.mime_type,
                                "data": part.data,
                                "description": f"Gemini 搜尋結果影片 (查詢: {query[:30]}...)"
                            })
            result["text_content"] = "\n".join(all_text_parts).strip()
            if result["text_content"]:
                print(f"提取的文字內容: {result['text_content'][:100]}...")
            else:
                print("警告: 回應中未找到有效的文字內容。")
        else:
            print("警告: 回應候選項目中未找到 'content' 或 'parts'。")

        # 處理 grounding metadata 取得來源與搜尋建議
        if hasattr(candidate, "grounding_metadata") and candidate.grounding_metadata:
            metadata = candidate.grounding_metadata

            # 取得搜尋建議 (若有提供)
            if hasattr(metadata, "web_search_queries") and metadata.web_search_queries:
                result["search_suggestions"] = metadata.web_search_queries
                print(f"提取到 {len(result['search_suggestions'])} 個搜尋建議。")
            
            # 嘗試從 grounding_chunks 中提取來源
            if hasattr(metadata, "grounding_chunks") and metadata.grounding_chunks:
                for chunk in metadata.grounding_chunks:
                    if hasattr(chunk, "web") and chunk.web:
                        if hasattr(chunk.web, "uri") and hasattr(chunk.web, "title"):
                            result["grounding_sources"].append({
                                "uri": chunk.web.uri,
                                "title": chunk.web.title
                            })
                if result["grounding_sources"]:
                    print(f"提取到 {len(result['grounding_sources'])} 個 grounding 來源。")
                else:
                    print("Grounding metadata 存在，但未找到有效的來源。")
        else:
            print("警告: 回應中未包含 grounding_metadata。")

        if not result["text_content"] and not result["images"] and not result["videos"] and not result["grounding_sources"]:
            print("警告: Gemini 回應似乎不包含有效內容。")
            if not result["error"]:
                result["error"] = "[Gemini 未提供有效回覆]"

        print(f"--- Gemini Grounded Search 執行完畢 ---")
        return result

    except Exception as e:
        import traceback
        print(f"❌ 使用 Gemini grounding tool 時發生錯誤: {e}")
        traceback.print_exc()
        return {
            "error": f"執行 Gemini grounding 搜尋時發生錯誤: {str(e)}",
            "text_content": "",
            "grounding_sources": [],
            "search_suggestions": []
        }
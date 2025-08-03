"""
記憶管理 - 智能代理的記憶系統
實現短期記憶（對話狀態）和長期記憶（語義搜尋）

主要功能：
1. 短期記憶：保存對話狀態和上下文
2. 長期記憶：語義搜尋和個人化記錄
3. 訊息管理：自動修剪過長的對話
"""

import os
from typing import Optional, List, Dict, Any
from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from dotenv import load_dotenv

load_dotenv()

#==================================
# 短期記憶
#==================================
def create_checkpointer():
    """創建短期記憶檢查點"""
    return InMemorySaver()

#==================================
# 長期記憶
#==================================
def create_store():
    """創建長期記憶存儲"""
    try:
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            # 創建 Gemini 嵌入類別
            class GeminiEmbeddings:
                def __init__(self):
                    import google.generativeai as genai
                    genai.configure(api_key=gemini_key)
                    self.model_name = "gemini-embedding-001"
                
                def embed_query(self, text: str) -> List[float]:
                    """嵌入單個查詢文本"""
                    import google.generativeai as genai
                    try:
                        result = genai.embed_content(
                            model=self.model_name,
                            content=text,
                            output_dimensionality=768
                        )
                        return result['embedding']
                    except Exception:
                        return []
                
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    """嵌入多個文檔"""
                    import google.generativeai as genai
                    embeddings = []
                    
                    for text in texts:
                        try:
                            result = genai.embed_content(
                                model=self.model_name,
                                content=text,
                                output_dimensionality=768
                            )
                            embeddings.append(result['embedding'])
                        except Exception:
                            embeddings.append([0.0] * 768)
                    
                    return embeddings
            
            # 創建 Gemini embeddings store
            try:
                embeddings = GeminiEmbeddings()
                store = InMemoryStore(
                    index={
                        "embed": embeddings,
                        "dims": 768,
                    }
                )
                return store
            except Exception:
                return InMemoryStore()
        
        return InMemoryStore()
        
    except Exception:
        return InMemoryStore()

#==================================
# 訊息管理
#==================================
def trim_message_history(messages: List[BaseMessage], max_tokens: int = 2000) -> List[BaseMessage]:
    """
    智能修剪訊息歷史以控制長度
    
    修剪策略：
    1. 保留包含圖片的訊息（轉為文字標示）
    2. 超過5條訊息後才開始修剪
    3. 確保始終保留最新的對話上下文
    """
    if not messages:
        return messages
    
    # 預處理圖片訊息
    processed_messages = []
    for i, msg in enumerate(messages):
        # 處理工具訊息 - 這些很重要，不能跳過
        if msg.type == "tool":
            if hasattr(msg, 'content') and msg.content:
                processed_messages.append(msg)
        # 處理文字訊息
        elif hasattr(msg, 'content'):
            content = msg.content
            
            # 處理 LangGraph Studio 的 list 格式消息
            if isinstance(content, list):
                # 提取文字和圖片部分
                text_parts = []
                has_image = False
                
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            has_image = True
                    elif isinstance(item, str):
                        text_parts.append(item)
                
                # 組合文字內容
                text_content = " ".join(text_parts).strip()
                if not text_content:
                    text_content = "請分析這張圖片" if has_image else "用戶請求"
                
                # 添加圖片標識
                if has_image:
                    final_content = f"{text_content} [用戶上傳了圖片]"
                else:
                    final_content = text_content
                
                # 創建新的消息對象
                from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
                if msg.type == "human":
                    processed_msg = HumanMessage(content=final_content)
                elif msg.type == "ai":
                    processed_msg = AIMessage(content=final_content)
                elif msg.type == "system":
                    processed_msg = SystemMessage(content=final_content)
                else:
                    processed_msg = HumanMessage(content=final_content)
                
                processed_messages.append(processed_msg)
                
            # 處理字符串格式的消息
            elif isinstance(content, str):
                # 檢查是否包含 base64 圖片
                if "data:image/" in content and "base64," in content:
                    from langchain_core.messages import HumanMessage, AIMessage
                    
                    if msg.type == "human":
                        text_part = content.split("data:image/")[0].strip()
                        if text_part:
                            simplified_content = f"{text_part} [用戶上傳了圖片]"
                        else:
                            simplified_content = "用戶上傳了一張圖片，請分析內容"
                        processed_msg = HumanMessage(content=simplified_content)
                    elif msg.type == "ai":
                        simplified_content = "[包含圖片的AI回應]"
                        processed_msg = AIMessage(content=simplified_content)
                    else:
                        simplified_content = "[多媒體內容]"
                        processed_msg = HumanMessage(content=simplified_content)
                    
                    processed_messages.append(processed_msg)
                else:
                    # 普通文字訊息
                    if content.strip():
                        processed_messages.append(msg)
            else:
                # 其他格式的 content
                content_str = str(content) if content else ""
                if content_str.strip():
                    processed_messages.append(msg)
    
    # 溫和的修剪策略
    if len(processed_messages) <= 5:
        return processed_messages
    
    # 逐步修剪直到符合 token 限制
    try:
        trimmed = trim_messages(
            processed_messages,
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=max_tokens,
            start_on="human",
            end_on=("human", "tool"),
            include_system=True,
        )
        
        if len(trimmed) < 2:
            # 備用策略：保留最後幾條訊息
            keep_count = min(5, len(processed_messages))
            important_messages = processed_messages[-keep_count:]
            return important_messages
        
        return trimmed
        
    except Exception:
        # 錯誤處理：保留最後幾條重要訊息
        fallback_count = min(3, len(processed_messages))
        fallback_messages = processed_messages[-fallback_count:]
        return fallback_messages

#==================================
# 長期記憶操作
#==================================
def save_to_long_term_memory(store: InMemoryStore, user_id: str, content: str, memory_type: str = "memory"):
    """保存內容到長期記憶"""
    if not store:
        return
        
    try:
        from datetime import datetime
        memory_id = f"{memory_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        namespace = (user_id, "memories")
        
        memory_data = {
            "text": content,
            "type": memory_type,
            "timestamp": datetime.now().isoformat()
        }
        
        store.put(namespace, memory_id, memory_data)
        
    except Exception:
        pass

def search_long_term_memory(store: InMemoryStore, user_id: str, query: str, limit: int = 3) -> str:
    """搜尋長期記憶"""
    if not store:
        return ""
        
    try:
        namespace = (user_id, "memories")
        
        # 嘗試語義搜尋
        if hasattr(store, 'search'):
            items = store.search(namespace, query=query, limit=limit)
        else:
            # 如果沒有語義搜尋，列出所有記憶
            items = store.list(namespace)
            items = items[:limit] if items else []
        
        if not items:
            return ""
        
        # 格式化記憶上下文
        memories = []
        for item in items:
            memory_data = item.value
            if isinstance(memory_data, dict) and "text" in memory_data:
                memories.append(memory_data["text"])
        
        if memories:
            return "\n## 相關記憶\n" + "\n".join(f"- {memory}" for memory in memories)
        
        return ""
        
    except Exception:
        return ""
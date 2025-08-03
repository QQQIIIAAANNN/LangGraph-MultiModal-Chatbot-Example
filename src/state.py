"""
狀態管理 - LangGraph的記憶系統(這個檔案定義了代理程式如何記住對話內容)
"""

from typing import Annotated, TypedDict, List, Dict, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    代理程式的狀態（就像代理程式的大腦記憶）

    這個狀態會在整個對話過程中保存所有訊息和生成的檔案
    LangGraph會自動管理這些訊息的新增和更新
    """
    # 對話記錄：所有的使用者訊息、AI回應、工具結果都會存在這裡
    # add_messages 是LangGraph提供的特殊功能，會自動處理訊息的合併
    messages: Annotated[list[BaseMessage], add_messages]
    
    # 生成的檔案：存放工具生成的圖片、文檔等檔案資訊
    # 這個欄位專門用於UI顯示生成的內容
    generated_files: List[Dict[str, Any]]


def create_empty_state() -> AgentState:
    """
    創建一個空的狀態來開始新對話

    Returns:
        包含空訊息列表和空檔案列表的新狀態
    """
    return AgentState(
        messages=[],           # 開始時沒有任何對話記錄
        generated_files=[]     # 開始時沒有任何生成的檔案
    )


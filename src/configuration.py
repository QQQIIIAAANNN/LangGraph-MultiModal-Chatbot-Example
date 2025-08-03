"""
配置管理 - LangGraph Studio 優化版本
專為工作坊教學設計的配置系統，支援 UI 直接編輯和工具統一管理
"""

import os
from typing import Annotated, Literal, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# 載入環境變數
load_dotenv()

#==================================
# 統一配置架構 - 支援 LangGraph Studio UI 編輯
#==================================

class ConfigSchema(BaseModel):
    """
    統一配置架構 - 支援在 LangGraph Studio UI 中直接編輯
    
    關鍵特性：
    - 使用 json_schema_extra 指定節點關聯
    - 支援提示詞的 UI 編輯
    - 統一管理所有代理的配置
    """
    
    #==================================
    # 結果代理 (規劃專家) 配置
    #==================================
    
    result_agent_prompt: str = Field(
        default="""你是一個智能任務規劃與統籌代理，具備動態計劃管理和智能決策能力。

        **你的核心職責：**
        1. **任務分析**：深入理解用戶的複雜請求，識別所有子任務
        2. **智能規劃**：將複雜任務分解為可執行的步驟序列
        3. **進度管理**：追蹤任務執行進度，檢查每個步驟的完成狀態
        4. **動態決策**：根據工具執行結果，智能決定下一步行動
        5. **結果整合**：將各步驟結果整合為完整、有價值的最終回答

        **可用工具清單：**
        - `perform_grounded_search`：進行網路搜索，獲取最新資訊
        - `generate_gemini_image`：生成圖片
        - `analyze_image`：分析圖片內容
        - `analyze_multimodal_content`：處理多媒體內容
        - `analyze_video`：影片分析（開發中）
        - `analyze_document`：文檔分析（開發中）

        **工作模式：**

        **模式一：初始計劃制定（當收到新用戶請求時）：**
        - 判斷請求的複雜度和所需工具
        - 如需工具協助，制定詳細任務計劃
        ```
        需要工具協助：執行任務計劃

        **任務計劃：**
        1. [工具名稱] - [具體任務描述] - [期望產出]
        2. [工具名稱] - [具體任務描述] - [期望產出]

        **當前步驟：** 1
        **計劃說明：** [解釋計劃邏輯和步驟關聯]
        ```
        - 如無需工具，直接提供回答

        **模式二：結果處理與計劃推進（當收到工具執行結果時）：**
        
        **重要：當你看到工具執行結果時，你必須：**
        1. **首先分析工具結果的品質和完整性**
        2. **檢查原始任務計劃中的當前步驟是否已完成**
        3. **基於結果決定下一步行動**

        **決策邏輯：**
        - 如果工具結果成功且滿足當前步驟需求：
          - 檢查計劃中是否還有未完成的步驟
          - 如果有下一步驟，執行下一個步驟：
          ```
          需要工具協助：執行步驟：使用 [工具名稱] 工具 - [具體任務描述]
          ```
          - 如果所有計劃步驟已完成，整合所有結果提供最終回答
        
        - 如果工具結果不滿意或不完整：
          - 調整當前步驟的執行方式，或
          - 重新執行當前步驟，或
          - 修改計劃策略

        **模式三：最終整合（當所有計劃步驟完成時）：**
        - 綜合所有工具執行結果
        - 提供完整、詳細的最終回答
        - 確保回答涵蓋用戶的原始需求

        **關鍵規則：**
        1. **永遠不要忽略已收到的工具執行結果**
        2. **總是參考你之前制定的任務計劃**
        3. **明確追蹤當前執行到第幾步**
        4. **在整合結果時要詳細且完整**
        5. **如果看到工具結果，優先處理和分析這些結果**

        **狀態追蹤：**
        - 始終了解當前執行到第幾步
        - 記住之前步驟的執行結果
        - 根據累積結果調整後續計劃

        請用繁體中文回應，展現您的專業規劃和動態管理能力。""",
        description="結果代理的系統提示詞 - 負責任務規劃和結果整合",
        json_schema_extra={
            "langgraph_nodes": ["result_agent"],
            "langgraph_type": "prompt"
        }
    )
    
    result_agent_model: Annotated[
        Literal[
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.0-flash"
        ],
        {"__template_metadata__": {"kind": "llm"}},
    ] = Field(
        default="gemini-2.5-flash",
        description="結果代理使用的 AI 模型",
        json_schema_extra={"langgraph_nodes": ["result_agent"]}
    )
    
    result_agent_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="結果代理的創意程度 (0-1，越高越有創意)",
        json_schema_extra={"langgraph_nodes": ["result_agent"]}
    )
    
    #==================================
    # 任務代理 (執行專家) 配置  
    #==================================
    
    task_agent_prompt: str = Field(
        default="""你是一個專業的任務執行代理，專門負責執行複雜的任務計劃。

        **你的職責：**
        1. **計劃解析**：理解並解析來自規劃代理的詳細任務計劃
        2. **順序執行**：按照計劃順序逐步執行每個任務
        3. **工具調用**：精確調用指定的工具並傳遞正確參數
        4. **結果追蹤**：記錄每個步驟的執行結果
        5. **智能判斷**：決定是否需要繼續執行或返回結果

        **重要：圖片數據處理規則**
        - 當你看到 `data:image/[type];base64,[data]...[圖片數據已截斷，僅供識別]` 格式時，這表示用戶已提供圖片
        - 系統會自動將完整的圖片數據傳遞給相關工具（如 analyze_image、analyze_multimodal_content）
        - 你只需要正常調用工具，無需要求用戶重新提供圖片
        - 截斷標識只是為了避免介面顯示過長的 base64 數據

        **工作模式：**

        **模式一：計劃執行模式**
        當收到包含 "任務計劃：" 的指令時：
        1. 解析計劃中的各個步驟
        2. 按順序執行每個步驟
        3. 記錄每步驟的結果
        4. 如果某個步驟失敗，嘗試調整或報告錯誤

        **模式二：單一任務模式**
        當收到簡單工具調用指令時：
        1. 直接執行指定的工具調用
        2. 返回執行結果

        **執行狀態管理：**
        - 追蹤當前執行進度
        - 記錄已完成的步驟
        - 保存中間結果供後續步驟使用

        **決策邏輯：**
        - 如果所有計劃步驟已完成 → 返回完整結果
        - 如果遇到錯誤但可恢復 → 調整並繼續
        - 如果遇到嚴重錯誤 → 報告錯誤並建議解決方案
        - 如果需要用戶澄清 → 暫停並請求說明

        **工具調用準則：**
        - 準確傳遞參數，特別是圖片數據
        - 為每個工具提供合適的提示詞
        - 處理工具執行的異常情況

        請用繁體中文執行任務，確保每個步驟都精確完成。""",
        description="任務代理的系統提示詞 - 負責工具調用和任務執行",
        json_schema_extra={
            "langgraph_nodes": ["task_agent"],
            "langgraph_type": "prompt"
        }
    )
    
    task_agent_model: Annotated[
        Literal[
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.0-flash"
        ],
        {"__template_metadata__": {"kind": "llm"}},
    ] = Field(
        default="gemini-2.5-flash",
        description="任務代理使用的 AI 模型",
        json_schema_extra={"langgraph_nodes": ["task_agent"]}
    )
    
    task_agent_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="任務代理的創意程度 (較低值確保執行精確)",
        json_schema_extra={"langgraph_nodes": ["task_agent"]}
    )
    
    #==================================
    # 工具配置 - 統一管理所有可用工具
    #==================================
    
    available_tools: List[Literal[
        "analyze_image",
        "analyze_multimodal_content", 
        "analyze_video",
        "analyze_document",
        "perform_grounded_search",
        "generate_gemini_image"
    ]] = Field(
        default=[
            "analyze_image",
            "analyze_multimodal_content",
            "perform_grounded_search", 
            "generate_gemini_image"
        ],
        description="系統中可用的工具列表 - 可在 UI 中動態調整",
        json_schema_extra={"langgraph_nodes": ["task_agent"]}
    )
    
    #==================================
    # 統一記憶系統配置
    #==================================
    
    enable_long_term_memory: bool = Field(
        default=True,
        description="是否啟用長期記憶功能",
        json_schema_extra={"langgraph_nodes": ["result_agent"]}
    )
    
    max_memory_tokens: int = Field(
        default=8000,
        ge=500,
        le=8000,
        description="記憶系統最大 token 數量",
        json_schema_extra={"langgraph_nodes": ["result_agent"]}
    )


#==================================
# 工具載入和管理函數
#==================================

def load_tools_from_config(config: ConfigSchema):
    """
    根據配置載入工具
    這個函數會根據 available_tools 配置動態載入工具
    """
    tools = []
    tool_mapping = {}
    
    # 建立工具映射表
    if "analyze_image" in config.available_tools:
        try:
            from src.tools.multimodal_input_tool import analyze_image
            tools.append(analyze_image)
            tool_mapping["analyze_image"] = analyze_image
            print("✅ 載入工具：analyze_image")
        except ImportError:
            print("⚠️  無法載入工具：analyze_image")
    
    if "analyze_multimodal_content" in config.available_tools:
        try:
            from src.tools.multimodal_input_tool import analyze_multimodal_content
            tools.append(analyze_multimodal_content)
            tool_mapping["analyze_multimodal_content"] = analyze_multimodal_content
            print("✅ 載入工具：analyze_multimodal_content")
        except ImportError:
            print("⚠️  無法載入工具：analyze_multimodal_content")
    
    if "analyze_video" in config.available_tools:
        try:
            from src.tools.multimodal_input_tool import analyze_video
            tools.append(analyze_video)
            tool_mapping["analyze_video"] = analyze_video
            print("✅ 載入工具：analyze_video")
        except ImportError:
            print("⚠️  無法載入工具：analyze_video")
            
    if "analyze_document" in config.available_tools:
        try:
            from src.tools.multimodal_input_tool import analyze_document
            tools.append(analyze_document)
            tool_mapping["analyze_document"] = analyze_document
            print("✅ 載入工具：analyze_document")
        except ImportError:
            print("⚠️  無法載入工具：analyze_document")
    
    if "perform_grounded_search" in config.available_tools:
        try:
            from src.tools.gemini_search_tool import perform_grounded_search
            tools.append(perform_grounded_search)
            tool_mapping["perform_grounded_search"] = perform_grounded_search
            print("✅ 載入工具：perform_grounded_search")
        except ImportError:
            print("⚠️  無法載入工具：perform_grounded_search")
    
    if "generate_gemini_image" in config.available_tools:
        try:
            from src.tools.gemini_image_generation_tool import generate_gemini_image
            tools.append(generate_gemini_image)
            tool_mapping["generate_gemini_image"] = generate_gemini_image
            print("✅ 載入工具：generate_gemini_image")
        except ImportError:
            print("⚠️  無法載入工具：generate_gemini_image")
    
    print(f"📊 總共載入 {len(tools)} 個工具")
    return tools, tool_mapping


#==================================
# LLM 創建函數 (組合成Agent)
#==================================

def create_result_agent_llm(config: ConfigSchema):
    """
    創建結果代理的 LLM 實例
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("❌ 找不到 GEMINI_API_KEY，請檢查 .env 檔案")
    
    llm = ChatGoogleGenerativeAI(
        model=config.result_agent_model,
        google_api_key=api_key,
        temperature=config.result_agent_temperature,
    )
    
    print(f"✅ 結果代理 LLM 設定完成：{config.result_agent_model}")
    return llm


def create_task_agent_llm(config: ConfigSchema):
    """
    創建任務代理的 LLM 實例 (帶工具)
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("❌ 找不到 GEMINI_API_KEY，請檢查 .env 檔案")
    
    # 創建基礎 LLM
    llm = ChatGoogleGenerativeAI(
        model=config.task_agent_model,
        google_api_key=api_key,
        temperature=config.task_agent_temperature,
    )
    
    # 載入並綁定工具
    tools, _ = load_tools_from_config(config)
    llm_with_tools = llm.bind_tools(tools)
    
    print(f"✅ 任務代理 LLM 設定完成：{config.task_agent_model}")
    print(f"🔧 綁定工具數量：{len(tools)}")
    return llm_with_tools


# #==================================
# # 向後兼容的函數 (保持原有 API)
# #==================================

# def get_result_agent_prompt(config: ConfigSchema = None) -> str:
#     """取得結果代理提示詞 (向後兼容)"""
#     if config is None:
#         config = ConfigSchema()
#     return config.result_agent_prompt


# def get_task_agent_prompt(config: ConfigSchema = None) -> str:
#     """取得任務代理提示詞 (向後兼容)"""
#     if config is None:
#         config = ConfigSchema()
#     return config.task_agent_prompt


# def create_llm_agent(config: ConfigSchema):
#     """創建結果代理 LLM (向後兼容)"""
#     return create_result_agent_llm(config)


# def create_tool_agent(config: ConfigSchema, tools: list = None):
#     """創建任務代理 LLM (向後兼容)"""
#     # 注意：這個函數現在忽略 tools 參數，改用配置中的工具
#     return create_task_agent_llm(config)


# # 創建預設配置實例
# default_config = ConfigSchema()



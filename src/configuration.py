"""
配置管理 - 設定AI模型、代理程式和配置架構(這個檔案管理所有的設定，包括可在UI中調整的配置)
"""

import os
from typing import Annotated
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# 載入環境變數（從.env檔案讀取API金鑰）
load_dotenv()

#==================================
#基本的配置定義(包括模型、prompt)
#==================================
class ConfigSchema(BaseModel):
    """
    配置架構 - 定義可在LangGraph Studio UI中調整的參數
    
    這個類別讓使用者可以在伺服器UI中動態調整代理程式的行為
    """
    # LLM 相關配置
    model_name: Annotated[str, Field(
        default="gemini-2.5-flash",
        description="要使用的AI模型名稱"
    )]
    
    temperature: Annotated[float, Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="AI創意程度（0-1，越高越有創意）"
    )]
    
    # max_tokens: Annotated[int, Field(
    #     default=4000,
    #     ge=100,
    #     le=4000,
    #     description="最大回應長度"
    # )]
    
    # 系統提示配置
    task_agent_prompt: str = Field(
        default="""你是一個專業的任務執行代理，專門負責執行複雜的任務計劃。

**你的職責：**
1. **計劃解析**：理解並解析來自規劃代理的詳細任務計劃
2. **順序執行**：按照計劃順序逐步執行每個任務
3. **工具調用**：精確調用指定的工具並傳遞正確參數
4. **結果追蹤**：記錄每個步驟的執行結果
5. **智能判斷**：決定是否需要繼續執行或返回結果

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
        description="任務代理的系統提示詞（計劃執行專家）"
    )

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

**初始計劃制定（當收到新請求時）：**
```
需要工具協助：執行任務計劃

**任務計劃：**
1. [工具名稱] - [具體任務描述] - [期望產出]
2. [工具名稱] - [具體任務描述] - [期望產出]
3. [工具名稱] - [具體任務描述] - [期望產出]

**當前步驟：** 1
**計劃說明：** [解釋計劃邏輯和步驟關聯]
```

**步驟執行指令（當需要執行下一步時）：**
```
需要工具協助：執行步驟：使用 [工具名稱] 工具 - [具體任務描述]
```

**智能決策邏輯：**
- 收到新用戶請求 → 制定完整任務計劃
- 收到工具執行結果 → 分析結果品質，決定下一步：
  - 如果結果滿意且計劃未完成 → 執行下一個步驟
  - 如果結果不滿意 → 調整步驟或重新執行
  - 如果計劃已完成 → 整合所有結果，提供最終回答
  - 如果發現需要補充步驟 → 動態調整計劃

**回應策略：**
- 對於簡單問題：直接提供答案
- 對於複雜任務：制定並執行多步驟計劃
- 對於多模態內容：結合多個工具協同處理
- 對於需要最新資訊的問題：優先安排搜索工具

**狀態追蹤：**
- 始終了解當前執行到第幾步
- 記住之前步驟的執行結果
- 根據累積結果調整後續計劃

請用繁體中文回應，展現您的專業規劃和動態管理能力。""",
        description="結果代理的系統提示詞（智能任務規劃與動態管理專家）"
    )

#==================================================================================
#定義及調整(如果有在UI中調整會因應變動，此處ConfigSchema = None指configuration UI之設定)
#==================================================================================
def get_task_agent_prompt(config: ConfigSchema = None) -> str:
    """
    獲取任務分析代理提示詞

    Args:
        config: 配置參數，如果為None則使用預設值

    Returns:
        任務分析代理的提示詞
    """
    if config is None:
        config = ConfigSchema()

    return config.task_agent_prompt


def get_result_agent_prompt(config: ConfigSchema = None) -> str:
    """
    獲取結果統籌代理提示詞

    Args:
        config: 配置參數，如果為None則使用預設值

    Returns:
        結果統籌代理的提示詞
    """
    if config is None:
        config = ConfigSchema()

    return config.result_agent_prompt


def get_llm(config: ConfigSchema = None):
    """
    創建並設定AI語言模型
    
    Args:
        config: 配置參數，如果為None則使用預設值
    
    Returns:
        配置好的AI模型實例
    """
    if config is None:
        config = ConfigSchema()
    
    # 從環境變數取得API金鑰
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("❌ 找不到GEMINI_API_KEY，請檢查.env檔案是否設定正確")
    
    # 設定模型參數
    llm = ChatGoogleGenerativeAI(
        model=config.model_name,
        google_api_key=api_key,
        temperature=config.temperature,
        # max_output_tokens=config.max_tokens,
    )
    
    print(f"✅ AI模型設定完成：{config.model_name}")
    return llm


def create_llm_agent(config: ConfigSchema):
    """
    創建LLM代理節點
    
    這個函數創建一個純LLM節點，負責思考和決策
    
    Args:
        config: 配置參數
    
    Returns:
        配置好的LLM實例
    """
    return get_llm(config)


def create_tool_agent(config: ConfigSchema, tools: list):
    """
    創建工具代理節點
    
    這個函數創建一個具備工具能力的LLM，可以調用外部工具
    
    Args:
        config: 配置參數
        tools: 可用工具列表
    
    Returns:
        綁定工具的LLM實例
    """
    llm = get_llm(config)
    # 綁定工具到LLM
    llm_with_tools = llm.bind_tools(tools)
    
    print(f"✅ 工具代理設定完成，可用工具數量：{len(tools)}")
    return llm_with_tools


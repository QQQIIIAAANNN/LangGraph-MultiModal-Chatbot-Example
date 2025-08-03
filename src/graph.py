"""
智能多代理任務規劃系統 - LangGraph 工作坊
這是一個展示如何使用 LangGraph 建立智能代理系統的範例

主要功能：
1. 智能任務規劃和分解
2. 動態執行管理
3. 多模態內容處理（圖片、文字）
4. 記憶功能（短期和長期）

作者：LangGraph 工作坊
"""

import time
import base64
import os
from typing import List, Dict
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from src.state import AgentState, create_empty_state
from src.configuration import ConfigSchema, create_result_agent_llm, create_task_agent_llm, load_tools_from_config
from src.memory import create_checkpointer, create_store, trim_message_history, save_to_long_term_memory, search_long_term_memory

#==================================
# 輔助函數區塊
#==================================
def extract_image_data(original_query) -> str:
    """從用戶查詢中提取圖片數據"""
    image_base64 = None
    
    if isinstance(original_query, list):
        for item in original_query:
            if isinstance(item, dict) and item.get("type") == "image_url":
                image_url = item.get("image_url", {}).get("url", "")
                if image_url.startswith("data:image/"):
                    image_base64 = image_url
                    break
    elif isinstance(original_query, str) and "data:image/" in original_query:
        import re
        pattern = r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+'
        matches = re.findall(pattern, original_query)
        if matches:
            image_base64 = matches[0]
    
    return image_base64

def extract_step_instruction(instruction: str) -> dict:
    """從指令中提取步驟信息"""
    try:
        if "執行步驟：" in instruction:
            step_content = instruction.split("執行步驟：")[1].strip()
            
            if " - " in step_content:
                parts = step_content.split(" - ", 1)
                tool_part = parts[0].strip()
                description = parts[1].strip()
                
                if "使用" in tool_part and "工具" in tool_part:
                    tool_name = tool_part.split("使用")[1].split("工具")[0].strip()
                else:
                    tool_name = tool_part
                
                return {'tool': tool_name, 'description': description}
        return None
    except Exception:
        return None

def clean_message_content(content) -> str:
    """
    清理訊息內容，將完整的 base64 圖片數據替換為截斷的版本，
    以提示 LLM 圖片的存在，同時避免 token 過量。
    """
    if not content:
        return ""

    import re
    # 正則表達式，用於尋找 base64 數據 URI
    pattern = r'data:image/([^;]+);base64,([A-Za-z0-9+/=]+)'

    def replacer(match):
        """正則替換函數，生成截斷後的 base64 標識。"""
        # 截斷 base64 數據，保留前 10 個字符
        truncated_data = match.group(2)[:10]
        # 重組為新的標識符，並添加說明
        return f"data:image/{match.group(1)};base64,{truncated_data}...[這是用戶上傳的圖片，數據已截斷僅供識別]"

    # 處理字串格式
    if isinstance(content, str):
        return re.sub(pattern, replacer, content)
    
    # 處理列表格式（例如 LangGraph Studio 的輸入）
    elif isinstance(content, list):
        processed_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text = item.get("text", "")
                    processed_parts.append(re.sub(pattern, replacer, text))
                elif item.get("type") == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    processed_parts.append(re.sub(pattern, replacer, image_url))
            elif isinstance(item, str):
                processed_parts.append(re.sub(pattern, replacer, item))
        
        return " ".join(part for part in processed_parts if part).strip()
    
    # 處理其他格式
    else:
        try:
            return re.sub(pattern, replacer, str(content))
        except Exception:
            return "[無法處理的訊息內容]"

#==================================
# 核心代理節點定義
#==================================
def result_agent(state: AgentState, config: RunnableConfig, *, store):
    """
    結果統籌代理 - 任務規劃和結果整合專家
    
    職責：
    1. 分析用戶請求
    2. 制定執行計劃
    3. 管理執行進度
    4. 整合最終結果
    """

    #==================================
    # 基本配置初始化區塊
    #==================================
    configuration = config.get("configurable", {})
    config_schema = ConfigSchema(**configuration)
    user_id = configuration.get("user_id", "workshop_user")
    
    # 創建 LLM 代理
    model = create_result_agent_llm(config_schema)
    result_prompt = config_schema.result_agent_prompt
    messages = state["messages"]
    
    #==================================
    # 訊息清理區塊 - 移除base64數據但保留所有訊息類型
    #==================================
    cleaned_messages = []
    for msg in messages:
        # 特殊處理：確保工具訊息始終被保留
        if msg.type == "tool":
            cleaned_messages.append(msg)
            continue
            
        if hasattr(msg, 'content') and msg.content:
            # 檢查是否包含圖片數據的 human 訊息
            if msg.type == "human" and (
                (isinstance(msg.content, str) and "data:image/" in msg.content) or
                (isinstance(msg.content, list) and any(
                    isinstance(item, dict) and item.get("type") == "image_url" 
                    for item in msg.content
                ))
            ):
                # 使用clean_message_content清理訊息
                cleaned_content = clean_message_content(msg.content)
                cleaned_msg = HumanMessage(content=cleaned_content)
                cleaned_messages.append(cleaned_msg)
            else:
                # 保留所有其他類型的訊息（包括 ai、system 等）
                cleaned_messages.append(msg)
        else:
            # 保留沒有內容或內容為空的訊息（但工具訊息已在上面處理）
            if msg.type != "tool":  # 避免重複處理工具訊息
                cleaned_messages.append(msg)
    
    #==================================
    # 訊息處理和記憶搜尋區塊
    #==================================
    # 訊息管理 - 按照上下文保留最近的訊息
    MAX_MESSAGES = 25
    trimmed_messages = cleaned_messages[-MAX_MESSAGES:]
    
    # 記憶搜尋（僅在需要時）
    memory_context = ""
    if cleaned_messages and len(cleaned_messages) > 1:
        last_user_message = None
        for msg in reversed(cleaned_messages):
            if msg.type == "human":
                last_user_message = str(msg.content)
                break
        
        if last_user_message and len(last_user_message) > 10:
            personal_keywords = ["我", "你記得", "之前", "上次", "習慣", "喜歡", "偏好"]
            if any(keyword in last_user_message for keyword in personal_keywords):
                memory_context = search_long_term_memory(store, user_id, last_user_message, limit=2)
                if memory_context:
                    result_prompt = f"{result_prompt}\n\n{memory_context}"

    #==================================
    # 圖片處理區塊 - 教學用簡化版本
    #==================================
    # 當工具生成圖片時，自動轉換為 base64 格式供 UI 顯示
    processed_images = []
    recent_tool_messages = [msg for msg in messages[-5:] if msg.type == "tool"]
    for tool_msg in recent_tool_messages:
        if hasattr(tool_msg, 'name') and tool_msg.name == "generate_gemini_image":
            tool_content = str(tool_msg.content)
            
            if "'generated_files':" in tool_content and "'path':" in tool_content:
                try:
                    path_start = tool_content.find("'path': '") + 9
                    path_end = tool_content.find("'", path_start)
                    image_path = tool_content[path_start:path_end].replace('\\\\', '\\')
                    
                    type_start = tool_content.find("'type': '") + 9
                    type_end = tool_content.find("'", type_start)
                    file_type = tool_content[type_start:type_end]
                    
                    if os.path.exists(image_path):
                        with open(image_path, "rb") as image_file:
                            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                        base64_data = f"data:{file_type};base64,{encoded_string}"
                        processed_images.append(base64_data)
                except Exception:
                    pass

    #==================================
    # LLM 調用和回應處理區塊
    #==================================
    # 構建訊息序列 - 使用清理後的訊息
    result_messages = [SystemMessage(content=result_prompt)]
    if trimmed_messages:
        result_messages.extend(trimmed_messages)

    # 檢查是否有未處理的工具結果需要整合
    recent_tool_messages = [msg for msg in trimmed_messages if msg.type == "tool"]
    if recent_tool_messages:
        # 為每個工具結果創建明確的整合提示
        for tool_msg in recent_tool_messages:
            tool_name = getattr(tool_msg, 'name', 'unknown')
            tool_content = str(tool_msg.content)
            
            # 創建明確標示工具結果的人類訊息
            tool_summary = (
                f"以下是 {tool_name} 工具的執行結果：\n\n"
                f"{tool_content}\n\n"
                f"請基於此結果完成你的任務並提供最終回答。"
            )
            result_messages.append(HumanMessage(content=tool_summary))

    # 驗證訊息
    validated_messages = []
    for msg in result_messages:
        if hasattr(msg, 'content') and msg.content and str(msg.content).strip():
            validated_messages.append(msg)

    if not validated_messages:
        validated_messages = [
            SystemMessage(content="你是一個智能助手。"),
            HumanMessage(content="請協助處理用戶請求。")
        ]

    # 確保符合 Gemini API 要求
    last_message = validated_messages[-1] if validated_messages else None
    if last_message and last_message.type != "human":
        # 只有在沒有工具結果時才添加通用提示
        tool_messages = [msg for msg in validated_messages if msg.type == "tool"]
        if not tool_messages:
            validated_messages.append(
                HumanMessage(content="請協助處理這個請求。")
            )

    # 調用 LLM
    try:
        response = model.invoke(validated_messages)
        
        # 僅在回應格式異常或為空時提供備用方案
        if not hasattr(response, 'content') or not response.content or not str(response.content).strip():
            response = AIMessage(content="模型回應為空，處理中斷。請檢查日誌或重試。")
            
    except Exception:
        response = AIMessage(content="很抱歉，處理您的請求時遇到了技術問題。請稍後再試。")

    #==================================
    # 圖片顯示區塊 - 教學用簡化版本
    #==================================
    if processed_images:
        output_files_list = []
        for tool_msg in recent_tool_messages:
            if hasattr(tool_msg, 'name') and tool_msg.name == "generate_gemini_image":
                tool_content = str(tool_msg.content)
                
                if "'generated_files':" in tool_content:
                    try:
                        filename_start = tool_content.find("'filename': '") + 13
                        filename_end = tool_content.find("'", filename_start)
                        filename = tool_content[filename_start:filename_end]
                        
                        path_start = tool_content.find("'path': '") + 9
                        path_end = tool_content.find("'", path_start)
                        file_path = tool_content[path_start:path_end].replace('\\\\', '\\')
                        
                        type_start = tool_content.find("'type': '") + 9
                        type_end = tool_content.find("'", type_start)
                        file_type = tool_content[type_start:type_end]
                        
                        if os.path.exists(file_path):
                            processed_file_info = {
                                "filename": os.path.basename(filename),
                                "path": file_path,
                                "type": file_type
                            }
                            
                            try:
                                with open(file_path, "rb") as image_file:
                                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                                processed_file_info["base64_data"] = f"data:{file_type};base64,{encoded_string}"
                            except Exception:
                                continue
                            
                            output_files_list.append(processed_file_info)
                    except Exception:
                        pass
        
        if output_files_list:
            #==================================
            # 記憶保存區塊
            #==================================
            if messages and len(messages) >= 3:
                response_content = str(response.content).lower()
                important_keywords = ["我喜歡", "我偏好", "我習慣", "重要的是"]
                if any(keyword in response_content for keyword in important_keywords):
                    save_to_long_term_memory(store, user_id, f"使用者互動記錄: {str(response.content)[:100]}", "interaction")
            
            return {"messages": [response], "generated_files": output_files_list}
    
    #==================================
    # 記憶保存區塊
    #==================================
    if messages and len(messages) >= 3:
        response_content = str(response.content).lower()
        important_keywords = ["我喜歡", "我偏好", "我習慣", "重要的是"]
        if any(keyword in response_content for keyword in important_keywords):
            save_to_long_term_memory(store, user_id, f"使用者互動記錄: {str(response.content)[:100]}", "interaction")
    
    return {"messages": [response]}

def task_agent(state: AgentState, config: RunnableConfig):
    """
    任務執行代理 - 工具調用執行專家
    
    職責：
    1. 接收執行指令
    2. 調用相應工具
    3. 返回執行結果
    """
    #==================================
    # 基本配置初始化區塊
    #==================================
    configuration = config.get("configurable", {})
    config_schema = ConfigSchema(**configuration)
    tools, tool_mapping = load_tools_from_config(config_schema)
    model = create_task_agent_llm(config_schema)
    task_prompt = config_schema.task_agent_prompt
    messages = state["messages"]
    
    #==================================
    # 圖片數據提取區塊 - 從原始訊息中提取
    #==================================
    image_base64 = None
    user_messages = [msg for msg in messages if msg.type == "human"]
    if user_messages:
        # 尋找包含圖片的原始訊息
        for msg in reversed(user_messages):
            original_query = msg.content
            image_base64 = extract_image_data(original_query)
            if image_base64:
                break
    
    #==================================
    # 指令解析區塊
    #==================================
    custom_prompt = None
    task_instruction = None
    
    for msg in reversed(messages):
        if msg.type == "ai" and "需要工具協助" in str(msg.content):
            instruction = str(msg.content)
            
            if "執行步驟：" in instruction:
                task_instruction = extract_step_instruction(instruction)
                if task_instruction:
                    custom_prompt = task_instruction['description']
            elif "請使用" in instruction and "具體指令：" in instruction:
                try:
                    tool_part = instruction.split("請使用")[1].split("處理檔案")[0].strip()
                    tool_name = tool_part
                    prompt_part = instruction.split("具體指令：")[1].strip()
                    custom_prompt = prompt_part
                    task_instruction = {'tool': tool_name, 'description': custom_prompt}
                except Exception:
                    pass
            break
    
    #==================================
    # 任務執行區塊
    #==================================
    # 使用清理後的用戶訊息內容進行工具調用，但保留base64傳遞給工具
    cleaned_user_content = None
    if user_messages:
        cleaned_user_content = clean_message_content(user_messages[-1].content)
    
    return execute_single_task_step(
        model, tools, tool_mapping, task_prompt, task_instruction, 
        custom_prompt, cleaned_user_content, 
        image_base64, messages
    )

def execute_single_task_step(model, tools: list, tool_mapping: dict, task_prompt: str, task_instruction: dict = None, 
                            custom_prompt: str = None, original_query = None, 
                            image_base64: str = None, original_messages: list = None):
    """執行單一任務步驟"""
    #==================================
    # 訊息構建區塊 - 使用清理後的內容
    #==================================
    task_messages = [SystemMessage(content=task_prompt)]

    if custom_prompt:
        if image_base64:
            instruction_content = f"執行工具調用：使用指定工具分析圖片，提示詞：{custom_prompt}"
        else:
            instruction_content = f"執行工具調用：使用指定工具，提示詞：{custom_prompt}"
        task_messages.append(HumanMessage(content=instruction_content))
    elif original_messages:
        # 優先使用最新的 AI 指令 (來自 result_agent 的計劃)
        last_ai_message = None
        for msg in reversed(original_messages):
            if msg.type == 'ai':
                last_ai_message = msg
                break
        
        if last_ai_message and hasattr(last_ai_message, 'content'):
            # 將 AI 的計劃作為給 task_agent 的主要指令
            task_messages.append(HumanMessage(content=str(last_ai_message.content)))
        elif original_query:
            # 如果沒有 AI 指令，則退回使用原始查詢
            task_messages.append(HumanMessage(content=str(original_query)))
    elif original_query:
        # 使用清理後的查詢內容
        task_messages.append(HumanMessage(content=str(original_query)))

    # 調用模型
    tool_to_call_name = task_instruction.get("tool") if task_instruction else None
    target_tool = None
    if tool_to_call_name:
        target_tool = next((t for t in tools if t.name == tool_to_call_name), None)

    if target_tool:
        # 強制模型調用指定的工具
        model_with_tool = model.bind_tools([target_tool])
        response = model_with_tool.invoke(task_messages)
    else:
        # 如果沒有指定或找到工具，則讓模型自行決定
        response = model.invoke(task_messages)

    #==================================
    # 工具調用處理區塊
    #==================================
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_results = []
        # 使用傳入的 tool_mapping，如果沒有則創建
        if tool_mapping:
            tools_dict = tool_mapping
        else:
            tools_dict = {tool.name: tool for tool in tools}
        
        for tool_call in response.tool_calls:
            tool_call_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]
            
            # 應用自定義提示詞
            if custom_prompt:
                if 'prompt' in tool_args:
                    tool_args['prompt'] = custom_prompt
                elif 'query' in tool_args:
                    tool_args['query'] = custom_prompt
            
            # 傳遞圖片數據（重要：保持原有成功邏輯）
            if image_base64:
                if tool_call_name == 'analyze_image':
                    tool_args['image_input'] = image_base64
                elif tool_call_name == 'analyze_multimodal_content':
                    tool_args['image_data'] = image_base64
                elif tool_call_name == 'generate_gemini_image':
                    # 對於圖片生成工具，可能需要圖片作為參考
                    tool_args['reference_image'] = image_base64
            
            # 執行工具
            if tool_call_name in tools_dict:
                try:
                    tool_result = tools_dict[tool_call_name].invoke(tool_args)
                except Exception as e:
                    tool_result = f"工具執行錯誤: {str(e)}"
            else:
                tool_result = f"未找到工具: {tool_call_name}"
            
            tool_results.append(
                ToolMessage(
                    content=str(tool_result),
                    name=tool_call_name,
                    tool_call_id=tool_call_id,
                )
            )
        
        # 只返回工具結果，不包含空的 AI 回應
        return {"messages": tool_results}
    else:
        # 如果沒有工具調用，返回原始回應
        return {"messages": [response]}

#==================================
# 條件邏輯定義區塊
#==================================
def should_use_tools(state: AgentState):
    """判斷是否需要工具協助"""
    messages = state["messages"]
    
    if not messages:
        return "end"
    
    last_message = messages[-1]
    last_content = str(last_message.content) if hasattr(last_message, 'content') else ""
    
    # 檢查是否已經有工具結果存在但還有新的工具請求
    recent_tool_messages = [msg for msg in messages[-10:] if msg.type == "tool"]
    if recent_tool_messages:
        # 檢查最新的 AI 訊息是否要求執行新的工具
        if last_message.type == "ai" and "需要工具協助" in last_content:
            # 檢查是否是不同的工具或新的步驟
            if any(keyword in last_content for keyword in ["執行步驟：", "當前步驟：", "下一步"]):
                return "use_tools"
        
        return "end"
    
    if last_message.type == "ai" and "需要工具協助" in last_content:
        if any(keyword in last_content for keyword in ["執行任務計劃", "執行步驟：", "請使用"]):
            return "use_tools"
    
    return "end"

def should_continue_or_integrate(state: AgentState):
    """判斷是否需要繼續執行計劃步驟或整合結果"""
    messages = state["messages"]
    
    if not messages:
        return "integrate"
    
    recent_messages = messages[-10:] if len(messages) >= 10 else messages
    
    # 檢查各種狀態
    has_tool_results = any(msg.type == "tool" for msg in recent_messages)
    has_plan_messages = any(
        msg.type == "ai" and ("任務計劃" in str(msg.content) or "執行步驟" in str(msg.content))
        for msg in recent_messages
    )
    has_step_instruction = any(
        msg.type == "ai" and "執行步驟：" in str(msg.content)
        for msg in recent_messages
    )
    
    if has_tool_results:
        tool_messages = [msg for msg in recent_messages if msg.type == "tool"]
        
        if tool_messages:
            latest_tool_result = str(tool_messages[-1].content)
            latest_tool_name = getattr(tool_messages[-1], 'name', '')
            
            # 簡化判斷邏輯：只要有工具訊息存在，就認為工具執行成功
            # 因為只有成功執行的工具才會產生 ToolMessage
            if latest_tool_name and len(latest_tool_result) > 10:
                return "integrate"
            else:
                return "continue_plan"
    
    # 如果有計劃但沒有工具結果，繼續執行
    if (has_plan_messages or has_step_instruction) and not has_tool_results:
        return "continue_plan"
    
    return "integrate"

#==================================
# 圖表建構區塊
#==================================
# 創建記憶組件
checkpointer = create_checkpointer()
store = create_store()

# 創建圖表
workflow = StateGraph(AgentState, config_schema=ConfigSchema)

# 添加節點
workflow.add_node("result_agent", result_agent)
workflow.add_node("task_agent", task_agent)

# 設置入口點
workflow.set_entry_point("result_agent")

# 添加條件邊
workflow.add_conditional_edges(
    "result_agent",
    should_use_tools,
    {
        "use_tools": "task_agent",
        "end": END,
    },
)

workflow.add_conditional_edges(
    "task_agent",
    should_continue_or_integrate,
    {
        "continue_plan": "result_agent",
        "integrate": "result_agent",
    },
)

# 編譯圖表
graph = workflow.compile(
    checkpointer=checkpointer,
    store=store
)
graph.name = "智能多代理任務規劃系統"


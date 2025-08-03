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
from src.configuration import ConfigSchema, create_llm_agent, create_tool_agent, get_task_agent_prompt, get_result_agent_prompt
from src.memory import create_checkpointer, create_store, trim_message_history, save_to_long_term_memory, search_long_term_memory

#==================================
# 工具載入區塊
#==================================
def load_tools():
    """載入所有可用的工具"""
    tools = []
    
    try:
        from src.tools.gemini_search_tool import perform_grounded_search
        tools.append(perform_grounded_search)
    except Exception:
        pass
    
    try:
        from src.tools.gemini_image_generation_tool import generate_gemini_image
        tools.append(generate_gemini_image)
    except Exception:
        pass
    
    try:
        from src.tools.multimodal_input_tool import (
            analyze_multimodal_content, analyze_image, analyze_video, analyze_document
        )
        tools.extend([analyze_multimodal_content, analyze_image, analyze_video, analyze_document])
    except Exception:
        pass
    
    return tools

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
    start_time = time.time()
    
    #==================================
    # 基本配置初始化區塊
    #==================================
    configuration = config.get("configurable", {})
    config_schema = ConfigSchema(**configuration)
    user_id = configuration.get("user_id", "workshop_user")
    
    # 創建 LLM 代理
    model = create_llm_agent(config_schema)
    result_prompt = get_result_agent_prompt(config_schema)
    messages = state["messages"]
    
    #==================================
    # 訊息處理和記憶搜尋區塊
    #==================================
    # 訊息管理
    trimmed_messages = trim_message_history(messages, max_tokens=2000)
    
    # 記憶搜尋（僅在需要時）
    memory_context = ""
    if messages and len(messages) > 1:
        last_user_message = None
        for msg in reversed(messages):
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
    # 構建訊息序列
    result_messages = [SystemMessage(content=result_prompt)]
    if trimmed_messages:
        result_messages.extend(trimmed_messages)

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
    if last_message and last_message.type not in ["human", "tool"]:
        tool_messages = [msg for msg in validated_messages if msg.type == "tool"]
        if tool_messages:
            validated_messages.append(
                HumanMessage(content="請基於上述工具分析結果，提供完整的回答。")
            )
        else:
            validated_messages.append(
                HumanMessage(content="請協助處理這個請求。")
            )

    # 調用 LLM
    try:
        response = model.invoke(validated_messages)
        
        # 檢查回應內容
        if hasattr(response, 'content'):
            content = response.content
            if not content or (isinstance(content, str) and not content.strip()):
                tool_messages = [msg for msg in validated_messages if msg.type == "tool"]
                if tool_messages:
                    latest_tool_result = str(tool_messages[-1].content)
                    if len(latest_tool_result) > 100:
                        summary_prompt = f"基於詳細的分析結果，我可以為您總結以下要點：\n\n{latest_tool_result[:500]}...\n\n如需了解更多細節或有具體問題，請隨時告訴我。"
                    else:
                        summary_prompt = f"根據分析結果：\n\n{latest_tool_result}\n\n這就是我對您問題的回答。還有其他問題嗎？"
                    response = AIMessage(content=summary_prompt)
                else:
                    response = AIMessage(content="我已完成分析。如果您有特定問題，請告訴我。")
        else:
            response = AIMessage(content="已收到您的請求，但回應格式異常。請重新嘗試。")
            
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
    tools = load_tools()
    model = create_tool_agent(config_schema, tools)
    task_prompt = get_task_agent_prompt(config_schema)
    messages = state["messages"]
    
    #==================================
    # 圖片數據提取區塊
    #==================================
    image_base64 = None
    user_messages = [msg for msg in messages if msg.type == "human"]
    if user_messages:
        original_query = user_messages[-1].content
        image_base64 = extract_image_data(original_query)
    
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
    return execute_single_task_step(
        model, tools, task_prompt, task_instruction, 
        custom_prompt, user_messages[-1].content if user_messages else None, 
        image_base64, messages
    )

def execute_single_task_step(model, tools: list, task_prompt: str, task_instruction: dict = None, 
                            custom_prompt: str = None, original_query = None, 
                            image_base64: str = None, original_messages: list = None):
    """執行單一任務步驟"""
    #==================================
    # 訊息構建區塊
    #==================================
    task_messages = [SystemMessage(content=task_prompt)]

    if custom_prompt:
        if image_base64:
            instruction_content = f"執行工具調用：使用指定工具分析圖片，提示詞：{custom_prompt}"
        else:
            instruction_content = f"執行工具調用：使用指定工具，提示詞：{custom_prompt}"
        task_messages.append(HumanMessage(content=instruction_content))
    elif original_query:
        task_messages.append(HumanMessage(content=str(original_query)))
    elif original_messages:
        task_messages.append(original_messages[-1])

    # 調用模型
    response = model.invoke(task_messages)

    #==================================
    # 工具調用處理區塊
    #==================================
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_results = []
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
        
        return {"messages": [response] + tool_results}
    else:
        return {"messages": [response]}

#==================================
# 條件邊邏輯定義區塊
#==================================
def should_use_tools(state: AgentState):
    """判斷是否需要工具協助"""
    messages = state["messages"]
    
    if not messages:
        return "end"
    
    last_message = messages[-1]
    last_content = str(last_message.content) if hasattr(last_message, 'content') else ""
    
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
            if len(latest_tool_result) > 50:
                return "continue_plan"
            else:
                return "integrate"
    
    if has_plan_messages and not has_tool_results:
        return "continue_plan"
    
    if has_step_instruction and not has_tool_results:
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


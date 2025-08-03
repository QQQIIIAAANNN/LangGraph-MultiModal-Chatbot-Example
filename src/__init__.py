# """
# LangGraph 工作坊教學範本 - 雙代理記憶系統（簡化版）
# 專為初學者設計的LangGraph學習範例，使用官方模組的基本記憶功能
# """

# from .state import AgentState, create_empty_state
# from .configuration import ConfigSchema, get_llm, get_task_agent_prompt, get_result_agent_prompt, create_llm_agent, create_tool_agent
# from .graph import create_graph, run_agent, run_agent_with_memory, load_tools, task_agent, result_agent
# from .memory import create_checkpointer, create_store, trim_message_history, save_to_long_term_memory, search_long_term_memory, get_thread_config

# __version__ = "3.1.0"
# __author__ = "Workshop Team"
# __description__ = "LangGraph工作坊教學範本 - 雙代理記憶系統（簡化版）"

# # 主要的匯出功能
# __all__ = [
#     # 狀態管理
#     "AgentState",
#     "create_empty_state",
    
#     # 配置管理
#     "ConfigSchema",
#     "get_llm",
#     "get_task_agent_prompt",
#     "get_result_agent_prompt",
#     "create_llm_agent",
#     "create_tool_agent",
    
#     # 圖表和代理程式
#     "create_graph",
#     "run_agent",
#     "run_agent_with_memory",
#     "load_tools",
    
#     # # 雙代理節點
#     "task_agent",
#     "result_agent",
    
#     # 記憶功能（簡化版）
#     "create_checkpointer",
#     "create_store",
#     "trim_message_history",
#     "save_to_long_term_memory",
#     "search_long_term_memory",
#     "get_thread_config",
# ] 
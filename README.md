# 智能多代理任務規劃系統

🔹中文描述（for 初學者）：
本專案為一場面向初學者的 LangGraph 工作坊所設計的教學範例，目標是透過 LangGraph Studio，帶領使用者理解多代理系統（Multi-Agent System）的基本架構與流程設計方式。本系統以「建築概念生成對話機器人」為主題，串接 Google Gemini API，實現文本輸入、圖像生成、資料查詢等任務，展示如何實現動態任務規劃和執行管理，適合做為設計導向 AI 協作流程的入門範例。

🔹English Description (for Beginners):
This project is a beginner-friendly tutorial developed for a LangGraph workshop, aiming to help users understand the foundational architecture and workflow logic of multi-agent systems using LangGraph Studio.The system centers on an AI-powered chatbot for architectural concept generation, integrated with the Google Gemini API to perform text input, image generation, and knowledge retrieval tasks.It demonstrates how to implement dynamic task planning and execution management, making it an ideal entry-level example for design-oriented, AI-assisted collaborative workflows.
If you need english version, please issues to me.

## 🎯 主要功能概述

這是一個完整的多模態多代理系統示範，透過 [LangGraph Studio](https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/) 啟動並運行，展示了智能代理系統的五大核心模組：

- 📁 **建檔模組** - 智能文檔處理和資料管理
- 🧠 **記憶模組** - 短期記憶（對話狀態）+ 長期記憶（語義搜尋）
- 📋 **規劃模組** - 動態任務分解和計劃制定
- ⚡ **行動模組** - 工具調用和任務執行
- 👁️ **感知模組** - 多模態內容理解（圖片、文字、語音）

## 📝 特點:
### 1. 智能分工
- **規劃與執行分離**：結果代理負責思考，任務代理負責行動
- **動態決策**：每步執行後重新評估下一步行動
- **錯誤恢復**：可以調整計劃或重試失敗的步驟

### 2. 模組化設計
- **易於擴展**：可輕鬆添加新的工具和功能
- **配置靈活**：支援 UI 動態調整參數
- **記憶整合**：自動管理短期和長期記憶

### 3. 學習友善
- **清晰結構**：代碼組織清楚，易於理解
- **豐富註釋**：詳細解釋關鍵概念和實作細節
- **實際範例**：展示真實的應用場景

## 🏗️ 系統架構

### 雙代理設計

```
用戶請求 
    ↓
[結果代理] - 任務規劃專家
    ↓ (制定計劃)
[任務代理] - 工具執行專家
    ↓ (執行工具)
[結果代理] - 檢查結果，決定下一步
    ↓ (繼續/完成)
最終回答
```

### 核心組件

1. **Result Agent (結果代理)**
   - 分析用戶請求
   - 制定執行計劃
   - 管理執行進度
   - 整合最終結果

2. **Task Agent (任務代理)**
   - 執行工具調用
   - 處理多模態輸入
   - 返回執行結果

3. **條件邊 (Conditional Edges)**
   - `should_use_tools`：判斷是否需要工具協助
   - `should_continue_or_integrate`：決定繼續執行或整合結果

## 🔧 可用工具

- `analyze_image`：分析圖片內容
- `perform_grounded_search`：網路搜尋最新資訊
- `generate_gemini_image`：生成圖片
- `analyze_multimodal_content`：處理多媒體內容

## 📝 工作流程範例

### 1. 簡單請求
```
用戶：上傳圖片並寫「分析這張圖片」
↓
結果代理：直接調用 analyze_image
↓
任務代理：執行圖片分析
↓
結果代理：整合結果並回答
```

### 2. 複雜請求
```
用戶：上傳圖片並寫「分析這張圖片並搜尋相關資訊」
↓
結果代理：制定兩步驟計劃
  1. 分析圖片
  2. 搜尋相關資訊
↓
任務代理：執行步驟1 (分析圖片)
↓
結果代理：檢查結果，決定執行步驟2
↓
任務代理：執行步驟2 (搜尋資訊)
↓
結果代理：整合所有結果並提供完整回答
```

## 🚀 安裝和設置

### 前置需求

- Python 3.11 或更高版本
- [Gemini API 金鑰](https://aistudio.google.com/)（免費註冊）
- [LangSmith API 金鑰](https://smith.langchain.com/)（免費註冊）

### 1. 克隆專案

```bash
git clone https://github.com/QQQIIIAAANNN/LangGraph-MultiModal-Chatbot-Example.git
cd LangGraph-MultiModal-Chatbot-Example
```
### 2. 創建虛擬環境

```bash
# 創建名為 langgraph-env 的虛擬環境
python -m venv langgraph-env

# 啟動虛擬環境
# Windows:
langgraph-env\Scripts\activate
# macOS/Linux:
source langgraph-env/bin/activate
```

### 3. 安裝專案依賴

```bash
# 以開發模式安裝，讓本地變更即時生效
pip install -r requirements.txt
```

### 4. 設定環境變數

```bash
# 複製環境變數範例
cp .env.example .env

# 編輯 .env 檔案，加入您的 API 金鑰
```

## 🖥️ 透過 LangGraph Studio 運行

### 1. 啟動本地服務器

```bash
# 啟動 LangGraph 開發服務器
langgraph dev

# 如果使用 Safari 瀏覽器，請加上 --tunnel 參數
langgraph dev --tunnel
```
### 2. 或使用批次檔(run.bat)運行

## 📁 檔案結構

```
LangGraph-MultiModal-Chatbot-Example/
├── src/
│ ├── init.py
│ ├── state.py # 狀態管理（記憶+感知模組）
│ ├── configuration.py # 配置管理(建檔+規劃模組)
│ ├── graph.py # 核心圖表邏輯（規劃+行動模組）
│ ├── memory.py # 記憶系統實作（記憶模組）
│ └── tools/ # 工具集（感知+行動模組）
│         ├── init.py
│         ├── gemini_search_tool.py # 搜尋工具
│         ├── gemini_image_generation_tool.py # 圖片生成
│         └── multimodal_input_tool.py # 多模態分析（感知模組）
├── langgraph.json # LangGraph 配置
├── pyproject.toml # 專案配置
├── requirements.txt # 依賴清單
├── .env.example # 環境變數範例
├── run.bat # Windows 快速啟動腳本
└── README.md # 專案說明
```

## 🎓 學習重點

### 對初學者
1. **多代理協作**：理解規劃代理和執行代理這類基礎多代理的分工
2. **狀態管理**：學習如何在複雜對話中維護上下文
3. **模組化設計**：了解五大模組的職責和互動

### 對進階學習者
1. **LangGraph 架構**：深入理解圖表建構和條件邊
2. **記憶系統**：實作短期和長期記憶的整合
3. **多模態處理**：處理不同類型輸入及輸出的策略

## 🔗 相關資源

- 📚 [LangGraph 官方文檔](https://langchain-ai.github.io/langgraph/)
- 🏠 [LangGraph Studio 指南](https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/)
- 🤖 [Google Gemini API](https://ai.google.dev/gemini-api)
- 🔬 [LangSmith Platform](https://smith.langchain.com/)

## 🤝 貢獻指南

歡迎提交 Issues 和 Pull Requests！

1. Fork 此專案
2. 創建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交變更 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 開啟 Pull Request

## 📄 授權

此專案採用 [MIT License](LICENSE) 授權。

---

**⭐ 如果這個專案對您有幫助，請給個 Star！**

> 開發者：NCKU IA Lab QQQIIIAAANNN| 適合：LangGraph 初學者到進階用戶

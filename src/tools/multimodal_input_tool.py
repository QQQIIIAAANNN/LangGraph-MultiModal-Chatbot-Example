"""
多模態輸入工具 - 支援圖片、影片、文檔理解（初學者友善版本）
參考官方文檔：https://ai.google.dev/gemini-api/docs/image-understanding?hl=zh-tw

主要功能：
1. 直接處理 base64 格式的圖片（無需文件路徑）
2. 支援文件路徑輸入（向後兼容）
3. 清晰的錯誤處理和日誌輸出
"""

import os
import base64
import mimetypes
from typing import List, Dict, Any, Optional, Union
from langchain.tools import tool
import google.generativeai as genai
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()


def prepare_image_content(image_input: Union[str, bytes], text: str = "") -> List[Any]:
    """
    準備圖片內容給 Gemini API（支援 base64 和文件路徑）
    
    Args:
        image_input: 圖片輸入（base64 字符串或文件路徑）
        text: 文字提示
        
    Returns:
        準備好給 Gemini 的內容列表
    """
    print("🔧 開始準備圖片內容...")
    print(f"📋 輸入類型：{type(image_input)}")
    print(f"📋 輸入長度：{len(str(image_input))}")
    print(f"📋 輸入預覽：{str(image_input)[:100]}...")
    
    # 基礎內容列表，從文字開始
    contents = [text] if text else ["請分析這張圖片"]
    
    try:
        # 情況1：base64 格式的圖片（來自網頁上傳）
        if isinstance(image_input, str) and image_input.startswith("data:image/"):
            print("📸 檢測到 base64 格式圖片")
            
            # 解析 data URL: data:image/jpeg;base64,/9j/4AAQ...
            header, data = image_input.split(",", 1)
            mime_type = header.split(";")[0].replace("data:", "")
            
            # 解碼 base64 數據
            image_bytes = base64.b64decode(data)
            
            # 按照官方文檔格式添加圖片
            contents.append({
                "mime_type": mime_type,
                "data": image_bytes
            })
            
            print(f"✅ 成功載入 {mime_type} 圖片 (大小: {len(image_bytes)} bytes)")
            return contents
            
        # 情況2：文件路徑（向後兼容和新增支援）
        elif isinstance(image_input, str) and os.path.exists(image_input):
            print(f"📁 檢測到文件路徑: {image_input}")
            
            # 猜測文件類型
            mime_type, _ = mimetypes.guess_type(image_input)
            if not mime_type or not mime_type.startswith('image/'):
                print(f"⚠️ 不支援的圖片格式: {mime_type}")
                return contents
            
            # 讀取文件
            with open(image_input, "rb") as f:
                image_bytes = f.read()
            
            # 添加圖片內容
            contents.append({
                "mime_type": mime_type,
                "data": image_bytes
            })
            
            print(f"✅ 成功載入文件 {os.path.basename(image_input)} ({mime_type})")
            return contents
            
        # 情況3：空輸入或無效輸入
        else:
            if not image_input or (isinstance(image_input, str) and image_input.strip() == ""):
                print("⚠️ 圖片輸入為空")
            else:
                print(f"⚠️ 無法識別圖片輸入格式: {image_input}")
            return contents
            
    except Exception as e:
        print(f"❌ 處理圖片時發生錯誤: {e}")
        return contents


@tool("analyze_image")
def analyze_image(
    image_input: str = "", 
    prompt: str = "請分析這張圖片",
    model_name: str = "gemini-2.5-pro"
) -> str:
    """
    分析圖片內容（支援 base64 和文件路徑輸入）
    
    這是一個初學者友善的圖片分析工具，可以：
    1. 直接處理網頁上傳的 base64 圖片
    2. 處理本地文件路徑
    3. 提供清晰的錯誤信息
    
    Args:
        image_input: 圖片輸入（base64 字符串或文件路徑）
        prompt: 分析提示詞
        model_name: 使用的 Gemini 模型
        
    Returns:
        圖片分析結果
    """
    print(f"🎯 開始圖片分析，提示：{prompt[:50]}...")
    print(f"📋 image_input 類型：{type(image_input)}")
    print(f"📋 image_input 長度：{len(str(image_input))} 字符")
    print(f"📋 image_input 預覽：{str(image_input)[:100]}...")
    
    try:
        # 檢查輸入是否為空
        if not image_input or image_input.strip() == "":
            print("❌ 圖片輸入為空")
            return "⚠️ 沒有提供圖片輸入，請確認圖片已正確上傳"
        
        # 檢查 API 金鑰
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ 找不到 API 金鑰")
            return "❌ 錯誤：找不到 GEMINI_API_KEY 環境變數"
        
        # 配置 Gemini API
        genai.configure(api_key=api_key)
        print("✅ Gemini API 已配置")
        
        # 準備圖片內容
        print("🔧 準備圖片內容...")
        contents = prepare_image_content(image_input, prompt)
        
        # 檢查是否成功載入圖片
        if len(contents) == 1:
            print("❌ 沒有成功載入圖片內容")
            return "⚠️ 沒有檢測到有效的圖片輸入，請確認圖片格式正確"
        
        print(f"✅ 成功準備 {len(contents)} 個內容項目")
        
        # 創建 Gemini 模型
        model = genai.GenerativeModel(model_name)
        print(f"✅ 已創建 {model_name} 模型")
        
        # 生成分析結果
        print("🤖 正在呼叫 Gemini API...")
        response = model.generate_content(contents)
        
        if response.text:
            print("✅ 圖片分析完成")
            print(f"📊 回應長度：{len(response.text)} 字符")
            return response.text
        else:
            print("❌ Gemini 沒有返回文字回應")
            return "❌ Gemini 沒有返回分析結果"
            
    except Exception as e:
        error_msg = f"圖片分析失敗: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        print(f"🔍 詳細錯誤：{traceback.format_exc()}")
        return error_msg


@tool("analyze_multimodal_content")
def analyze_multimodal_content(
    query: str, 
    file_paths: Optional[str] = None,
    image_data: Optional[str] = None,
    model_name: str = "gemini-2.5-pro"
) -> Dict[str, Any]:
    """
    分析多模態內容（圖片、影片、文檔）的通用工具
    
    支援多種輸入方式：
    1. file_paths: 文件路徑（向後兼容）
    2. image_data: base64 圖片數據（新功能）
    
    Args:
        query: 要詢問的問題
        file_paths: 檔案路徑（多個路徑用逗號分隔）
        image_data: base64 格式的圖片數據
        model_name: 使用的模型名稱
        
    Returns:
        分析結果字典
    """
    print(f"🔍 多模態分析：{query[:50]}...")
    print(f"📋 輸入參數：file_paths={bool(file_paths)}, image_data={bool(image_data)}")
    
    try:
        # 優先處理 base64 圖片數據
        if image_data:
            print("📸 使用 base64 圖片數據進行分析")
            result = analyze_image(image_data, query, model_name)
            return {
                "response": result,
                "files_processed": 1,
                "model_used": model_name,
                "input_type": "base64_image"
            }
        
        # 處理文件路徑（向後兼容）
        if file_paths:
            print(f"📁 使用文件路徑進行分析：{file_paths}")
            # 簡單處理：將查詢轉發給圖片分析工具
            if isinstance(file_paths, str) and (
                file_paths.startswith("data:image/") or 
                (os.path.exists(file_paths) and file_paths.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')))
            ):
                result = analyze_image(file_paths, query, model_name)
                return {
                    "response": result,
                    "files_processed": 1,
                    "model_used": model_name,
                    "input_type": "file_path"
                }
        
        # 如果沒有提供任何媒體輸入
        return {
            "response": f"📝 純文字查詢：{query}",
            "files_processed": 0,
            "model_used": model_name,
            "input_type": "text_only"
        }
        
    except Exception as e:
        print(f"❌ 多模態分析錯誤：{e}")
        return {"error": f"多模態分析失敗: {str(e)}"}


# 保留其他工具以維持向後兼容性
@tool("analyze_video")
def analyze_video(
    video_path: str,
    prompt: str = "請分析這段影片的內容",
    model_name: str = "gemini-2.5-pro"
) -> str:
    """影片分析工具（佔位符，待開發）"""
    return "影片分析功能正在開發中"


@tool("analyze_document")
def analyze_document(
    document_path: str,
    prompt: str = "請處理這份文檔的內容",
    model_name: str = "gemini-2.5-pro"
) -> str:
    """文檔分析工具（佔位符，待開發）"""
    return "文檔分析功能正在開發中" 
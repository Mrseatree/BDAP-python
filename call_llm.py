from config import CONSUL_HOST, CONSUL_PORT, SERVICE_NAME, SEARXNG_URL, SHARED_DIR
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from consul_utils import register_service, deregister_service
from enum import Enum
import httpx
from typing import Optional, Dict, Any, List
import consul
import time
import subprocess
import shutil
import socket
import glob
import uuid
import os
import atexit
import requests
import traceback
import json
import pandas as pd

# 模型枚举
class ModelName(str, Enum):
    silicon_flow = "silicon-flow"
    moonshot = "moonshot"
    deepseek = "deepseek"
    Qwen = "Qwen"

MODEL_TO_APIKEY = {
    "silicon-flow": "app-OFaVMpobX30c0Tv36i1luC2U", 
    "moonshot": "app-p9c2JEIrsJariPYeIxU3otjB",
    "deepseek": "app-zJT765lCyC0UkeNqRik4vYRw",
    "Qwen": "app-VH46JNigYuWdqf62sBucCOcw"
}

dify_url = "http://10.92.64.224/v1/chat-messages"

class ChatRequest(BaseModel):
    model: ModelName
    question: str
    requestId: str
    use_web_search: Optional[bool]
    user_id: Optional[str] # 默认为匿名用户
    conversation_id: Optional[str] = None # 默认为新对话


class ChatResponse(BaseModel):
    answer: str
    requestId: str
    conversation_id: Optional[str] = None # 默认为新对话

# 数据处理请求模型
class DataProcessRequest(BaseModel):
    model: ModelName  # 模型名称
    user_prompt: str  # 用户需求描述
    user_id: Optional[str] = "defaultid"  # 用户ID
    file_path1: str  # HDFS文件路径1
    file_content1: str  # 文件1的部分内容（JSON字符串，包含列名等）
    file_path2: Optional[str] = None  # HDFS文件路径2
    file_content2: Optional[str] = None  # 文件2的部分内容
    output_path: str  # HDFS输出路径

# 数据处理响应模型
class DataProcessResponse(BaseModel):
    status: str  # success 或 error
    message: str  # 处理结果描述
    output_path: Optional[str] = None  # 输出文件HDFS路径
    error_details: Optional[str] = None

app = FastAPI()

# 服务注册相关代码...
@app.on_event("startup")
async def startup_event():
    service_id = start_call_llm_service()
    if service_id:
        app.state.service_id = service_id
        print(f"call_llm服务已注册到Consul，服务ID: {service_id}")

@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(app.state, 'service_id'):
        deregister_service(app.state.service_id)

def start_call_llm_service():
    SERVICE_PORT = 8000
    tags = ['llm', 'ai', 'dify', 'data-processing']
    service_id = register_service(SERVICE_PORT, tags)
    return service_id

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": SERVICE_NAME}

def perform_web_search(query: str) -> str:
    try:
        # 使用正确的变量
        search_url = f"{SEARXNG_URL}/search"
        print(f"正在搜索: {query}")
        print(f"搜索URL: {search_url}")

        resp = requests.get(
            search_url,
            params={"q": query, "format": "json"},
            timeout=10  # 适当增加超时时间
        )

        print(f"搜索响应状态码: {resp.status_code}")

        # 检查HTTP状态码
        if resp.status_code != 200:
            return f"【联网搜索失败】：HTTP {resp.status_code} - {resp.text}\n"

        try:
            json_data = resp.json()
        except ValueError as e:
            return f"【联网搜索失败】：无法解析JSON响应 - {e}\n"

        results = json_data.get("results", [])
        if not results:
            return f"【联网搜索结果】：未找到相关信息\n"

        # 取前3个结果
        top_results = results[:3]
        formatted = "\n".join([
            f"{i + 1}. {r.get('title', '无标题')}\nURL: {r.get('url', '无URL')}\n摘要: {r.get('content', '无摘要')}"
            for i, r in enumerate(top_results)
        ])

        return f"【以下为联网搜索结果】：\n{formatted}\n"

    except requests.exceptions.ConnectionError as e:
        print(f"连接错误: {e}")
        return f"【联网搜索失败】：无法连接到搜索服务 ({SEARXNG_URL})\n"
    except requests.exceptions.Timeout as e:
        print(f"超时错误: {e}")
        return f"【联网搜索失败】：搜索服务响应超时\n"
    except Exception as e:
        print(f"未知错误: {e}")
        return f"【联网搜索失败】：{e}\n"

async def call_dify(model: str, prompt: str, user_id: str, conversation_id: Optional[str] = None) -> str:
    api_key = MODEL_TO_APIKEY.get(model)

    if not api_key:
        return f"[error]模型{model}未配置API KEY"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "inputs": {},
        "query": prompt,
        "response_mode": "blocking",
        "user": user_id,
        "conversation_id": conversation_id   # 若有上下文则填
    }

    timeout = httpx.Timeout(120.0, read=120.0, connect=10.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(dify_url, headers=headers, json=data)

            print("状态码:", resp.status_code)
            print("原始内容:", resp.text)

            if resp.status_code == 504:
                raise HTTPException(status_code=504, detail="[Dify错误]模型响应超时，稍后再试")

            try:
                result = resp.json()  # 只在成功时赋值
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"[响应格式错误]无法解析JSON:{e}\n原始响应:{resp.text}")

            if "answer" in result:
                return result["answer"], result.get("conversation_id")
            elif "message" in result:
                raise HTTPException(status_code=502, detail=f"[Dify错误] {result['message']}")
            else:
                raise HTTPException(status_code=502, detail="[Dify响应格式异常]")


    except httpx.ReadTimeout:
        raise HTTPException(status_code=504, detail="[超时] Dify 响应超时")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"[请求失败] {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[未知错误] {e}")


# 接口的返回值应当符合ChatResponse的Pydantic模型结构
@app.post("/llm", response_model=ChatResponse)
async def get_model(request: ChatRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="question 不能为空")

    # 原始问题
    question = request.question.strip()

    if request.use_web_search:
        try:
            web_result = perform_web_search(question).strip()
            if not web_result:
                raise ValueError("Empty web result")

            full_prompt = (
                f"你是一名知识渊博的智能助手。\n\n"
                f"以下是与用户问题相关的最新搜索信息：\n"
                f"{web_result}\n\n"
                f"请根据以上资料，结合用户的问题，进行精准和详尽的解答。\n\n"
                f"【用户问题】：{question}"
            )
        except Exception as e:
            # 联网失败，降级处理
            full_prompt = (
                "【提示】：联网搜索失败，以下为基于已有知识的回答。\n\n"
                f"【用户问题】：{question}"
            )
    else:
        # 不使用联网搜索时直接使用原问题
        full_prompt = question

    # 调用 Dify 接口
    answer, new_conversation_id = await call_dify(
        request.model,
        full_prompt,
        user_id=request.user_id,
        conversation_id=str(request.conversation_id) if request.conversation_id else None
    )

    return ChatResponse(
        answer=answer,
        requestId=request.requestId,
        conversation_id=new_conversation_id
    )

def download_hdfs_file(hdfs_path: str, local_dir: str) -> str:
    """
    从HDFS下载文件到本地目录
    返回本地文件路径
    """
    try:
        # 解析HDFS路径
        hdfs_prefix = "hdfs://bdap-cluster-01:8020"
        clean_hdfs_path = hdfs_path.replace(hdfs_prefix, "")
        if not clean_hdfs_path.startswith("/"):
            clean_hdfs_path = "/" + clean_hdfs_path
        
        # 生成本地文件路径
        filename = os.path.basename(clean_hdfs_path)
        short_uuid = str(uuid.uuid4())[:8]
        local_path = os.path.join(local_dir, f"{short_uuid}_{filename}")
        
        # 确保本地目录存在
        os.makedirs(local_dir, exist_ok=True)
        
        # 下载文件
        result = subprocess.run(
            ["hdfs", "dfs", "-get", clean_hdfs_path, local_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print(f"已下载HDFS文件: {hdfs_path} -> {local_path}")
        return local_path
        
    except subprocess.CalledProcessError as e:
        error_msg = f"HDFS文件下载失败: {e.stderr.decode() if e.stderr else str(e)}"
        raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"下载文件时发生错误: {str(e)}")

def upload_to_hdfs(local_path: str, hdfs_path: str, max_retries: int = 3) -> None:
    """
    将本地文件上传到HDFS
    """
    try:
        # 解析HDFS路径
        hdfs_prefix = "hdfs://bdap-cluster-01:8020"
        clean_hdfs_path = hdfs_path.replace(hdfs_prefix, "")
        if not clean_hdfs_path.startswith("/"):
            clean_hdfs_path = "/" + clean_hdfs_path
        
        # 确保父目录存在
        hdfs_parent = os.path.dirname(clean_hdfs_path)
        subprocess.run(
            ["hdfs", "dfs", "-mkdir", "-p", hdfs_parent],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 上传文件（带重试）
        for attempt in range(1, max_retries + 1):
            try:
                subprocess.run(
                    ["hdfs", "dfs", "-put", "-f", local_path, clean_hdfs_path],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                print(f"已上传到HDFS: {local_path} -> {hdfs_path}")
                return
            except subprocess.CalledProcessError as e:
                if attempt == max_retries:
                    raise e
                print(f"上传重试 {attempt}/{max_retries}")
                time.sleep(2)
                
    except subprocess.CalledProcessError as e:
        error_msg = f"HDFS文件上传失败: {e.stderr.decode() if e.stderr else str(e)}"
        raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传文件时发生错误: {str(e)}")

def cleanup_local_files(*file_paths: str) -> None:
    """清理本地临时文件"""
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"已删除本地文件: {file_path}")
            except Exception as e:
                print(f"删除文件失败 {file_path}: {e}")

async def call_dify_with_local_files(model: str, query: str, local_file_path1: str, 
                                   file_content1: str, local_file_path2: Optional[str] = None,
                                   file_content2: Optional[str] = None, 
                                   output_path: Optional[str] = None,
                                   user_id: str = "default_user") -> str:

    api_key = MODEL_TO_APIKEY.get(model)
    if not api_key:
        raise HTTPException(status_code=400, detail=f"模型{model}未配置API KEY")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 构建inputs，传递本地文件路径和内容给大模型
    inputs = {
        "file_path1": local_file_path1,
        "file_content1": file_content1,
    }
    
    if output_path:
        # 为输出文件生成本地路径
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        local_output_path = os.path.join(SHARED_DIR, f"output_{base_name}_{str(uuid.uuid4())[:8]}.csv")
        inputs["output_path"] = local_output_path
    
    # 如果有第二个文件
    if local_file_path2 and file_content2:
        inputs["file_path2"] = local_file_path2
        inputs["file_content2"] = file_content2

    data = {
        "inputs": inputs,
        "query": query,
        "response_mode": "blocking",
        "user": user_id
    }

    print("=== 发送到Dify的数据 ===")
    print(f"- 文件1: {local_file_path1}")
    print(f"- 文件2: {local_file_path2 or 'None'}")
    print(f"- 输出路径: {inputs.get('output_path', 'Auto-generated')}")
    print(f"- query: {query}")
    print("========================")

    timeout = httpx.Timeout(120.0, read=120.0, connect=10.0)

    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            resp = await client.post(dify_url, headers=headers, json=data)

            print("状态码:", resp.status_code)
            print("Dify响应:", resp.text[:500] + "..." if len(resp.text) > 500 else resp.text)

            if resp.status_code == 504:
                raise HTTPException(status_code=504, detail="Dify响应超时")

            try:
                result = resp.json()
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"响应解析失败: {e}")

            if "answer" in result:
                # 从answer中提取输出文件路径
                answer = result["answer"]
                
                # 查找实际生成的输出文件
                expected_output = inputs.get("output_path")
                if expected_output and os.path.exists(expected_output):
                    return expected_output, answer
                else:
                    # 如果预期路径不存在，尝试从SHARED_DIR中找到最新的输出文件
                    output_files = glob.glob(os.path.join(SHARED_DIR, "*_output_*.csv"))
                    if output_files:
                        # 返回最新创建的文件
                        latest_file = max(output_files, key=os.path.getctime)
                        return latest_file, answer
                    else:
                        raise HTTPException(status_code=500, detail="工具函数未生成预期的输出文件")
            else:
                error_msg = result.get("message", "Dify响应格式异常")
                raise HTTPException(status_code=502, detail=f"Dify错误: {error_msg}")

    except httpx.ReadTimeout:
        raise HTTPException(status_code=504, detail="Dify响应超时")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"请求失败: {e}")
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"调用Dify失败: {repr(e)}\n{tb}")

@app.post("/data-process/execute", response_model=DataProcessResponse)
async def execute_data_process(request: DataProcessRequest) -> DataProcessResponse:

    local_file1_path = None
    local_file2_path = None
    local_output_path = None
    
    try:
        # 1. 下载主文件
        print(f"开始下载主文件: {request.file_path1}")
        local_file1_path = download_hdfs_file(request.file_path1, SHARED_DIR)
        
        # 2. 下载副文件
        local_file2_path = None
        if request.file_path2:
            print(f"开始下载副文件: {request.file_path2}")
            local_file2_path = download_hdfs_file(request.file_path2, SHARED_DIR)
        
        # 3. 调用大模型进行数据处理
        print("调用大模型进行数据处理...")
        local_output_path, answer = await call_dify_with_local_files(
            model=request.model,
            query=request.user_prompt,
            local_file_path1=local_file1_path,
            file_content1=request.file_content1,
            local_file_path2=local_file2_path,
            file_content2=request.file_content2,
            output_path=request.output_path,
            user_id=request.user_id
        )
        
        # 4. 上传结果到HDFS
        print(f"上传结果到HDFS: {request.output_path}")
        upload_to_hdfs(local_output_path, request.output_path)
        
        return DataProcessResponse(
            status="success",
            message=f"数据处理完成: {answer}",
            output_path=request.output_path
        )
        
    except Exception as e:
        tb = traceback.format_exc()
        return DataProcessResponse(
            status="error",
            message="数据处理失败",
            error_details=f"{repr(e)}\n{tb}"
        )
    finally:
        # 5. 清理本地文件
        print("清理临时文件...")
        cleanup_local_files(local_file1_path, local_file2_path, local_output_path)

if __name__ == "__main__":
    import uvicorn
    
    # 确保共享目录存在
    os.makedirs(SHARED_DIR, exist_ok=True)
    
    service_id = register_service()
    if service_id:
        atexit.register(deregister_service, service_id)

    SERVICE_PORT = 8000
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)

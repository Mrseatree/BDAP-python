from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
import httpx
from typing import Optional, Dict, List
import asyncio
import threading
from queue import Queue
import time
from datetime import datetime
import uuid


# 模型名称枚举
class ModelName(str, Enum):
    silicon_flow = "silicon-flow"
    OpenAI = "OpenAI"
    moonshot = "moonshot"


MODEL_TO_APIKEY = {
    "OpenAI": "Bearer app-dsGpzD1RooaIGsnroHR70NR1",
    "silicon-flow": "Bearer app-9A6JzkmmfNH6uY2o5aQFyIR0",
    "moonshot": "Bearer app-k8YszVVuxK8ep6Dj1YKRv20Q"
}

dify_url = "https://api.dify.ai/v1/chat-messages"


# 请求模型
class ProcessRequest(BaseModel):
    requestId: str
    model: str
    prompt: str


# 响应模型
class ProcessResponse(BaseModel):
    status: str
    requestId: str
    model: str
    queuePosition: int


# 队列更新模型
class QueueInfo(BaseModel):
    queueLength: int
    processingCount: int


class CompletedRequest(BaseModel):
    requestId: str
    model: str
    status: str
    result: str
    timestamp: str


class QueueUpdateRequest(BaseModel):
    queues: Dict[str, QueueInfo]
    completedRequests: List[CompletedRequest]


# 队列管理类
class QueueManager:
    def __init__(self):
        self.queues: Dict[str, Queue] = {}
        self.processing_count: Dict[str, int] = {}
        self.completed_requests: Dict[str, dict] = {}
        self.workers: Dict[str, threading.Thread] = {}
        self.running = True
        
        for model in ["gpt-4", "DeepSeek-R1", "moonshot-v1-8k"]:
            self.queues[model] = Queue()
            self.processing_count[model] = 0
            self.start_worker(model)
    
    def start_worker(self, model: str):
        """为每个模型启动一个工作线程"""
        worker = threading.Thread(target=self._worker, args=(model,), daemon=True)
        worker.start()
        self.workers[model] = worker
    
    def _worker(self, model: str):
        """工作线程处理函数"""
        while self.running:
            try:
                if not self.queues[model].empty():
                    request_data = self.queues[model].get(timeout=1)
                    self.processing_count[model] += 1
                    
                    # 处理请求
                    asyncio.run(self._process_request(request_data, model))
                    
                    self.processing_count[model] -= 1
                    self.queues[model].task_done()
                else:
                    time.sleep(0.1)
            except:
                continue
    
    async def _process_request(self, request_data: dict, model: str):
        """处理单个请求"""
        try:
            result = await call_dify(model, request_data["prompt"])
            
            self.completed_requests[request_data["requestId"]] = {
                "requestId": request_data["requestId"],
                "model": model,
                "status": "completed",
                "result": result,
                "timestamp": datetime.now().isoformat() + "Z"
            }
        except Exception as e:
            # 处理失败的请求
            self.completed_requests[request_data["requestId"]] = {
                "requestId": request_data["requestId"],
                "model": model,
                "status": "failed",
                "result": f"处理失败: {str(e)}",
                "timestamp": datetime.now().isoformat() + "Z"
            }
    
    def add_request(self, request: ProcessRequest) -> int:
        """添加请求到队列"""
        request_data = {
            "requestId": request.requestId,
            "prompt": request.prompt
        }
        
        if request.model not in self.queues:
            self.queues[request.model] = Queue()
            self.processing_count[request.model] = 0
            self.start_worker(request.model)
        
        self.queues[request.model].put(request_data)
        return self.queues[request.model].qsize()
    
    def get_queue_position(self, model: str) -> int:
        """获取队列位置"""
        return self.queues.get(model, Queue()).qsize()


# 全局队列管理器
queue_manager = QueueManager()

app = FastAPI()


async def call_dify(model: str, question: str) -> str:
    """原有的 call_dify 函数"""
    api_key = MODEL_TO_APIKEY.get(model)

    if not api_key:
        return f"[error]模型{model}未配置API KEY"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "inputs": {},
        "query": question,
        "response_mode": "blocking",
        "user": "test-user-id",
        "conversation_id": None
    }

    timeout = httpx.Timeout(120.0, read=120.0, connect=10.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(dify_url, headers=headers, json=data)

            print("状态码:", resp.status_code)
            print("响应内容:", resp.text)

            if resp.status_code == 504:
                raise HTTPException(status_code=504, detail="[Dify错误]模型响应超时，请稍后再试")

            try:
                result = resp.json()
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"[响应格式错误]无法解析JSON:{e}\n响应内容:{resp.text}")

            if "answer" in result:
                return result["answer"]
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


@app.post("/llm/process", response_model=ProcessResponse)
async def process_request(request: ProcessRequest):
    """接收Java端的处理请求"""
    if not request.prompt:
        raise HTTPException(status_code=400, detail="prompt 不能为空")
    
    # 添加请求到队列
    queue_position = queue_manager.add_request(request)
    
    return ProcessResponse(
        status="queued",
        requestId=request.requestId,
        model=request.model,
        queuePosition=queue_position
    )


@app.post("/llm/queue/update")
async def update_queue_status(request: QueueUpdateRequest):
    """从Java端接收队列状态更新"""
    return {"status": "success", "message": "队列状态已更新"}


@app.get("/llm/result/{request_id}")
async def get_result(request_id: str):
    """获取处理结果"""
    if request_id in queue_manager.completed_requests:
        return queue_manager.completed_requests[request_id]
    else:
        raise HTTPException(status_code=404, detail="请求结果未找到")


@app.get("/llm/queue/status")
async def get_queue_status():
    """获取队列状态"""
    status = {}
    for model, queue in queue_manager.queues.items():
        status[model] = {
            "queueLength": queue.qsize(),
            "processingCount": queue_manager.processing_count[model]
        }
    return status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
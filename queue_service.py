import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
import asyncio
import threading
from queue import Queue
import time
from datetime import datetime
from call_llm import call_dify
import uuid
import consul
import socket
import os
import atexit


class ProcessRequest(BaseModel):
    requestId: str
    model: str
    prompt: str


class ProcessResponse(BaseModel):
    status: str
    requestId: str
    model: str
    queuePosition: int


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


class QueueManager:
    def __init__(self):
        self.queues: Dict[str, Queue] = {}
        self.processing_count: Dict[str, int] = {}
        self.completed_requests: Dict[str, dict] = {}
        self.workers: Dict[str, threading.Thread] = {}
        self.running = True

        # 实现队列信息更新新增
        self.pushed_request_ids: set = set()
        self.push_lock = threading.Lock()

        for model in ["OpenAI", "silicon-flow", "moonshot"]:
            self.queues[model] = Queue()
            self.processing_count[model] = 0
            self.start_worker(model)

    def start_background_push_loop(self):
        loop = asyncio.get_event_loop()
        loop.create_task(self.periodic_push_to_java())

    def start_worker(self, model: str):
        worker = threading.Thread(target = self._worker, args = (model,), daemon = True)
        worker.start()
        self.workers[model] = worker

    def _worker(self, model: str):
        while self.running:
            try:
                if not self.queues[model].empty():
                    request_data = self.queues[model].get(timeout = 1)
                    self.processing_count[model] += 1
                    asyncio.run(self._process_request(request_data, model))
                    self.processing_count[model] -= 1
                    self.queues[model].task_done()
                else:
                    time.sleep(0.1)
            except:
                continue

    async def _process_request(self, request_data: dict, model: str):
        try:
            async with httpx.AsyncClient(timeout = 30.0,proxies = None) as client:
                response = await client.post(
                    "http://localhost:8000/llm",  ##
                    json = {
                        "model":model,
                        "question":request_data["prompt"],
                        "requestId":request_data["requestId"]
                    }
                )

            if response.status_code == 200:
                result = response.json().get("answer", "[错误] 未返回answer字段")
                self.completed_requests[request_data["requestId"]] = {
                    "requestId":request_data["requestId"],
                    "model":model,
                    "status":"completed",
                    "result":result,
                    "timestamp":datetime.now().isoformat() + "Z"
                }
            else:
                self.completed_requests[request_data["requestId"]] = {
                    "requestId":request_data["requestId"],
                    "model":model,
                    "status":"failed",
                    "result":f"[HTTP错误] 状态码: {response.status_code}, 内容: {response.text}",
                    "timestamp":datetime.now().isoformat() + "Z"
                }
        except Exception as e:
            self.completed_requests[request_data["requestId"]] = {
                "requestId":request_data["requestId"],
                "model":model,
                "status":"failed",
                "result":f"Processing failed: {str(e)}",
                "timestamp":datetime.now().isoformat() + "Z"
            }

    def add_request(self, request: ProcessRequest) -> int:
        request_data = {
            "requestId":request.requestId,
            "prompt":request.prompt
        }

        if request.model not in self.queues:
            self.queues[request.model] = Queue()
            self.processing_count[request.model] = 0
            self.start_worker(request.model)

        self.queues[request.model].put(request_data)
        return self.queues[request.model].qsize()

    def get_queue_position(self, model: str) -> int:
        return self.queues.get(model, Queue()).qsize()

    async def periodic_push_to_java(self, interval: float = 5.0, batch_size: int = 3):
        while self.running:
            try:
                await asyncio.sleep(interval)

                new_completed = []
                with self.push_lock:
                    for req_id, data in self.completed_requests.items():
                        if req_id not in self.pushed_request_ids:
                            new_completed.append(data)
                            if len(new_completed) >= batch_size:
                                break

                if not new_completed:
                    continue

                payload = {
                    "queues":{
                        model:{
                            "queueLength":self.queues[model].qsize(),
                            "processingCount":self.processing_count[model]
                        } for model in self.queues
                    },
                    "completedRequests":new_completed
                }

                #java端口的地址
                java_backend_url = "http://localhost:8001/llm/queue/update"  # 使用实际的Java服务地址和端口

                async with httpx.AsyncClient(timeout = 10.0) as client:
                    resp = await client.post(java_backend_url, json = payload)
                    if resp.status_code == 200:
                        with self.push_lock:
                            for item in new_completed:
                                self.pushed_request_ids.add(item["requestId"])
                    else:
                        print(f"[WARN] Java 返回 {resp.status_code}: {resp.text}")

            except Exception as e:
                print(f"[ERROR] 推送失败: {e}")


queue_manager = QueueManager()
queue_manager.start_background_push_loop()
app = FastAPI()

# 添加服务启动和关闭事件
@app.on_event("startup")
async def startup_event():
    """服务启动时注册到Consul"""
    service_id = register_service()
    if service_id:
        app.state.service_id = service_id
        print(f"queue_service服务已注册到Consul，服务ID: {service_id}")

@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭时从Consul注销"""
    if hasattr(app.state, 'service_id'):
        deregister_service(app.state.service_id)

# Consul配置
CONSUL_HOST = os.getenv('CONSUL_HOST', 'localhost')
CONSUL_PORT = int(os.getenv('CONSUL_PORT', 8500))
SERVICE_NAME = 'queue-service'
SERVICE_PORT = 8001
SERVICE_ID = f"{SERVICE_NAME}-8001" 
DATACENTER = "dssc"  

def get_host_info():
    """获取主机信息"""
    return "bdap-cluster-03"  # 直接返回固定主机名

def register_service():
    """注册服务到Consul"""
    try:
        c = consul.Consul(host=CONSUL_HOST, port=CONSUL_PORT)
        service_id = SERVICE_ID
        host = get_host_info()
        
        c.agent.service.register(
            name=SERVICE_NAME,
            service_id=service_id,
            address=host,
            port=SERVICE_PORT,
            tags=["secure=false"],  # 匹配现有标签
            check=consul.Check.http(f"http://{host}:{SERVICE_PORT}/health", interval="10s"),
            datacenter=DATACENTER
        )
        print(f"服务 {SERVICE_NAME} 已注册到Consul: {host}:{SERVICE_PORT}")
        return service_id
    except Exception as e:
        print(f"注册服务到Consul失败: {e}")
        return None

def deregister_service(service_id):
    """从Consul注销服务"""
    try:
        c = consul.Consul(host=CONSUL_HOST, port=CONSUL_PORT)
        c.agent.service.deregister(service_id)
        print(f"服务 {service_id} 已从Consul注销")
    except Exception as e:
        print(f"从Consul注销服务失败: {e}")

# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": SERVICE_NAME}


@app.post("/llm/process", response_model = ProcessResponse)
async def process_request(request: ProcessRequest):
    if not request.prompt:
        raise HTTPException(status_code = 400, detail = "prompt cannot be empty")

    queue_position = queue_manager.add_request(request)

    return ProcessResponse(
        status = "queued",
        requestId = request.requestId,
        model = request.model,
        queuePosition = queue_position
    )


@app.post("/llm/queue/update")
async def update_queue_status(request: QueueUpdateRequest):
    return {"status":"success", "message":"Queue status updated"}


@app.get("/llm/result/{request_id}")
async def get_result(request_id: str):
    if request_id in queue_manager.completed_requests:
        return queue_manager.completed_requests[request_id]
    else:
        raise HTTPException(status_code = 404, detail = "Request result not found")


@app.get("/llm/queue/status")
async def get_queue_status():
    status = {}
    for model, queue in queue_manager.queues.items():
        status[model] = {
            "queueLength":queue.qsize(),
            "processingCount":queue_manager.processing_count[model]
        }
    return status


if __name__ == "__main__":
    import uvicorn
    
    # 注册服务到Consul
    service_id = register_service()
    
    # 程序退出时注销服务
    if service_id:
        atexit.register(deregister_service, service_id)
    
    uvicorn.run(app, host="localhost", port=SERVICE_PORT)

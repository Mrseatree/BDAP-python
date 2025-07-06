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


# ģ������ö��
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


# ����ģ��
class ProcessRequest(BaseModel):
    requestId: str
    model: str
    prompt: str


# ��Ӧģ��
class ProcessResponse(BaseModel):
    status: str
    requestId: str
    model: str
    queuePosition: int


# ���и���ģ��
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


# ���й�����
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
        """Ϊÿ��ģ������һ�������߳�"""
        worker = threading.Thread(target=self._worker, args=(model,), daemon=True)
        worker.start()
        self.workers[model] = worker
    
    def _worker(self, model: str):
        """�����̴߳�����"""
        while self.running:
            try:
                if not self.queues[model].empty():
                    request_data = self.queues[model].get(timeout=1)
                    self.processing_count[model] += 1
                    
                    # ��������
                    asyncio.run(self._process_request(request_data, model))
                    
                    self.processing_count[model] -= 1
                    self.queues[model].task_done()
                else:
                    time.sleep(0.1)
            except:
                continue
    
    async def _process_request(self, request_data: dict, model: str):
        """����������"""
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
            # ����ʧ�ܵ�����
            self.completed_requests[request_data["requestId"]] = {
                "requestId": request_data["requestId"],
                "model": model,
                "status": "failed",
                "result": f"����ʧ��: {str(e)}",
                "timestamp": datetime.now().isoformat() + "Z"
            }
    
    def add_request(self, request: ProcessRequest) -> int:
        """������󵽶���"""
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
        """��ȡ����λ��"""
        return self.queues.get(model, Queue()).qsize()


# ȫ�ֶ��й�����
queue_manager = QueueManager()

app = FastAPI()


async def call_dify(model: str, question: str) -> str:
    """ԭ�е� call_dify ����"""
    api_key = MODEL_TO_APIKEY.get(model)

    if not api_key:
        return f"[error]ģ��{model}δ����API KEY"

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

            print("״̬��:", resp.status_code)
            print("��Ӧ����:", resp.text)

            if resp.status_code == 504:
                raise HTTPException(status_code=504, detail="[Dify����]ģ����Ӧ��ʱ�����Ժ�����")

            try:
                result = resp.json()
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"[��Ӧ��ʽ����]�޷�����JSON:{e}\n��Ӧ����:{resp.text}")

            if "answer" in result:
                return result["answer"]
            elif "message" in result:
                raise HTTPException(status_code=502, detail=f"[Dify����] {result['message']}")
            else:
                raise HTTPException(status_code=502, detail="[Dify��Ӧ��ʽ�쳣]")

    except httpx.ReadTimeout:
        raise HTTPException(status_code=504, detail="[��ʱ] Dify ��Ӧ��ʱ")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"[����ʧ��] {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[δ֪����] {e}")


@app.post("/llm/process", response_model=ProcessResponse)
async def process_request(request: ProcessRequest):
    """����Java�˵Ĵ�������"""
    if not request.prompt:
        raise HTTPException(status_code=400, detail="prompt ����Ϊ��")
    
    # ������󵽶���
    queue_position = queue_manager.add_request(request)
    
    return ProcessResponse(
        status="queued",
        requestId=request.requestId,
        model=request.model,
        queuePosition=queue_position
    )


@app.post("/llm/queue/update")
async def update_queue_status(request: QueueUpdateRequest):
    """��Java�˽��ն���״̬����"""
    return {"status": "success", "message": "����״̬�Ѹ���"}


@app.get("/llm/result/{request_id}")
async def get_result(request_id: str):
    """��ȡ������"""
    if request_id in queue_manager.completed_requests:
        return queue_manager.completed_requests[request_id]
    else:
        raise HTTPException(status_code=404, detail="������δ�ҵ�")


@app.get("/llm/queue/status")
async def get_queue_status():
    """��ȡ����״̬"""
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
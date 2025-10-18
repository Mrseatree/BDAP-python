from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import httpx
import asyncio
from queue import PriorityQueue
import time
from consul_utils import register_service, deregister_service
from config import SERVICE_NAME
from WorkflowValidator import SimplifiedWorkflowValidator
from call_llm import call_dify

app = FastAPI()

# 每个模型的并发任务数
NUM_WORKERS_PER_MODEL = 30 

# 重传机制配置
MAX_RETRIES = 3  # 最大重试次数
RETRY_DELAY = 5  # 重试延迟（秒）

# 请求模型
class WorkflowGenerationRequest(BaseModel):
    model: str
    requestId: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    user_prompt: str
    template_type: Optional[str] = "data_processing"
    service_type: Optional[str] = "ml"
    isWorkFlow: bool = True

# 响应模型
class AsyncWorkflowResponse(BaseModel):
    requestId: str
    status: str
    message: str

# 工作流结果模型
class WorkflowResult(BaseModel):
    requestId: str
    status: str
    conversation_id: Optional[str] = None
    workflow_info: Optional[Dict[str, Any]] = None
    nodes: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = ""

# 内部工作流请求（用于重传）
class InternalWorkflowRequest:
    def __init__(self, request_data: dict, retry_count: int = 0):
        self.request_data = request_data
        self.retry_count = retry_count
        self.created_time = time.time()
    
    def __lt__(self, other):
        # 用于PriorityQueue的优先级比较（retry_count越少优先级越高）
        return self.retry_count < other.retry_count

# 更新后的模型定义
class WorkflowInfo(BaseModel):
    userId: str

class SimpleAttribute(BaseModel):
    name: str
    value: str
    valueType: str

class ComplicatedAttribute(BaseModel):
    name: str
    value: Dict[str, Any]

class SourceAnchor(BaseModel):
    nodeName: str
    nodeMark: int
    seq: int = 0

class TargetAnchor(BaseModel):
    nodeName: str
    nodeMark: int
    seq: int = 0

class InputAnchor(BaseModel):
    seq: int = 0
    numOfConnectedEdges: int = 0
    sourceAnchor: Optional[SourceAnchor] = None

class OutputAnchor(BaseModel):
    seq: int = 0
    numOfConnectedEdges: int = 0
    targetAnchors: List[TargetAnchor] = []

class Node(BaseModel):
    id: str
    mark: str
    inputAnchors: List[InputAnchor] = []
    outputAnchors: List[OutputAnchor] = []


class WorkflowQueueManager:
    def __init__(self, num_workers_per_model: int = NUM_WORKERS_PER_MODEL):
        self.num_workers_per_model = num_workers_per_model
        self.pending_queues: Dict[str, asyncio.Queue] = {}
        self.retry_queues: Dict[str, PriorityQueue] = {}
        self.processing_count: Dict[str, int] = {}
        self.completed_requests: Dict[str, WorkflowResult] = {}
        self.worker_tasks: Dict[str, List[asyncio.Task]] = {}
        self.retry_tasks: Dict[str, asyncio.Task] = {}
        self.running = True
        self.lock = asyncio.Lock()
        
        # 初始化支持的模型队列
        for model in ["silicon-flow", "moonshot", "deepseek", "Qwen"]:
            self.pending_queues[model] = asyncio.Queue()
            self.retry_queues[model] = PriorityQueue()
            self.processing_count[model] = 0
            self.worker_tasks[model] = []

    async def start_workers(self, model: str):
        """为指定模型启动多个异步工作任务"""
        for i in range(self.num_workers_per_model):
            task = asyncio.create_task(self._worker(model, i))
            self.worker_tasks[model].append(task)
            print(f"启动异步工作任务: {model}-worker-{i}")
        
        # 启动重试工作任务
        retry_task = asyncio.create_task(self._retry_worker(model))
        self.retry_tasks[model] = retry_task
        print(f"启动重试工作任务: {model}-retry-worker")

    async def _worker(self, model: str, worker_id: int):
        """异步工作任务处理队列中的工作流请求"""
        print(f"工作任务 {model}-worker-{worker_id} 已启动")
        
        while self.running:
            try:
                # 非阻塞获取，1秒超时
                try:
                    request_data = await asyncio.wait_for(
                        self.pending_queues[model].get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                print(f"[{model}-worker-{worker_id}] 获取到请求: {request_data['requestId']}")
                
                self.processing_count[model] += 1
                
                try:
                    await self._process_workflow_request(request_data, model, worker_id, retry_count=0)
                finally:
                    self.processing_count[model] -= 1
                    self.pending_queues[model].task_done()
                    
            except Exception as e:
                print(f"[{model}-worker-{worker_id}] 工作任务错误: {e}")
                if model in self.processing_count:
                    self.processing_count[model] = max(0, self.processing_count[model] - 1)
                continue

    async def _retry_worker(self, model: str):
        """重试工作任务处理失败的工作流请求"""
        print(f"重试工作任务 {model}-retry-worker 已启动")
        
        while self.running:
            try:
                await asyncio.sleep(0.5)
                
                # 检查重试队列（非阻塞）
                if self.retry_queues[model].empty():
                    continue
                
                priority, internal_request = self.retry_queues[model].get_nowait()
                
                # 检查是否达到重试延迟时间
                elapsed_time = time.time() - internal_request.created_time
                if elapsed_time < RETRY_DELAY:
                    # 还没到重试时间，放回队列
                    self.retry_queues[model].put((priority, internal_request))
                    await asyncio.sleep(0.5)
                    continue
                
                request_id = internal_request.request_data['requestId']
                retry_count = internal_request.retry_count
                
                print(f"[{model}-retry-worker] 开始第 {retry_count + 1} 次重试，requestId: {request_id}")
                
                self.processing_count[model] += 1
                
                try:
                    await self._process_workflow_request(
                        internal_request.request_data, 
                        model, 
                        -1,  # 标记为重试工作任务
                        retry_count=retry_count
                    )
                finally:
                    self.processing_count[model] -= 1
                    self.retry_queues[model].task_done()
                    
            except Exception as e:
                print(f"[{model}-retry-worker] 重试任务错误: {e}")
                if model in self.processing_count:
                    self.processing_count[model] = max(0, self.processing_count[model] - 1)
                continue

    async def _process_workflow_request(self, request_data: dict, model: str, worker_id: int, retry_count: int = 0):
        try:
            print(f"[{model}-worker-{worker_id}] 开始处理工作流请求 {request_data['requestId']}，重试次数: {retry_count}")
            
            # 1. 调用大模型生成工作流
            llm_response, new_conversation_id = await call_dify_with_workflow(
                model=model,
                prompt=request_data["user_prompt"],
                user_id=request_data["user_id"],
                conversation_id=request_data["conversation_id"],
                request_id=request_data["requestId"],
                isWorkFlow=str(request_data["isWorkFlow"]).lower()
            )
            
            # 2. 解析LLM响应
            workflow_structure = parse_llm_response(
                llm_response=llm_response,
                user_id=request_data["user_id"],
                service_type=request_data["service_type"],
                request_id=request_data["requestId"],
                conversation_id=new_conversation_id
            )
            
            # 3. 工作流校验
            validator = SimplifiedWorkflowValidator()
            sanitized_workflow, warnings, errors = await validator.sanitize(workflow_structure)
            
            if sanitized_workflow is None:
                # 校验失败 - 需要重传
                error_result = WorkflowResult(
                    requestId=request_data["requestId"],
                    status="validation_failed",
                    conversation_id=new_conversation_id,
                    error_message=f"工作流结构校验失败: {', '.join(errors)}"
                )
                
                # 检查是否需要重传
                if retry_count < MAX_RETRIES:
                    print(f"[{model}-worker-{worker_id}] 工作流 {request_data['requestId']} 校验失败，将其加入重试队列")
                    # 将请求加入重试队列
                    internal_request = InternalWorkflowRequest(request_data, retry_count + 1)
                    self.retry_queues[model].put((retry_count + 1, internal_request))
                    # 暂时不缓存结果，等待重试
                    return
                else:
                    # 已达到最大重试次数
                    print(f"[{model}-worker-{worker_id}] 工作流 {request_data['requestId']} 校验失败，已达到最大重试次数")
                    error_result.error_message = f"工作流结构校验失败，已重试{retry_count}次: {', '.join(errors)}"
                    
                    async with self.lock:
                        self.completed_requests[request_data["requestId"]] = error_result
                    
                    await self._push_single_result_to_java(error_result)
                    return
            
            if warnings:
                print(f"[{model}-worker-{worker_id}] 工作流校验警告: {', '.join(warnings)}")
            
            # 4. 创建成功结果
            result = WorkflowResult(
                requestId=request_data["requestId"],
                status="success",
                conversation_id=new_conversation_id,
                workflow_info=sanitized_workflow["workflow_info"],
                nodes=sanitized_workflow["nodes"],
                error_message=""
            )
            
            # 缓存成功结果
            async with self.lock:
                self.completed_requests[request_data["requestId"]] = result
            
            # 立即推送成功结果
            await self._push_single_result_to_java(result)
            
            print(f"[{model}-worker-{worker_id}] 工作流请求 {request_data['requestId']} 处理成功")
            
        except ValueError as e:
            error_msg = str(e)
            status = "error"
            
            # 根据错误信息判断状态
            if "超时" in error_msg or "timeout" in error_msg.lower():
                status = "timeout"
            elif "繁忙" in error_msg or "busy" in error_msg.lower():
                status = "busy"
            
            print(f"[{model}-worker-{worker_id}] 处理工作流请求 {request_data['requestId']} 时发生错误: {error_msg}")
            
            # 创建失败结果
            error_result = WorkflowResult(
                requestId=request_data["requestId"],
                status=status,
                error_message=error_msg
            )
            
            # 检查是否需要重传（只有timeout状态需要重传）
            if status == "timeout" and retry_count < MAX_RETRIES:
                print(f"[{model}-worker-{worker_id}] 工作流 {request_data['requestId']} 超时，将其加入重试队列")
                # 将请求加入重试队列
                internal_request = InternalWorkflowRequest(request_data, retry_count + 1)
                self.retry_queues[model].put((retry_count + 1, internal_request))
                # 暂时不缓存结果，等待重试
                return
            elif status == "timeout" and retry_count >= MAX_RETRIES:
                # 已达到最大重试次数
                print(f"[{model}-worker-{worker_id}] 工作流 {request_data['requestId']} 超时，已达到最大重试次数")
                error_result.error_message = f"请求超时，已重试{retry_count}次: {error_msg}"
            
            # 缓存失败结果
            async with self.lock:
                self.completed_requests[request_data["requestId"]] = error_result
            
            # 立即推送失败结果
            await self._push_single_result_to_java(error_result)
            
        except Exception as e:
            error_msg = f"工作流生成失败: {str(e)}"
            print(f"[{model}-worker-{worker_id}] 处理工作流请求 {request_data['requestId']} 时发生未知错误: {error_msg}")
            
            # 创建失败结果
            error_result = WorkflowResult(
                requestId=request_data["requestId"],
                status="error",
                error_message=error_msg
            )
            
            # 缓存失败结果
            async with self.lock:
                self.completed_requests[request_data["requestId"]] = error_result
            
            # 立即推送失败结果
            await self._push_single_result_to_java(error_result)

    async def _push_single_result_to_java(self, result: WorkflowResult):
        """推送工作流结果到Java后端"""
        try:
            callback_url = "http://10.92.64.219:7003/llm/result/experiment"
            
            headers = {
                "Content-Type": "application/json"
            }
            
            timeout = httpx.Timeout(30.0, read=30.0, connect=10.0)
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    callback_url,
                    headers=headers,
                    json=result.dict()
                )
                
                if response.status_code == 200:
                    print(f"成功推送工作流结果到Java端，requestId: {result.requestId}")
                else:
                    print(f"推送工作流结果到Java端失败，状态码: {response.status_code}, 响应: {response.text}")
                    
        except httpx.RequestError as e:
            print(f"推送工作流结果到Java端请求失败: {e}")
        except Exception as e:
            print(f"推送工作流结果到Java端时发生未知错误: {e}")

    async def add_request(self, request: WorkflowGenerationRequest):
        """添加工作流请求到队列"""
        request_data = {
            "requestId": request.requestId,
            "user_prompt": request.user_prompt,
            "user_id": request.user_id or "anonymous",
            "conversation_id": request.conversation_id,
            "template_type": request.template_type,
            "service_type": request.service_type,
            "isWorkFlow": request.isWorkFlow
        }

        if request.model not in self.pending_queues:
            print(f"创建新的工作流队列和工作任务: {request.model}")
            self.pending_queues[request.model] = asyncio.Queue()
            self.retry_queues[request.model] = PriorityQueue()
            self.processing_count[request.model] = 0
            self.worker_tasks[request.model] = []
            await self.start_workers(request.model)

        await self.pending_queues[request.model].put(request_data)
        queue_size = self.pending_queues[request.model].qsize()
        
        print(f"工作流请求 {request.requestId} 已添加到 {request.model} 队列，当前队列长度: {queue_size}")

    def get_queue_position(self, model: str) -> int:
        queue = self.pending_queues.get(model)
        return queue.qsize() if queue else 0

    async def get_result(self, request_id: str) -> Optional[WorkflowResult]:
        async with self.lock:
            return self.completed_requests.get(request_id)

    async def clear_all_results(self):
        async with self.lock:
            count = len(self.completed_requests)
            self.completed_requests.clear()
            return count

    async def stop(self):
        self.running = False
        # 等待所有任务完成
        for model, tasks in self.worker_tasks.items():
            for task in tasks:
                task.cancel()
            if model in self.retry_tasks:
                self.retry_tasks[model].cancel()


# dify调用函数
async def call_dify_with_workflow(model: str, prompt: str, user_id: str, request_id: str, 
                                 conversation_id: Optional[str] = None, isWorkFlow: str = "false") -> tuple:
    try:
        from call_llm import MODEL_TO_APIKEY, dify_url
        import httpx
        
        api_key = MODEL_TO_APIKEY.get(model)
        if not api_key:
            raise ValueError(f"模型{model}未配置API KEY")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "inputs": {
                "requestId": request_id,
                "isWorkFlow": isWorkFlow,
                "file_content1": ""
            },
            "query": prompt,
            "response_mode": "blocking",
            "user": user_id,
            "conversation_id": conversation_id or ""
        }

        timeout = httpx.Timeout(120.0, read=120.0, connect=10.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(dify_url, headers=headers, json=data)

            print("状态码:", resp.status_code)
            print("原始内容:", resp.text)

            if resp.status_code == 504:
                raise ValueError("[Dify错误]模型响应超时，稍后再试")
            
            if resp.status_code != 200:
                raise ValueError(f"[Dify API错误]状态码: {resp.status_code}, 响应: {resp.text}")

            try:
                result = resp.json()
            except Exception as e:
                raise ValueError(f"[响应格式错误]无法解析JSON:{e}\n原始响应:{resp.text}")

            if "answer" in result:
                return result["answer"], result.get("conversation_id")
            elif "message" in result:
                raise ValueError(f"[Dify错误] {result['message']}")
            else:
                raise ValueError("[Dify响应格式异常]")

    except httpx.ReadTimeout:
        raise ValueError("[超时] Dify 响应超时")
    except httpx.RequestError as e:
        raise ValueError(f"[请求失败] {e}")
    except HTTPException as e:
        raise ValueError(f"[HTTP错误] {e.detail}")
    except Exception as e:
        raise ValueError(f"[未知错误] {e}")

def parse_llm_response(llm_response: Any, user_id: str, service_type: str, request_id: str, conversation_id: str = None) -> Dict[str, Any]:
    try:
        if isinstance(llm_response, tuple) and len(llm_response) >= 1:
            response_data = llm_response[0]
        else:
            response_data = llm_response
        
        if not isinstance(response_data, str):
            response_data = str(response_data)
        
        print(f"解析的响应数据: {response_data[:500]}...")
        
        start_idx = response_data.find('{')
        end_idx = response_data.rfind('}') + 1
        
        if start_idx != -1 and end_idx != -1:
            json_str = response_data[start_idx:end_idx]
            print(f"提取的JSON字符串: {json_str[:200]}...")
            
            workflow_data = json.loads(json_str)
            
            if not isinstance(workflow_data, dict):
                raise ValueError(f"解析的工作流数据不是字典类型，而是: {type(workflow_data)}")
            
            # 初始化基本结构
            if "workflow_info" not in workflow_data:
                workflow_data["workflow_info"] = {}
            
            if "nodes" not in workflow_data:
                workflow_data["nodes"] = []
            
            if not isinstance(workflow_data["nodes"], list):
                raise ValueError(f"nodes字段不是列表类型，而是: {type(workflow_data['nodes'])}")
            
            workflow_data["requestId"] = request_id
            workflow_data["conversation_id"] = conversation_id
            
            if not isinstance(workflow_data["workflow_info"], dict):
                workflow_data["workflow_info"] = {}
            workflow_data["workflow_info"]["userId"] = user_id or "anonymous"
            
            # 处理节点数据
            for i, node in enumerate(workflow_data["nodes"]):
                if not isinstance(node, dict):
                    raise ValueError(f"节点{i}不是字典类型，而是: {type(node)}")
                
                if "id" not in node:
                    node["id"] = f"node_{i}"
                
                if "mark" not in node:
                    node["mark"] = str(i)
                
                node.pop("name", None)
                node.pop("position", None)
                node.pop("simpleAttributes", None)
                node.pop("complicatedAttributes", None)
                
                node.setdefault("inputAnchors", [])
                node.setdefault("outputAnchors", [])
                
                if not isinstance(node["inputAnchors"], list):
                    node["inputAnchors"] = []
                if not isinstance(node["outputAnchors"], list):
                    node["outputAnchors"] = []
                
                for j, input_anchor in enumerate(node["inputAnchors"]):
                    if isinstance(input_anchor, dict):
                        input_anchor.setdefault("seq", j)
                        input_anchor.setdefault("numOfConnectedEdges", 0)
                        
                        if "sourceAnchors" in input_anchor and input_anchor["sourceAnchors"]:
                            if isinstance(input_anchor["sourceAnchors"], list) and len(input_anchor["sourceAnchors"]) > 0:
                                old_source = input_anchor["sourceAnchors"][0]
                                input_anchor["sourceAnchor"] = {
                                    "nodeName": old_source.get("nodeName", old_source.get("id", "")),
                                    "nodeMark": old_source.get("nodeMark", old_source.get("mark", 0)),
                                    "seq": old_source.get("seq", 0)
                                }
                            input_anchor.pop("sourceAnchors", None)
                        
                        if "sourceAnchor" in input_anchor and input_anchor["sourceAnchor"]:
                            source_anchor = input_anchor["sourceAnchor"]
                            source_anchor.setdefault("seq", 0)
                            
                            if "nodeMark" in source_anchor:
                                try:
                                    source_anchor["nodeMark"] = int(source_anchor["nodeMark"])
                                except (ValueError, TypeError):
                                    source_anchor["nodeMark"] = 0
                            
                            input_anchor["numOfConnectedEdges"] = 1 if input_anchor.get("sourceAnchor") else 0
                
                for j, output_anchor in enumerate(node["outputAnchors"]):
                    if isinstance(output_anchor, dict):
                        output_anchor.setdefault("seq", j)
                        output_anchor.setdefault("numOfConnectedEdges", 0)
                        output_anchor.setdefault("targetAnchors", [])
                        
                        for k, target_anchor in enumerate(output_anchor["targetAnchors"]):
                            if isinstance(target_anchor, dict):
                                target_anchor.setdefault("nodeName", target_anchor.get("id", ""))
                                target_anchor.setdefault("seq", k)
                                
                                if "nodeMark" not in target_anchor:
                                    target_anchor["nodeMark"] = target_anchor.get("mark", 0)
                                
                                try:
                                    target_anchor["nodeMark"] = int(target_anchor["nodeMark"])
                                except (ValueError, TypeError):
                                    target_anchor["nodeMark"] = 0
                                
                                target_anchor.pop("mark", None)
                                target_anchor.pop("id", None)
                        
                        output_anchor["numOfConnectedEdges"] = len(output_anchor.get("targetAnchors", []))
            
            return workflow_data
        else:
            raise ValueError("LLM响应中未找到有效的JSON结构")
            
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON解析失败: {e}")
    except Exception as e:
        raise ValueError(f"解析LLM响应失败: {e}")


# 创建全局工作流队列管理器
workflow_queue_manager = WorkflowQueueManager(num_workers_per_model=NUM_WORKERS_PER_MODEL)

@app.on_event("startup")
async def startup_event():
    """服务启动时注册到Consul并启动所有工作任务"""
    SERVICE_PORT = 8004
    service_id = register_service(SERVICE_PORT)
    if service_id:
        app.state.service_id = service_id
        print(f"workflow_generator服务已注册到Consul，服务ID: {service_id}")
    
    # 启动所有模型的工作任务
    for model in ["silicon-flow", "moonshot", "deepseek", "Qwen"]:
        await workflow_queue_manager.start_workers(model)

@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭时从Consul注销"""
    await workflow_queue_manager.stop()
    
    if hasattr(app.state, 'service_id'):
        deregister_service(app.state.service_id)

@app.post("/workflow/generate", response_model=AsyncWorkflowResponse)
async def generate_workflow(request: WorkflowGenerationRequest):
    try:
        if not request.user_prompt:
            raise HTTPException(status_code=400, detail="user_prompt cannot be empty")
        
        print(f"收到工作流生成请求: {request.requestId}, 模型: {request.model}")
        
        # 添加请求到队列
        await workflow_queue_manager.add_request(request)
        
        return AsyncWorkflowResponse(
            requestId=request.requestId,
            status="processing",
            message="工作流生成任务已提交到队列，正在处理中..."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"提交工作流生成任务失败: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "workflow_generator"}

@app.get("/workflow/result/{request_id}")
async def get_workflow_result(request_id: str):
    """获取工作流生成结果"""
    result = await workflow_queue_manager.get_result(request_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"未找到请求ID为 {request_id} 的工作流结果")
    
    return result

@app.delete("/workflow/results")
async def clear_all_workflow_results():
    """清空所有工作流结果缓存（用于测试）"""
    count = await workflow_queue_manager.clear_all_results()
    return {"message": f"已清空所有工作流结果缓存，共删除 {count} 条记录"}

@app.get("/workflow/queue/status")
async def get_workflow_queue_status():
    """获取工作流队列状态（用于测试）"""
    total_queue_length = 0
    total_processing_count = 0
    total_retry_queue_length = 0
    
    for model, queue in workflow_queue_manager.pending_queues.items():
        total_queue_length += queue.qsize()
        total_processing_count += workflow_queue_manager.processing_count[model]
        total_retry_queue_length += workflow_queue_manager.retry_queues[model].qsize()
    
    status = {
        "queueLength": total_queue_length,
        "processingCount": total_processing_count,
        "retryQueueLength": total_retry_queue_length,
        "numWorkersPerModel": workflow_queue_manager.num_workers_per_model,
        "maxRetries": MAX_RETRIES,
        "retryDelay": RETRY_DELAY,
        "modelQueues": {
            model: {
                "queueLength": queue.qsize(),
                "retryQueueLength": workflow_queue_manager.retry_queues[model].qsize(),
                "processingCount": workflow_queue_manager.processing_count[model],
                "numWorkers": len(workflow_queue_manager.worker_tasks.get(model, []))
            } for model, queue in workflow_queue_manager.pending_queues.items()
        }
    }
    
    print(f"工作流队列状态: {status}")
    return status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)

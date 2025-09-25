from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import httpx
import asyncio
import threading
from queue import Queue
import time
from consul_utils import register_service, deregister_service
from config import SERVICE_NAME
from WorkflowValidator import SimplifiedWorkflowValidator
from call_llm import call_dify

app = FastAPI()

# 请求模型
class WorkflowGenerationRequest(BaseModel):
    model: str                          # 模型名称
    requestId: str                      # 请求ID
    user_id: Optional[str] = None       # 用户ID
    conversation_id: Optional[str] = None # 对话ID
    user_prompt: str                    # 用户需求描述
    template_type: Optional[str] = "data_processing" # 模板类型
    service_type: Optional[str] = "ml"  # 服务类型
    isWorkFlow: bool = True             # 设为 true

# 响应模型
class AsyncWorkflowResponse(BaseModel):
    requestId: str
    status: str  # "processing", "completed", "failed"
    message: str

# 工作流结果模型
class WorkflowResult(BaseModel):
    requestId: str
    status: str  # "success" or "error"
    conversation_id: Optional[str] = None
    workflow_info: Optional[Dict[str, Any]] = None
    nodes: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None

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
    name: str
    mark: str                           # 组件唯一标识
    position: List[int]
    simpleAttributes: List[SimpleAttribute] = []
    complicatedAttributes: List[ComplicatedAttribute] = []
    inputAnchors: List[InputAnchor] = []
    outputAnchors: List[OutputAnchor] = []


class WorkflowQueueManager:
    def __init__(self):
        self.queues: Dict[str, Queue] = {}
        self.processing_count: Dict[str, int] = {}
        self.completed_requests: Dict[str, WorkflowResult] = {}
        self.workers: Dict[str, threading.Thread] = {}
        self.running = True
        
        # 添加锁来保护 completed_requests
        self.completed_requests_lock = threading.Lock()

        # 初始化支持的模型队列
        for model in ["silicon-flow", "moonshot"]:
            self.queues[model] = Queue()
            self.processing_count[model] = 0
            self.start_worker(model)

    def start_worker(self, model: str):
        """启动指定模型的工作线程"""
        worker = threading.Thread(target=self._worker, args=(model,), daemon=True)
        worker.start()
        self.workers[model] = worker
        print(f"启动工作流 {model} 模型的工作线程")

    def _worker(self, model: str):
        """工作线程处理队列中的工作流请求"""
        print(f"启动工作流 {model} 模型的工作线程")
        while self.running:
            try:
                if not self.queues[model].empty():
                    request_data = self.queues[model].get(timeout=1)
                    print(f"工作流工作线程获取到请求: {request_data['requestId']}")
                    
                    self.processing_count[model] += 1
                    
                    # 创建新的事件循环来处理异步请求
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        loop.run_until_complete(self._process_workflow_request(request_data, model))
                    finally:
                        loop.close()
                    
                    self.processing_count[model] -= 1
                    self.queues[model].task_done()
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"工作流工作线程错误: {e}")
                if model in self.processing_count:
                    self.processing_count[model] = max(0, self.processing_count[model] - 1)
                continue

    async def _process_workflow_request(self, request_data: dict, model: str):
        try:
            print(f"开始处理工作流请求 {request_data['requestId']}")
            
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
            sanitized_workflow, warnings, errors = validator.sanitize(workflow_structure)
            
            if sanitized_workflow is None:
                # 创建失败结果
                error_result = WorkflowResult(
                    requestId=request_data["requestId"],
                    status="error",
                    error_message=f"工作流结构校验失败: {', '.join(errors)}"
                )
                
                # 缓存失败结果
                with self.completed_requests_lock:
                    self.completed_requests[request_data["requestId"]] = error_result
                
                # 立即推送失败结果
                await self._push_single_result_to_java(error_result)
                return
            
            if warnings:
                print(f"工作流校验警告: {', '.join(warnings)}")
            
            # 4. 创建成功结果
            result = WorkflowResult(
                requestId=request_data["requestId"],
                status="success",
                conversation_id=new_conversation_id,
                workflow_info=sanitized_workflow["workflow_info"],
                nodes=sanitized_workflow["nodes"]
            )
            
            # 缓存成功结果
            with self.completed_requests_lock:
                self.completed_requests[request_data["requestId"]] = result
            
            # 立即推送成功结果
            await self._push_single_result_to_java(result)
            
            print(f"工作流请求 {request_data['requestId']} 处理成功")
            
        except Exception as e:
            error_msg = f"工作流生成失败: {str(e)}"
            print(f"处理工作流请求 {request_data['requestId']} 时发生错误: {error_msg}")
            
            # 创建失败结果
            error_result = WorkflowResult(
                requestId=request_data["requestId"],
                status="error",
                error_message=error_msg
            )
            
            # 缓存失败结果
            with self.completed_requests_lock:
                self.completed_requests[request_data["requestId"]] = error_result
            
            # 立即推送失败结果
            await self._push_single_result_to_java(error_result)

    async def _push_single_result_to_java(self, result: WorkflowResult):
        """推送工作流结果到Java后端"""
        try:
            callback_url = "http://localhost:7003/llm/result/experiment"
            
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

    def add_request(self, request: WorkflowGenerationRequest) -> int:
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

        if request.model not in self.queues:
            print(f"创建新的工作流队列和工作线程: {request.model}")
            self.queues[request.model] = Queue()
            self.processing_count[request.model] = 0
            self.start_worker(request.model)

        self.queues[request.model].put(request_data)
        queue_size = self.queues[request.model].qsize()
        
        print(f"工作流请求 {request.requestId} 已添加到 {request.model} 队列，当前队列长度: {queue_size}")

    def get_queue_position(self, model: str) -> int:
        return self.queues.get(model, Queue()).qsize()

    def get_result(self, request_id: str) -> Optional[WorkflowResult]:
        with self.completed_requests_lock:
            return self.completed_requests.get(request_id)

    def clear_all_results(self):
        with self.completed_requests_lock:
            count = len(self.completed_requests)
            self.completed_requests.clear()
            return count

    def stop(self):
        self.running = False


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
                "file_content1": ""  # 添加默认的文件内容参数
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
            
            # 处理节点数据，适配新格式
            for i, node in enumerate(workflow_data["nodes"]):
                if not isinstance(node, dict):
                    raise ValueError(f"节点{i}不是字典类型，而是: {type(node)}")
                
                # 确保必要字段存在
                if "id" not in node:
                    node["id"] = f"node_{i}"
                
                if "mark" not in node:
                    node["mark"] = str(i)  # 使用字符串类型的mark
                
                if "position" not in node:
                    node["position"] = [100 + i * 200, 100]
                
                # 处理position字段格式
                if isinstance(node["position"], dict):
                    if "x" in node["position"] and "y" in node["position"]:
                        node["position"] = [node["position"]["x"], node["position"]["y"]]
                
                if "name" not in node:
                    node["name"] = node.get("id", f"node_{i}")
                
                # 初始化属性列表
                node.setdefault("simpleAttributes", [])
                node.setdefault("complicatedAttributes", [])
                node.setdefault("inputAnchors", [])
                node.setdefault("outputAnchors", [])
                
                # 确保锚点是列表类型
                if not isinstance(node["inputAnchors"], list):
                    node["inputAnchors"] = []
                if not isinstance(node["outputAnchors"], list):
                    node["outputAnchors"] = []
                
                # 处理inputAnchors
                for j, input_anchor in enumerate(node["inputAnchors"]):
                    if isinstance(input_anchor, dict):
                        # 添加seq字段
                        input_anchor.setdefault("seq", j)
                        input_anchor.setdefault("numOfConnectedEdges", 0)
                        
                        #  从sourceAnchors转换为sourceAnchor
                        if "sourceAnchors" in input_anchor and input_anchor["sourceAnchors"]:
                            if isinstance(input_anchor["sourceAnchors"], list) and len(input_anchor["sourceAnchors"]) > 0:
                                old_source = input_anchor["sourceAnchors"][0]
                                input_anchor["sourceAnchor"] = {
                                    "nodeName": old_source.get("nodeName", old_source.get("id", "")),
                                    "nodeMark": old_source.get("nodeMark", old_source.get("mark", 0)),
                                    "seq": old_source.get("seq", 0)
                                }
                            input_anchor.pop("sourceAnchors", None)
                        
                        # 确保sourceAnchor包含所有必需字段
                        if "sourceAnchor" in input_anchor and input_anchor["sourceAnchor"]:
                            source_anchor = input_anchor["sourceAnchor"]
                            source_anchor.setdefault("seq", 0)
                            # 确保nodeMark是整数类型
                            if "nodeMark" in source_anchor:
                                try:
                                    source_anchor["nodeMark"] = int(source_anchor["nodeMark"])
                                except (ValueError, TypeError):
                                    source_anchor["nodeMark"] = 0
                            
                            # 更新numOfConnectedEdges
                            input_anchor["numOfConnectedEdges"] = 1 if input_anchor.get("sourceAnchor") else 0
                
                # 处理outputAnchors
                for j, output_anchor in enumerate(node["outputAnchors"]):
                    if isinstance(output_anchor, dict):
                        # 添加seq字段
                        output_anchor.setdefault("seq", j)
                        output_anchor.setdefault("numOfConnectedEdges", 0)
                        output_anchor.setdefault("targetAnchors", [])
                        
                        # 确保targetAnchors中的每个元素都有正确的格式和seq字段
                        for k, target_anchor in enumerate(output_anchor["targetAnchors"]):
                            if isinstance(target_anchor, dict):
                                target_anchor.setdefault("nodeName", target_anchor.get("id", ""))
                                target_anchor.setdefault("seq", k)
                                
                                # 处理nodeMark字段，从mark字段转换或设置默认值
                                if "nodeMark" not in target_anchor:
                                    target_anchor["nodeMark"] = target_anchor.get("mark", 0)
                                
                                # 确保nodeMark是整数类型
                                try:
                                    target_anchor["nodeMark"] = int(target_anchor["nodeMark"])
                                except (ValueError, TypeError):
                                    target_anchor["nodeMark"] = 0
                                
                                target_anchor.pop("mark", None)
                                target_anchor.pop("id", None)
                        
                        # 更新numOfConnectedEdges为实际的目标锚点数量
                        output_anchor["numOfConnectedEdges"] = len(output_anchor.get("targetAnchors", []))
            
            return workflow_data
        else:
            raise ValueError("LLM响应中未找到有效的JSON结构")
            
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON解析失败: {e}")
    except Exception as e:
        raise ValueError(f"解析LLM响应失败: {e}")


# 创建全局工作流队列管理器
workflow_queue_manager = WorkflowQueueManager()

@app.on_event("startup")
async def startup_event():
    """服务启动时注册到Consul"""
    SERVICE_PORT = 8004
    service_id = register_service(SERVICE_PORT)
    if service_id:
        app.state.service_id = service_id
        print(f"workflow_generator服务已注册到Consul，服务ID: {service_id}")

@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭时从Consul注销"""
    workflow_queue_manager.stop()  # 停止队列管理器
    
    if hasattr(app.state, 'service_id'):
        deregister_service(app.state.service_id)

@app.post("/workflow/generate", response_model=AsyncWorkflowResponse)
async def generate_workflow(request: WorkflowGenerationRequest):
    try:
        if not request.user_prompt:
            raise HTTPException(status_code=400, detail="user_prompt cannot be empty")
        
        print(f"收到工作流生成请求: {request.requestId}, 模型: {request.model}")
        
        # 添加请求到队列
        workflow_queue_manager.add_request(request)
        
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
    result = workflow_queue_manager.get_result(request_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"未找到请求ID为 {request_id} 的工作流结果")
    
    return result

@app.delete("/workflow/results")
async def clear_all_workflow_results():
    """清空所有工作流结果缓存（用于测试）"""
    count = workflow_queue_manager.clear_all_results()
    return {"message": f"已清空所有工作流结果缓存，共删除 {count} 条记录"}

@app.get("/workflow/queue/status")
async def get_workflow_queue_status():
    """获取工作流队列状态（用于测试）"""
    total_queue_length = 0
    total_processing_count = 0
    
    for model, queue in workflow_queue_manager.queues.items():
        total_queue_length += queue.qsize()
        total_processing_count += workflow_queue_manager.processing_count[model]
    
    status = {
        "queueLength": total_queue_length,
        "processingCount": total_processing_count,
        "modelQueues": {
            model: {
                "queueLength": queue.qsize(),
                "processingCount": workflow_queue_manager.processing_count[model]
            } for model, queue in workflow_queue_manager.queues.items()
        }
    }
    
    print(f"工作流队列状态: {status}")
    return status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)

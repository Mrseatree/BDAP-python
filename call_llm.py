from config import CONSUL_HOST, CONSUL_PORT, SERVICE_NAME
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from consul_utils import register_service, deregister_service
from enum import Enum
import httpx
from typing import Optional
import consul
import socket
import os
import atexit


# 例子，后续需要将其中的模型名字进行规范
class ModelName(str, Enum):
    silicon_flow = "silicon-flow"
    OpenAI = "OpenAI"
    moonshot = "moonshot"


MODEL_TO_APIKEY = {
    "OpenAI": "app-dsGpzD1RooaIGsnroHR70NR1",
    "silicon-flow": "app-9A6JzkmmfNH6uY2o5aQFyIR0",
    "moonshot": "app-k8YszVVuxK8ep6Dj1YKRv20Q"
}

dify_url = "https://api.dify.ai/v1/chat-messages"


class ChatRequest(BaseModel):
    model: ModelName
    question: str
    requestId: str


class ChatResponse(BaseModel):
    answer: str
    requestId: str


app = FastAPI()


# 添加服务启动和关闭事件

@app.on_event("startup")
async def startup_event():
    """服务启动时注册到Consul"""

    service_id = start_call_llm_service()

    if service_id:
        app.state.service_id = service_id

        print(f"call_llm服务已注册到Consul，服务ID: {service_id}")


@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭时从Consul注销"""

    if hasattr(app.state, 'service_id'):
        deregister_service(app.state.service_id)


def start_call_llm_service():
    SERVICE_PORT = 8000
    tags = ['llm', 'ai', 'dify']
    service_id = register_service(SERVICE_PORT, tags)
    # 这里可以添加更多服务启动后的逻辑，比如启动FastAPI应用等
    return service_id


# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": SERVICE_NAME}


async def call_dify(model: str, question: str) -> str:
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
        "conversation_id": None  # 若有上下文则填
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


# 接口的返回值应当符合ChatResponse的Pydantic模型结构
@app.post("/llm", response_model=ChatResponse)
async def get_model(request: ChatRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="question 不能为空")

    answer = await call_dify(request.model, request.question)
    return ChatResponse(answer=answer, requestId=request.requestId)


if __name__ == "__main__":
    import uvicorn

    # 注册服务到Consul
    service_id = register_service()

    # 程序退出时注销服务
    if service_id:
        atexit.register(deregister_service, service_id)

    SERVICE_PORT = 8000
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)


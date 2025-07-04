from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
import httpx
from typing import Optional


# 例子，后续需要将其中的模型名字进行规范
class ModelName(str, Enum):
    deepseek = "deepseek"
    chatGPT = "chatGPT"
    tongyi = "tongyi"


MODEL_TO_APIKEY = {
    "chatGPT":"sk-xxxxxxxxxx-chatgpt",
    "deepseek":"app-ApKEHa7JA4vqjwEFxbzjXAZq",
    "tongyi":"sk-zzzzzzzzzz-tongyi"
}

dify_url = "https://api.dify.ai/v1/chat-messages"


class ChatRequest(BaseModel):
    model: ModelName
    question: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str


app = FastAPI()


async def call_dify(model: str, question: str) -> str:
    api_key = MODEL_TO_APIKEY.get(model)

    if not api_key:
        return f"[error]模型{model}未配置API KEY"

    headers = {
        "Authorization":f"Bearer {api_key}",
        "Content-Type":"application/json"
    }

    data = {
        "inputs":{},
        "query":question,
        "response_mode":"blocking",
        "user":"test-user-id",
        "conversation_id":None  # 若有上下文则填
    }

    timeout = httpx.Timeout(120.0, read = 120.0, connect = 10.0)

    try:
        async with httpx.AsyncClient(timeout = timeout) as client:
            resp = await client.post(dify_url, headers = headers, json = data)

            print("状态码:",resp.status_code)
            print("原始内容:",resp.text)

            if resp.status_code == 504:
                return "[Dify错误] Gateway Timeout（504），模型响应超时，稍后再试"

            try:
                result = resp.json()  # 只在成功时赋值
            except Exception as e:
                return f"[响应格式错误]无法解析JSON:{e}\n原始响应:{resp.text}"

            if "answer" in result:
                return result["answer"]
            elif "message" in result:
                return f"[Dify错误] {result['message']}"
            else:
                return "[Dify响应格式异常]"

    except httpx.ReadTimeout:
        return "[超时] Dify 响应超时"
    except httpx.RequestError as e:
        return f"[请求失败] {e}"
    except Exception as e:
        return f"[未知错误] {e}"


# 接口的返回值应当符合ChatResponse的Pydantic模型结构
@app.post("/models", response_model = ChatResponse)
async def get_model(request: ChatRequest):
    if not request.question:
        return ChatResponse(answer = "question不能为空")
    # print(f"Received request: model={request.model}, question={request.question}")

    # fake_answer = f"[{request.model}] 你说的是：{request.question}"
    # return ChatResponse(answer = fake_answer)
    answer = await call_dify(request.model, request.question)
    return ChatResponse(answer = answer)

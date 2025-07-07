import subprocess

# 启动 call_llm（绑定8000端口）
subprocess.Popen(["uvicorn", "call_llm:app", "--host", "localhost", "--port", "8000"])

# 启动 queue_service（绑定8001端口）
subprocess.Popen(["uvicorn", "queue_service:app", "--host", "localhost", "--port", "8001"])

# 阻塞等待，让脚本不退出
import time
while True:
    time.sleep(1)

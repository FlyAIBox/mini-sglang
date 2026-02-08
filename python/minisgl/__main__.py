
from .server import launch_server

assert __name__ == "__main__"


# Mini-SGLang 主程序入口
#
# 当通过 `python -m minisgl` 运行时，会执行此文件。
# 它负责启动标准的 API Server，对外提供兼容 OpenAI 格式的 HTTP 接口。
#
# 默认配置下：
# - 启动 Uvicorn HTTP 服务器
# - 监听端口 (默认 8000)
# - 自动初始化后台推理引擎和相关组件
launch_server()


from .server import launch_server

if __name__ == "__main__":
    """
    启动交互式 Shell 模式
    
    允许用户直接在终端与模型进行交互，方便调试和测试。
    """
    launch_server(run_shell=True)

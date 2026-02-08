
from .server import launch_server


if __name__ == "__main__":
    # 进入交互式 Shell 模式
    #
    # 该脚本允许用户在命令行直接启动 Mini-SGLang 服务并进入交互式对话环境。
    # 相比于启动 API Server，Shell 模式更适合开发调试和快速体验模型效果。
    #
    # 启动流程：
    # 1. 初始化 Mini-SGLang 服务 (Backend, Tokenizer等)
    # 2. 启动一个简单的 REPL (Read-Eval-Print Loop) 界面
    # 3. 用户输入 Prompt，系统实时流式输出生成的回复
    launch_server(run_shell=True)

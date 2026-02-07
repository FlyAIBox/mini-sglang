
from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from minisgl.distributed import DistributedInfo
from minisgl.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    """
    运行调度器进程 (Backend)
    
    这是每个 tensor parallel rank 的主循环。它会创建一个 Scheduler 实例并运行其事件循环。
    """
    import torch
    from minisgl.scheduler import Scheduler

    with torch.inference_mode():
        # 初始化调度器
        scheduler = Scheduler(args)
        # 等待所有rank初始化完成
        scheduler.sync_all_ranks()

        # 只有 rank 0 向主进程发送 ready 信号
        if args.tp_info.is_primary():
            ack_queue.put("Scheduler is ready")

        if args.silent_output:
            logging.disable(logging.INFO)

        try:
            # 进入调度循环
            scheduler.run_forever()
        except KeyboardInterrupt:
            logger = init_logger(__name__)
            if scheduler.tp_info.is_primary():
                print()  # for a clean newline after ^C
                logger.info("Scheduler exiting gracefully...")
            scheduler.shutdown()


def launch_server(run_shell: bool = False) -> None:
    """
    启动整个服务器系统
    
    协调启动各个组件：
    1. 解析命令行参数
    2. 定义启动后端子进程的函数
    3. 启动 HTTP API Server (Frontend)，并在其中回调启动后端
    """
    from .api_server import run_api_server
    from .args import parse_args

    # 解析参数 (run_shell 可能由命令行参数覆盖)
    server_args, run_shell = parse_args(sys.argv[1:], run_shell)
    logger = init_logger(__name__, "initializer")

    def start_subprocess() -> None:
        """
        启动后端子进程的回调函数
        
        包括:
        - N 个 Scheduler 进程 (用于 Tensor Parallelism)
        - 1 个 Detokenizer 进程
        - M 个 Tokenizer 进程
        """
        import multiprocessing as mp

        from minisgl.tokenizer import tokenize_worker

        # 使用 'spawn' 模式启动进程，确保 CUDA 上下文隔离
        mp.set_start_method("spawn", force=True)

        world_size = server_args.tp_info.size
        # 用于接收子进程启动确认的队列
        ack_queue: mp.Queue[str] = mp.Queue()

        # 1. 启动 Scheduler 进程 (每个 GPU 一个)
        for i in range(world_size):
            new_args = replace(
                server_args,
                tp_info=DistributedInfo(i, world_size),
            )
            mp.Process(
                target=_run_scheduler,
                args=(new_args, ack_queue),
                daemon=False,
                name=f"minisgl-TP{i}-scheduler",
            ).start()

        num_tokenizers = server_args.num_tokenizer
        
        # 2. 启动 Detokenizer 进程 (总是 1 个)
        # 如果 num_tokenizer == 0，它也负责 Tokenize
        mp.Process(
            target=tokenize_worker,
            kwargs={
                "tokenizer_path": server_args.model_path,
                "addr": server_args.zmq_detokenizer_addr,
                "backend_addr": server_args.zmq_backend_addr,
                "frontend_addr": server_args.zmq_frontend_addr,
                "local_bs": 1,
                "create": server_args.tokenizer_create_addr,
                "tokenizer_id": num_tokenizers, # ID 用于区分和日志
                "ack_queue": ack_queue,
            },
            daemon=False,
            name="minisgl-detokenizer-0",
        ).start()
        
        # 3. 启动额外的 Tokenizer 进程 (可选)
        for i in range(num_tokenizers):
            mp.Process(
                target=tokenize_worker,
                kwargs={
                    "tokenizer_path": server_args.model_path,
                    "addr": server_args.zmq_tokenizer_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "frontend_addr": server_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": i,
                    "ack_queue": ack_queue,
                },
                daemon=False,
                name=f"minisgl-tokenizer-{i}",
            ).start()

        # 等待所有工作进程的确认信号:
        # - 1个 scheduler ack (来自 primary rank)
        # - num_tokenizers 个 tokenizer acks
        # - 1个 detokenizer ack
        # 总共需要的 acks: num_tokenizers + 2
        for _ in range(num_tokenizers + 2):
            logger.info(ack_queue.get())

    # 启动 API Server (Frontend)，它会调用 start_subprocess 启动后端
    run_api_server(server_args, start_subprocess, run_shell=run_shell)


if __name__ == "__main__":
    launch_server()

"""
测试调度器（Scheduler）模块

本测试用于验证 Scheduler 的基本功能，包括：
1. Scheduler 进程的启动和初始化
2. 消息队列的通信（ZMQ）
3. Prompt 的 Tokenization和推理
4. 生成结果的接收和解码

测试流程：
1. 在独立进程中启动 Scheduler
2. 通过 ZMQ 队列发送用户请求（包含 input_ids 和采样参数）
3. 循环接收生成的 token，直到完成
4. 验证整个流程的正确性
"""

from __future__ import annotations

import torch
import multiprocessing as mp
from transformers import AutoTokenizer

from minisgl.distributed import DistributedInfo
from minisgl.message import BaseBackendMsg, BaseTokenizerMsg, DetokenizeMsg, ExitMsg, UserMsg
from minisgl.scheduler import Scheduler, SchedulerConfig
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, call_if_main, init_logger
from minisgl.core import SamplingParams

logger = init_logger(__name__)


@torch.inference_mode()
def scheduler(config: SchedulerConfig, queue: mp.Queue) -> None:
    """
    调度器工作进程的入口函数
    
    在独立进程中启动 Scheduler，处理推理请求。
    使用 multiprocessing.Queue 进行进程间同步，通知主进程初始化完成。
    
    Args:
        config: Scheduler 配置对象
        queue: 多进程队列，用于通知主进程 Scheduler 已就绪
    """
    scheduler = Scheduler(config)
    # 通知主进程：Scheduler 已初始化完成
    queue.put(None)
    try:
        # 进入事件循环，持续处理请求
        scheduler.run_forever()
    except KeyboardInterrupt:
        logger.info_rank0("Scheduler exiting...")


@call_if_main(__name__)
def main():
    """
    主测试函数
    
    执行完整的端到端测试流程：
    1. 配置并启动 Scheduler 进程
    2. 设置与 Scheduler 的通信通道（ZMQ）
    3. 发送测试 Prompt
    4. 接收并打印生成结果
    """
    # ========================================
    # 1. 配置 Scheduler
    # ========================================
    config = SchedulerConfig(
        model_path="meta-llama/Llama-3.1-8B-Instruct",  # 使用的模型路径
        tp_info=DistributedInfo(0, 1),  # TP Rank 0, 总共 1 个 GPU
        dtype=torch.bfloat16,  # 使用 bfloat16 精度（节省显存，保持精度）
        max_running_req=4,  # 最大并发请求数
        cuda_graph_bs=[2, 4, 8],  # CUDA Graph 批处理大小配置
    )

    # ========================================
    # 2. 启动 Scheduler 进程
    # ========================================
    # 使用 spawn 方式启动子进程（更安全，避免 CUDA 初始化冲突）
    mp.set_start_method("spawn", force=True)
    q = mp.Queue()
    p = mp.Process(target=scheduler, args=(config, q))
    p.start()
    # 等待 Scheduler 初始化完成
    q.get()

    # ========================================
    # 3. 设置与 Scheduler 的通信通道
    # ========================================
    # 发送队列：向 Scheduler 发送用户请求
    send_backend = ZmqPushQueue(
        config.zmq_backend_addr,  # Scheduler 监听的地址
        create=False,  # 不创建 socket（Scheduler 已创建）
        encoder=BaseBackendMsg.encoder,  # 消息编码器
    )

    # 接收队列：从 Scheduler 接收生成的 token
    recv_backend = ZmqPullQueue(
        config.zmq_detokenizer_addr,  # Scheduler 发送结果的地址
        create=False,
        decoder=BaseTokenizerMsg.decoder,  # 消息解码器
    )

    # ========================================
    # 4. 准备测试 Prompt
    # ========================================
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    prompt = "What's the answer to life, the universe, and everything?"  # 经典测试问题
    # 将 Prompt 编码为 token IDs（int32 格式）
    ids = tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
    
    # 发送用户请求
    send_backend.put(
        UserMsg(
            uid=0,  # 请求唯一 ID
            input_ids=ids,  # 输入 token IDs
            sampling_params=SamplingParams(max_tokens=100),  # 采样参数：最多生成 100 个 token
        )
    )

    # ========================================
    # 5. 接收生成结果
    # ========================================
    while True:
        msg = recv_backend.get()
        assert isinstance(msg, DetokenizeMsg)  # 验证消息类型
        # 将新生成的 token 追加到 ids 中
        ids = torch.cat([ids, torch.tensor([msg.next_token], dtype=torch.int32)])
        if msg.finished:  # 检查是否生成完成（遇到 EOS 或达到 max_tokens）
            break

    # ========================================
    # 6. 打印完整结果
    # ========================================
    logger.info(tokenizer.decode(ids.tolist()))
    
    # 发送退出消息，优雅关闭 Scheduler
    send_backend.put(ExitMsg())


from __future__ import annotations

import multiprocessing as mp
from typing import List

import torch
from minisgl.message import (
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchBackendMsg,
    BatchFrontendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
)
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, init_logger
from transformers import AutoTokenizer, LlamaTokenizer


def _unwrap_msg(msg: BaseTokenizerMsg) -> List[BaseTokenizerMsg]:
    """如果消息是批量的，解包成单个消息列表"""
    if isinstance(msg, BatchTokenizerMsg):
        return msg.data
    return [msg]


@torch.inference_mode()
def tokenize_worker(
    *,
    tokenizer_path: str,
    addr: str,
    create: bool,
    backend_addr: str,
    frontend_addr: str,
    local_bs: int,
    tokenizer_id: int = -1,
    ack_queue: mp.Queue[str] | None = None,
) -> None:
    """
    Tokenizer 工作进程
    
    独立进程运行，负责处理 Tokenize (文本转ID) 和 Detokenize (ID转文本) 请求。
    使用 ZeroMQ 进行进程间通信，减轻主进程负担。
    
    参数:
        tokenizer_path: 模型路径 (包含 tokenizer 配置)
        addr: 接收请求的 ZMQ 地址
        create: 是否创建 ZMQ socket (bind vs connect)
        backend_addr: 发送给 Backend (Engine) 的 ZMQ 地址
        frontend_addr: 发送给 Frontend (API Server) 的 ZMQ 地址
        local_bs: 本地批处理大小
        tokenizer_id: 进程 ID 用于日志
        ack_queue: 启动确认队列
    """
    # 初始化 ZMQ 队列
    send_backend = ZmqPushQueue(backend_addr, create=False, encoder=BaseBackendMsg.encoder)
    send_frontend = ZmqPushQueue(frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)
    recv_listener = ZmqPullQueue(addr, create=create, decoder=BatchTokenizerMsg.decoder)
    
    assert local_bs > 0
    # 加载 HuggingFace Tokenizer
    tokenizer: LlamaTokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    logger = init_logger(__name__, f"tokenizer_{tokenizer_id}")

    from .detokenize import DetokenizeManager
    from .tokenize import TokenizeManager

    tokenize_manager = TokenizeManager(tokenizer)
    detokenize_manager = DetokenizeManager(tokenizer)

    if ack_queue is not None:
        ack_queue.put(f"Tokenize server {tokenizer_id} is ready")

    try:
        while True:
            # 1. 接收消息
            pending_msg = _unwrap_msg(recv_listener.get())
            # 尝试获取更多消息以形成 batch，提高效率
            while len(pending_msg) < local_bs and not recv_listener.empty():
                pending_msg.extend(_unwrap_msg(recv_listener.get()))

            logger.debug(f"Received {len(pending_msg)} messages")

            # 2. 分类处理 (Detokenize vs Tokenize)
            detokenize_msg = [m for m in pending_msg if isinstance(m, DetokenizeMsg)]
            tokenize_msg = [m for m in pending_msg if isinstance(m, TokenizeMsg)]
            assert len(detokenize_msg) + len(tokenize_msg) == len(pending_msg)
            
            # 3. 处理 Detokenize 请求 (解码 Generated Token IDs -> 文本)
            if len(detokenize_msg) > 0:
                replies = detokenize_manager.detokenize(detokenize_msg)
                # 构造响应发回前端 (Streaming response)
                batch_output = BatchFrontendMsg(
                    data=[
                        UserReply(
                            uid=msg.uid,
                            incremental_output=reply,
                            finished=msg.finished,
                        )
                        for msg, reply in zip(detokenize_msg, replies, strict=True)
                    ]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_frontend.put(batch_output)

            # 4. 处理 Tokenize 请求 (Prompt 文本 -> Token IDs)
            if len(tokenize_msg) > 0:
                tensors = tokenize_manager.tokenize(tokenize_msg)
                # 构造请求发给后端 Engine (进行 Prefill/Generation)
                batch_output = BatchBackendMsg(
                    data=[
                        UserMsg(
                            uid=msg.uid,
                            input_ids=t,
                            sampling_params=msg.sampling_params,
                        )
                        for msg, t in zip(tokenize_msg, tensors, strict=True)
                    ]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_backend.put(batch_output)
    except KeyboardInterrupt:
        pass

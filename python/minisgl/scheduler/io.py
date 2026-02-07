
from __future__ import annotations

from typing import TYPE_CHECKING, Final, List

import torch
from minisgl.message import BaseBackendMsg, BaseTokenizerMsg, BatchTokenizerMsg, DetokenizeMsg
from minisgl.utils import ZmqPubQueue, ZmqPullQueue, ZmqPushQueue, ZmqSubQueue, init_logger

if TYPE_CHECKING:
    from .config import SchedulerConfig

logger = init_logger(__name__)


class SchedulerIOMixin:
    """
    调度器 I/O 混合类 (Mixin)
    
    处理调度器与 Tokenizer 以及其他 Rank 之间的通信。
    
    包含两种模式：
    1. 离线模式 (Offline Mode): 直接调用方法，不使用 ZMQ。
    2. 在线模式 (Online Mode): 使用 ZeroMQ 进行进程间通信 (IPC)。
    
    通信链路：
    - Input: Tokenizer -> Scheduler (ZMQ PULL)
    - Output: Scheduler -> Detokenizer (ZMQ PUSH)
    - Sync: Rank 0 -> Other Ranks (ZMQ PUB/SUB)
    """

    def __init__(self, config: SchedulerConfig, tp_cpu_group: torch.distributed.ProcessGroup):
        tp_info = config.tp_info
        self.tp_cpu_group: Final = tp_cpu_group
        if config.offline_mode:
            # 离线模式：替换接收和发送方法为离线版本
            self.receive_msg = self.offline_receive_msg
            self.send_result = self.offline_send_result
            return  # early exit

        if tp_info.is_primary():
            # Rank 0 (Primary) 负责主要的外部通信
            self._recv_from_tokenizer: Final = ZmqPullQueue(
                config.zmq_backend_addr,
                create=True,
                decoder=BaseBackendMsg.decoder,
            )
            self._send_into_tokenizer: Final = ZmqPushQueue(
                config.zmq_detokenizer_addr,
                create=config.backend_create_detokenizer_link,
                encoder=BaseTokenizerMsg.encoder,
            )

        recv = self._recv_msg_single_rank
        send = self._reply_tokenizer_rank0
        if tp_info.size > 1:
            # 分布式模式 (Tensor Parallelism)
            if tp_info.is_primary():
                # Rank 0: 接收外部请求，并广播给其他 Rank
                recv = self._recv_msg_multi_rank0
                self._send_into_ranks: Final = ZmqPubQueue(
                    config.zmq_scheduler_broadcast_addr, create=True, encoder=BaseBackendMsg.encoder
                )
            else:
                # Rank 1+: 接收 Rank 0 的广播
                recv = self._recv_msg_multi_rank1
                send = self._reply_tokenizer_rank1
                self._recv_from_rank0: Final = ZmqSubQueue(
                    config.zmq_scheduler_broadcast_addr,
                    create=False,
                    decoder=BaseBackendMsg.decoder,
                )

        self.receive_msg = recv
        self.send_result = send

    def run_when_idle(self):
        """
        空闲时执行的任务 (抽象方法)
        
        子类应实现此方法，在等待消息时执行一些后台任务 (e.g. 整理内存)。
        """
        raise NotImplementedError("should be implemented")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """离线模式接收消息接口 (应由子类实现)"""
        raise NotImplementedError("should be implemented")

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        """离线模式发送结果接口 (应由子类实现)"""
        raise NotImplementedError("should be implemented")

    def sync_all_ranks(self) -> None:
        """同步所有 CPU 进程"""
        self.tp_cpu_group.barrier().wait()

    def _recv_msg_single_rank(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """单 Rank 模式：从 Tokenizer 接收消息"""
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            # 阻塞模式：等待至少一条消息
            self.run_when_idle()
            pending_msgs.append(self._recv_from_tokenizer.get())
        # 非阻塞模式：获取所有当前可用的消息
        while not self._recv_from_tokenizer.empty():
            pending_msgs.append(self._recv_from_tokenizer.get())
        return pending_msgs

    def _recv_msg_multi_rank0(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """多 Rank 模式 (Rank 0): 从 Tokenizer 接收并广播"""
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            raw = self._recv_from_tokenizer.get_raw()
            # 收到消息后立即广播给其他 Rank
            self._send_into_ranks.put_raw(raw)
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))

        pending_raw_msgs: List[bytes] = []
        while not self._recv_from_tokenizer.empty():
            pending_raw_msgs.append(self._recv_from_tokenizer.get_raw())

        # 广播本次接收的消息数量，确保各 Rank 同步
        src_tensor = torch.tensor(len(pending_raw_msgs))
        self.tp_cpu_group.broadcast(src_tensor, root=0).wait()

        # 广播具体消息内容
        for raw in pending_raw_msgs:
            self._send_into_ranks.put_raw(raw)
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))
        return pending_msgs

    def _recv_msg_multi_rank1(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """多 Rank 模式 (Rank 1+): 接收 Rank 0 的广播"""
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            pending_msgs.append(self._recv_from_rank0.get())

        # 接收本次应接收的消息数量
        dst_tensor = torch.tensor(-1)
        self.tp_cpu_group.broadcast(dst_tensor, root=0).wait()
        dst_length = int(dst_tensor.item())

        # 接收具体消息
        for _ in range(dst_length):
            pending_msgs.append(self._recv_from_rank0.get())
        return pending_msgs

    def _reply_tokenizer_rank0(self, reply: List[DetokenizeMsg]) -> None:
        """Rank 0: 将生成的 token 发送回 Tokenizer (进行 Detokenization)"""
        num_reply = len(reply)
        logger.debug_rank0(f"Replying to tokenizer: {num_reply} messages")
        if num_reply == 1:
            self._send_into_tokenizer.put(reply[0])
        elif num_reply > 1:
            # 批量发送，减少通信开销
            self._send_into_tokenizer.put(BatchTokenizerMsg(data=reply))  # type: ignore

    def _reply_tokenizer_rank1(self, reply: List[DetokenizeMsg]) -> None:
        """Rank 1+: 不发送任何结果 (仅 Rank 0 负责输出)"""
        _ = reply  # do nothing for non-primary ranks


from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.message import (
    BaseBackendMsg,
    DetokenizeMsg,
    UserMsg,
)
from minisgl.scheduler import Scheduler, SchedulerConfig



class RequestAllFinished(Exception):
    """自定义异常，表示所有等待中的请求都已处理完成。"""
    pass


@dataclass
class RequestStatus:
    """
    请求状态数据类。

    用于跟踪单个生成请求的进度和结果，包括输入 prompt、生成的 token 等信息。
    """
    uid: int
    """请求的唯一标识符。"""
    input_ids: List[int]
    """输入的 Token ID 列表。"""
    output_ids: List[int]
    """生成的输出 Token ID 列表。"""


class LLM(Scheduler):
    """
    LLM (Large Language Model) 高级接口类。

    该类主要用于离线批处理推理（Offline Batch Inference）场景。
    它继承自 `Scheduler`，直接在本地管理请求的调度和生成过程，
    绕过了通常用于服务的 API Server 和 ZMQ 通信层，从而减少了开销并简化了调试和离线使用流程。
    """
    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, **kwargs):
        """
        初始化 LLM 实例。

        Args:
            model_path: 模型权重的路径 (HuggingFace 格式)。
            dtype: 模型加载的数据类型，默认为 bfloat16。
            **kwargs: 传递给 SchedulerConfig 的其他参数。
        """
        config = SchedulerConfig(
            model_path=model_path,
            tp_info=DistributedInfo(0, 1),
            dtype=dtype,
            offline_mode=True, # 启用离线模式，跳过网络通信相关的初始化
            **kwargs,
        )
        super().__init__(config)
        self.pending_requests: List[Tuple[List[int] | str, SamplingParams]] = []
        self.status_map: Dict[int, RequestStatus] = {}
        self.counter = 0

    def _tokenize_one(self, prompt: List[int] | str) -> torch.Tensor:
        """
        对单个 prompt 进行 tokenization 处理。

        如果输入是字符串，则使用 tokenizer 进行编码；
        如果输入已经是 token ID 列表，则直接转换为 Tensor。
        """
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
        else:
            return torch.tensor(prompt, dtype=torch.int32, device="cpu")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """
        模拟调度器接收消息的接口 (Override Scheduler 方法)。

        在离线模式下，该方法直接从 `pending_requests` 列表中获取预先添加的请求，
        并将它们封装为 `UserMsg` 返回给调度器，模拟从网络接收请求的过程。
        """
        if blocking and len(self.pending_requests) == 0:
            raise RequestAllFinished()
        results: List[BaseBackendMsg] = []
        added, sum_input_len = 0, 0
        for tokens_or_prompt, sampling_params in self.pending_requests:
            if sum_input_len >= self.prefill_budget:
                break
            input_ids = self._tokenize_one(tokens_or_prompt)
            sum_input_len += len(input_ids)
            uid, added = self.counter + added, added + 1
            results.append(UserMsg(uid=uid, input_ids=input_ids, sampling_params=sampling_params))
            self.status_map[uid] = RequestStatus(
                uid=uid,
                input_ids=(
                    input_ids.tolist() if isinstance(tokens_or_prompt, str) else tokens_or_prompt
                ),
                output_ids=[],
            )
        self.counter += added
        self.pending_requests = self.pending_requests[added:]
        return results

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        """
        模拟调度器发送结果的接口 (Override Scheduler 方法)。

        在离线模式下，该方法接收调度器生成的 `DetokenizeMsg`，
        并将生成的 token 更新到 `status_map` 中对应的请求状态里，
        而不是通过网络发送回客户端。
        """
        for msg in reply:
            status = self.status_map[msg.uid]
            if not (msg.finished and msg.next_token == self.eos_token_id):
                status.output_ids.append(msg.next_token)

    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: List[SamplingParams] | SamplingParams,
    ) -> List[Dict[str, str | List[int]]]:
        """
        执行推理生成。

        这是 LLM 类的主要入口点，用于处理一批 prompts。

        Args:
            prompts: 输入的 prompt 列表，可以是字符串列表或 token ID 列表。
            sampling_params: 采样参数，可以是单个参数对象（应用于所有 prompts）或参数列表（一一对应）。

        Returns:
            List[Dict]: 生成结果列表，每个元素包含:
                - "text": 生成的文本字符串。
                - "token_ids": 生成的 token ID 列表。
        """
        self.pending_requests = []
        self.status_map = {}
        self.counter = 0
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.pending_requests.append((prompt, sp))
        try:
            self.run_forever()
        except RequestAllFinished:
            pass
        results: List[Dict[str, str | List[int]]] = []
        for i in range(len(prompts)):
            status = self.status_map[i]
            output_text = self.tokenizer.decode(status.output_ids)
            results.append({"text": output_text, "token_ids": status.output_ids})
        return results

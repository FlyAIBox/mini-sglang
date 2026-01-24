"""
调度器模块

Scheduler是Mini-SGLang的核心组件，负责：
1. 接收和管理用户请求
2. 决策何时执行prefill和decode
3. 组batch以最大化GPU利用率
4. 调用Engine执行模型forward
5. 处理输出并返回给用户

调度策略：
- Continuous Batching: 动态调整batch，无需等待所有请求完成
- Chunked Prefill: 长输入分块处理，与decode交错执行
- Overlap Scheduling: CPU调度与GPU计算重叠，提高吞吐量
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
import torch.nn.functional as F
from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.message import (
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.utils import init_logger
from transformers import AutoTokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)


class ForwardInput(NamedTuple):
    """
    Forward操作的输入数据
    
    为了支持重叠调度，需要缓存forward所需的所有数据，
    避免数据在准备下一个batch时被覆盖。
    
    属性:
        batch: 要执行的批次
        sample_args: 采样参数
        load_indices: 从token_pool加载token的索引
        write_indices: 将生成的token写回token_pool的索引
    """
    batch: Batch
    sample_args: BatchSamplingArgs
    load_indices: torch.Tensor
    write_indices: torch.Tensor


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin):
    """
    调度器主类
    
    负责整个推理系统的请求调度和资源管理。继承自SchedulerIOMixin
    以获得与Tokenizer/Detokenizer的通信能力。
    
    主要职责:
    1. 请求管理：接收新请求，维护等待队列和运行队列
    2. 批次调度：决定哪些请求组成batch，是prefill还是decode
    3. 资源分配：管理KV Cache、page table等GPU资源
    4. 执行协调：调用Engine执行模型计算
    5. 结果处理：处理输出token，判断是否完成，返回给用户
    
    核心管理器:
    - table_manager: 管理请求表和token池
    - cache_manager: 管理KV Cache（Radix或Naive）
    - prefill_manager: 管理待prefill的请求
    - decode_manager: 管理正在decode的请求
    - engine: 执行模型forward的引擎
    
    调度模式:
    - overlap_loop: 重叠调度模式（默认，高性能）
    - normal_loop: 正常调度模式（用于调试）
    """
    
    def __init__(self, config: SchedulerConfig):
        """
        初始化调度器
        
        参数:
            config: 调度器配置，包含模型路径、最大batch size等参数
        """
        from minisgl.engine import Engine

        # 初始化推理引擎
        self.engine = Engine(config)
        
        # 初始化I/O通信（与Tokenizer/Detokenizer）
        super().__init__(config, self.engine.tp_cpu_group)

        # 创建单独的CUDA stream用于重叠调度
        # self.stream: 用于CPU端的元数据处理
        # self.engine.stream: 用于GPU计算
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # 初始化各个管理器
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(self.device, self.engine.num_pages, config.cache_type)
        self.decode_manager = DecodeManager()
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        # 其他配置
        self.tp_info = config.tp_info
        self.finished_reqs: Set[Req] = set()  # 已完成但可能还在使用中的请求
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.page_table = self.engine.page_table
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        self.dummy_write_2d_pos = (self.engine.dummy_req.table_idx, 1, 2)

    def _process_last_data(
        self, last_data: ForwardData | None, ongoing_data: ForwardData | None
    ) -> None:
        if last_data is None:
            return
        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        reply: List[DetokenizeMsg] = []

        for i, req in enumerate(batch.reqs):
            if req in self.finished_reqs or isinstance(req, ChunkedReq):
                continue

            next_token_id = next_tokens_cpu[i]
            req.append_host(next_token_id.unsqueeze(0))
            next_token = int(next_token_id.item())
            finished = not req.can_decode()
            if not req.sampling_params.ignore_eos:
                finished |= next_token == self.eos_token_id
            reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

            # free resources if the req is finished and not ongoing
            if finished:
                self.finished_reqs.add(req)
                self.decode_manager.remove_req(req)
                logger.debug_rank0("Request %s is finished", req)

        # free resources for finished but not ongoing reqs
        ongoing_reqs = ongoing_data[0].batch.reqs if ongoing_data else []
        for req in self.finished_reqs.difference(ongoing_reqs):
            self.table_manager.free(req.table_idx)
            self.cache_manager.free_and_cache_finished_req(
                req.cache_handle,
                req.input_ids[: req.cached_len],
                self.page_table[req.table_idx, : req.cached_len],
            )

        # keep only ongoing reqs in the finished set
        self.finished_reqs.intersection_update(ongoing_reqs)
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        needed_size = sum(r.extend_len for r in batch.reqs)
        batch.out_loc = self.cache_manager.allocate(needed_size)
        # NOTE: Pad the batch if needed
        if padding_size := self.engine.graph_runner.pad_batch(batch):
            batch.out_loc = F.pad(batch.out_loc, (0, padding_size), value=self.engine.dummy_page)
        # NOTE: prepare 2d indices for token ids loading and writing
        load_indices = self._make_2d_indices(
            [(r.table_idx, r.cached_len, r.device_len) for r in batch.padded_reqs]
        )
        write_indices = self._make_2d_indices(
            [
                (
                    (r.table_idx, r.device_len, r.device_len + 1)
                    if r.can_decode()  # NOTE: for chunked req, write to dummy pos
                    else self.dummy_write_2d_pos
                )
                for r in batch.reqs
            ]
        )
        assert all(r.device_len < self.engine.max_seq_len for r in batch.reqs)
        # NOTE: write out_loc to page_table before `prepare_metadata`
        self.page_table.view(-1)[load_indices] = batch.out_loc
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            load_indices=load_indices,
            write_indices=write_indices,
        )

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        return self._prepare_batch(batch) if batch else None

    def _make_2d_indices(self, ranges: List[Tuple[int, int, int]]) -> torch.Tensor:
        """
        Return the 1D indices for the given 2D table and ranges.

        Example: The underlying indices of a 2D table (3, 4) are:
            [[ 0,  1,  2,  3],
             [ 4,  5,  6,  7],
             [ 8,  9, 10, 11]]
        For ranges [(0, 1, 3), (2, 0, 2)], the returned indices are [1, 2, 8, 9].

        Args:
            ranges (List[Tuple[int, int, int]]): A list of tuples (entry, begin, end),
                where `entry` is the row index in the 2D table, and `begin` and `end`
                specify the range of column indices to include.
        Returns:
            torch.Tensor: A 1D tensor of indices.
        """
        STRIDE = self.token_pool.stride(0)
        needed_size = sum(end - begin for _, begin, end in ranges)
        indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
        offset = 0
        for entry, begin, end in ranges:
            length = end - begin
            offset += length
            torch.arange(
                begin + entry * STRIDE,
                end + entry * STRIDE,
                dtype=torch.int32,
                out=indices_host[offset - length : offset],
            )
        return indices_host.to(self.device, non_blocking=True)

    def _load_token_ids(self, input: ForwardInput) -> None:
        input.batch.input_ids = self.token_pool.view(-1)[input.load_indices]

    def _write_token_ids(self, input: ForwardInput, output: ForwardOutput) -> None:
        self.token_pool.view(-1)[input.write_indices] = output.next_tokens_gpu

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        self._load_token_ids(forward_input)
        batch, sample_args = forward_input.batch, forward_input.sample_args
        if ENV.OVERLAP_EXTRA_SYNC:  # NOTE: https://github.com/sgl-project/mini-sglang/issues/58
            self.stream.synchronize()
        forward_output = self.engine.forward_batch(batch, sample_args)
        self._write_token_ids(forward_input, forward_output)
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        blocking = not (
            last_data  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data, ongoing_data)
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data, None)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

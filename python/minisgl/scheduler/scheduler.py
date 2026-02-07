

"""
调度器模块

Scheduler是Mini-SGLang的核心组件，负责协调整个推理过程。
它实现了 Continuous Batching (持续批处理) 和 Overlap Scheduling (重叠调度) 等关键技术。

主要职责：
1. 接收和管理用户请求 (UserMsg)
2. 决策调度策略 (何时执行Prefill，何时执行Decode)
3. 组装Batch并管理显存资源 (KV Cache)
4. 调用Engine执行模型计算
5. 处理计算结果并返回给用户

调度策略：
- Continuous Batching: 动态地将新请求插入到正在运行的batch中，无需等待所有请求完成。
- Chunked Prefill: 将长Prompt分块处理，避免单次计算量过大导致阻塞。
- Overlap Scheduling: 在GPU计算当前Batch的同时，CPU并行处理上一个Batch的结果和下一个Batch的准备工作。
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
    Forward操作的输入数据封装
    
    为了支持 CPU-GPU 重叠调度 (Overlap Scheduling)，我们需要将执行一次Forward
    所需的所有数据打包。这样即使在准备下一个Batch时，上一个Batch的数据也不会被覆盖。
    
    Attributes:
        batch (Batch): 当前要执行的批次对象，包含请求列表等信息。
        sample_args (BatchSamplingArgs): 采样参数，如 temperature, top_p 等。
        load_indices (torch.Tensor): 1D Tensor，用于从全局Token Pool中加载当前Batch所需的Input Tokens。
        write_indices (torch.Tensor): 1D Tensor，用于将生成的Next Token写回全局Token Pool的指定位置。
    """
    batch: Batch
    sample_args: BatchSamplingArgs
    load_indices: torch.Tensor
    write_indices: torch.Tensor


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin):
    """
    调度器主类 (Scheduler)
    
    负责整个推理系统的核心调度逻辑。继承自 SchedulerIOMixin 以获得与
    Tokenizer (前端) 和 Detokenizer (后端) 的通信能力。
    
    架构要点:
    - 拥有一个独立的 CUDA Stream (self.stream) 用于元数据准备和CPU辅助操作。
    - 与 Engine (负责模型计算) 交互，Engine 拥有计算用的 CUDA Stream。
    - 管理多个子管理器：TableManager (页表), CacheManager (KV缓存), 
      PrefillManager (预填充), DecodeManager (解码).
    """
    
    def __init__(self, config: SchedulerConfig):
        """
        初始化调度器
        
        Args:
            config: 调度器配置对象，包含模型路径、最大请求数、显存配置等。
        """
        from minisgl.engine import Engine

        # 初始化推理引擎
        self.engine = Engine(config)
        
        # 初始化与前端/后端的通信管道
        super().__init__(config, self.engine.tp_cpu_group)

        # 设置多Stream环境以支持Overlap Scheduling
        # self.stream: 用于调度器自身的CUDA操作（如数据拷贝、元数据准备）
        # self.engine.stream: 用于模型推理计算
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # 初始化资源管理器
        # TableManager: 管理全局Token Pool和页表
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        # CacheManager: 管理KV Cache显存块
        self.cache_manager = CacheManager(self.device, self.engine.num_pages, config.cache_type)
        # DecodeManager: 管理解码阶段的请求
        self.decode_manager = DecodeManager()
        # PrefillManager: 管理预填充阶段的请求
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        # 其他辅助对象
        self.tp_info = config.tp_info
        self.finished_reqs: Set[Req] = set()  # 暂存已完成的请求，等待清理
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.page_table = self.engine.page_table
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens  # 每次Prefill允许的最大Token数
        # 用于只需要解码不需要写入的dummy位置 (例如Chunked Prefill的中间步骤)
        self.dummy_write_2d_pos = (self.engine.dummy_req.table_idx, 1, 2)

    def _process_last_data(
        self, last_data: ForwardData | None, ongoing_data: ForwardData | None
    ) -> None:
        """
        处理上一次 Forward 的结果 (CPU Post-processing)
        
        在重叠调度循环中，这个函数在CPU上运行，与此同时GPU正在执行当前Batch的计算。
        
        主要步骤:
        1. 同步等待数据从GPU拷贝回CPU (copy_done event)。
        2. 遍历Batch中的每个请求，获取生成的Token。
        3. 更新请求状态 (append new token)。
        4. 检查停止条件 (EOS token 或 达到最大长度)。
        5. 将结果发送给 Detokenizer。
        6. 回收已完成请求的资源。
        
        Args:
            last_data: 上一个Batch的输入和输出数据。
            ongoing_data: 当前正在GPU上运行的Batch数据 (用于防止回收正在使用的资源)。
        """
        if last_data is None:
            return
        # 解包数据：batch信息，以及 (logits, next_tokens, copy_event)
        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        
        # 必须等待 "GPU->CPU 拷贝" 这一步完成，才能读取 next_tokens_cpu
        copy_done.synchronize()
        reply: List[DetokenizeMsg] = []

        for i, req in enumerate(batch.reqs):
            # 跳过已完成或分块请求（分块请求还没生成有效输出）
            if req in self.finished_reqs or isinstance(req, ChunkedReq):
                continue

            # 获取生成的Token ID
            next_token_id = next_tokens_cpu[i]
            # 将Token追加到请求的CPU端buffer中
            req.append_host(next_token_id.unsqueeze(0))
            next_token = int(next_token_id.item())
            
            # 判断请求是否结束
            finished = not req.can_decode() # 是否达到最大长度
            if not req.sampling_params.ignore_eos:
                finished |= next_token == self.eos_token_id # 是否遇到EOS
            
            # 构造返回消息
            reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

            # 如果请求结束，标记清理
            if finished:
                self.finished_reqs.add(req)
                self.decode_manager.remove_req(req)
                logger.debug_rank0("Request %s is finished", req)

        # 资源回收
        # 注意：只能回收那些 "已完成 且 不在当前正在运行Batch中" 的请求
        ongoing_reqs = ongoing_data[0].batch.reqs if ongoing_data else []
        for req in self.finished_reqs.difference(ongoing_reqs):
            self.table_manager.free(req.table_idx) # 释放页表项
            # 释放KV Cache，如果可能，将其存入RadixCache以供复用
            self.cache_manager.free_and_cache_finished_req(
                req.cache_handle,
                req.input_ids[: req.cached_len],
                self.page_table[req.table_idx, : req.cached_len],
            )

        # 更新finished_reqs，只保留还未完全释放（因为被ongoing batch引用）的请求
        self.finished_reqs.intersection_update(ongoing_reqs)
        # 发送处理结果
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        """处理单条控制消息"""
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            
            # 检查输入长度限制
            if max_output_len <= 0:
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            # 调整最大生成长度
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            # 将新请求加入Prefill队列
            self.prefill_manager.add_one_req(msg)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        """
        为Batch准备执行资源和元数据
        
        步骤：
        1. (Allocate) 为KV Cache分配显存页。
        2. (Pad) 如果需要，对Batch进行Padding（CUDA Graph要求）。
        3. (Indices) 计算Token Loading和Writing的索引。
        4. (Page Table) 更新全局页表。
        5. (Metadata) 准备Attention后端所需的元数据。
        6. (Sample Args) 准备采样参数。
        
        Args:
            batch: 待执行的Batch对象
            
        Returns:
            ForwardInput: 包含所有执行所需数据的对象
        """
        needed_size = sum(r.extend_len for r in batch.reqs)
        # 1. 分配显存页
        batch.out_loc = self.cache_manager.allocate(needed_size)
        
        # 2. Padding (为了适配CUDA Graph)
        if padding_size := self.engine.graph_runner.pad_batch(batch):
            batch.out_loc = F.pad(batch.out_loc, (0, padding_size), value=self.engine.dummy_page)
            
        # 3. 计算Input Loading索引 (读取历史token)
        load_indices = self._make_2d_indices(
            [(r.table_idx, r.cached_len, r.device_len) for r in batch.padded_reqs]
        )
        
        # 3. 计算Output Writing索引 (写入新生成的token)
        write_indices = self._make_2d_indices(
            [
                (
                    (r.table_idx, r.device_len, r.device_len + 1)
                    if r.can_decode()  # 如果还能解码，写入正确位置
                    else self.dummy_write_2d_pos # 否则写入dummy位置
                )
                for r in batch.reqs
            ]
        )
        assert all(r.device_len < self.engine.max_seq_len for r in batch.reqs)
        
        # 4. 更新页表
        # 将分配到的物理页映射到逻辑页表
        self.page_table.view(-1)[load_indices] = batch.out_loc
        
        # 5. 准备Attention metadata (如 block tables, sequence lengths)
        self.engine.attn_backend.prepare_metadata(batch)
        
        # 6. 打包返回
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            load_indices=load_indices,
            write_indices=write_indices,
        )

    def _schedule_next_batch(self) -> ForwardInput | None:
        """
        决策并调度下一个Batch
        
        优先级策略：
        1. Prefill优先: 只要有新请求且显存/计算资源允许，优先执行Prefill。
           这有助于快速响应新请求。
        2. Decode次之: 如果没有Prefill任务，则执行Decode（生成Token）。
        
        Returns:
            ForwardInput | None: 准备好的Batch输入数据，如果无任务则返回None。
        """
        # 尝试从PrefillManager获取Batch
        batch = self.prefill_manager.schedule_next_batch(self.prefill_budget)
        
        # 如果没有Prefill Batch，尝试从DecodeManager获取
        if batch is None:
            batch = self.decode_manager.schedule_next_batch()
            
        return self._prepare_batch(batch) if batch else None

    def _make_2d_indices(self, ranges: List[Tuple[int, int, int]]) -> torch.Tensor:
        """
        辅助函数：构造展平的索引
        
        将一组 (row, start_col, end_col) 范围转换为 1D 的 flat indices。
        用于在展平的 Page Table 或 Token Pool 中进行批量读写。
        
        Args:
            ranges: List of (row_idx, start_col, end_col)
        """
        STRIDE = self.token_pool.stride(0)
        needed_size = sum(end - begin for _, begin, end in ranges)
        # 使用 pin_memory 加速 CPU->GPU 传输
        indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
        offset = 0
        for entry, begin, end in ranges:
            length = end - begin
            offset += length
            # 生成对应范围的线性索引
            torch.arange(
                begin + entry * STRIDE,
                end + entry * STRIDE,
                dtype=torch.int32,
                out=indices_host[offset - length : offset],
            )
        return indices_host.to(self.device, non_blocking=True)

    def _load_token_ids(self, input: ForwardInput) -> None:
        """从Token Pool加载Input Tokens到Batch中"""
        input.batch.input_ids = self.token_pool.view(-1)[input.load_indices]

    def _write_token_ids(self, input: ForwardInput, output: ForwardOutput) -> None:
        """将生成的Next Tokens写回Token Pool"""
        self.token_pool.view(-1)[input.write_indices] = output.next_tokens_gpu

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        """
        执行模型的前向计算 (Forward Pass)
        
        步骤:
        1. Load: 从Pool加载Token IDs。
        2. Sync: (可选) 如果需要额外的同步。
        3. Compute: 调用Engine执行Forward。
        4. Write: 将结果写回Pool。
        5. Update: 更新DecodeManager中的请求状态。
        """
        self._load_token_ids(forward_input)
        batch, sample_args = forward_input.batch, forward_input.sample_args
        if ENV.OVERLAP_EXTRA_SYNC:  # NOTE: 某些特定情况下的同步需求
            self.stream.synchronize()
        
        # 执行模型计算
        forward_output = self.engine.forward_batch(batch, sample_args)
        
        # 保存结果并更新状态
        self._write_token_ids(forward_input, forward_output)
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output

    def run_when_idle(self) -> None:
        """闲置时执行的任务 (例如后台整理碎片)"""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        重叠调度主循环 (Overlap Loop)
        
        这是实现高吞吐量的关键。它让 CPU 和 GPU 工作流水线化。
        
        时间线示意:
        GPU:  [Batch N 计算] ... [Batch N+1 计算]
        CPU:  [Batch N-1 后处理] -> [Batch N+1 准备] -> [Batch N 后处理]
        
        Args:
            last_data: 上一个Loop返回的正在运行的数据 (Ongoing Data)
        """
        # 决定是否阻塞等待新消息
        # 只要有: 1. 上个Batch在运行 2. 有Prefill任务 3. 有Decode任务
        # 就不阻塞，否则阻塞等待以免空转
        blocking = not (
            last_data
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        # 接收新消息
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # 1. 调度并准备下一个 Batch (Pure CPU work)
        forward_input = self._schedule_next_batch()
        
        ongoing_data = None
        if forward_input is not None:
            # 2. 提交下一个 Batch 到 GPU (Async launch)
            # 使用 engine_stream_ctx 确保操作在计算流中排队
            with self.engine_stream_ctx:
                # 确保数据拷贝等准备工作已完成
                self.engine.stream.wait_stream(self.stream)
                # 启动Forward，立即返回
                ongoing_data = (forward_input, self._forward(forward_input))

        # 3. 处理上一个 Batch 的结果 (CPU work parallel with GPU)
        self._process_last_data(last_data, ongoing_data)
        
        return ongoing_data

    def normal_loop(self) -> None:
        """
        普通调度循环 (无重叠)
        
        串行执行所有步骤，逻辑更简单，用于调试或不支持重叠调度的环境。
        """
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
        """
        启动调度器主循环
        """
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                # 循环调用 overlap_loop，并在循环间传递状态
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        """关闭调度器，清理资源"""
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

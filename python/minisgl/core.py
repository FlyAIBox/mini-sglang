"""
Mini-SGLang 核心数据结构模块

本模块定义了Mini-SGLang推理系统的核心数据结构：
- SamplingParams: 采样参数配置
- Req: 单个推理请求的状态
- Batch: 批处理请求集合
- Context: 全局推理上下文

这些数据结构贯穿整个推理流程，是理解系统工作原理的关键。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal

import torch

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend, BaseAttnMetadata
    from minisgl.kvcache import BaseCacheHandle


@dataclass
class SamplingParams:
    """
    采样参数类
    
    控制文本生成时的随机性和多样性。这些参数决定了模型如何从预测的token概率分布中选择下一个token。
    
    属性说明:
        temperature (float): 温度参数，控制输出的随机性
            - 0.0: 贪心采样，始终选择概率最高的token（确定性输出）
            - 0.0-1.0: 较保守，输出更加确定和一致
            - 1.0: 正常采样，按原始概率分布
            - >1.0: 更随机，输出更有创造性但可能不太连贯
            
        top_k (int): Top-K采样，只从概率最高的K个token中采样
            - -1: 不使用top_k限制（默认）
            - >0: 只考虑前K个最可能的token
            例如: top_k=50 表示只从概率最高的50个token中选择
            
        top_p (float): Nucleus采样（Top-P采样），累积概率阈值
            - 1.0: 不使用top_p限制（默认）
            - 0.0-1.0: 选择累积概率达到p的最小token集合
            例如: top_p=0.9 表示从累积概率达到90%的token集合中采样
            
        ignore_eos (bool): 是否忽略结束符（EOS token）
            - False: 遇到EOS立即停止生成（默认）
            - True: 忽略EOS，继续生成直到达到max_tokens
            
        max_tokens (int): 最大生成token数量
            - 默认: 1024
            - 达到此数量后停止生成，即使未遇到EOS
    
    使用示例:
        # 贪心采样（确定性输出）
        params = SamplingParams(temperature=0.0)
        
        # 平衡的创造性采样
        params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)
        
        # 高度随机的创造性输出
        params = SamplingParams(temperature=1.5, top_k=100)
    """
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024

    @property
    def is_greedy(self) -> bool:
        """
        判断是否为贪心采样模式
        
        贪心采样意味着始终选择概率最高的token，输出是确定性的。
        满足以下任一条件即为贪心采样：
        1. temperature <= 0.0（温度为0或负数）
        2. top_k == 1（只考虑概率最高的1个token）
        且同时 top_p == 1.0（不使用nucleus采样限制）
        
        返回:
            bool: True表示贪心采样，False表示随机采样
        """
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass(eq=False)
class Req:
    """
    请求对象（Request）
    
    表示单个推理请求的完整状态。每个Req对象跟踪一个用户请求从prefill到decode的整个生命周期。
    
    核心概念:
        - cached_len: 已经计算并缓存的token数量（KV Cache中的长度）
        - device_len: 当前总共有多少token在处理中（包括已缓存和新计算的）
        - max_device_len: 最大允许的总token数（输入 + 要生成的输出）
        
    状态转换示例:
        初始状态（prefill前）:
            input_ids = [1, 2, 3, 4, 5]  (5个输入token)
            cached_len = 0  (还没有缓存)
            device_len = 5  (当前5个token)
            max_device_len = 5 + 100 = 105  (输入5 + 输出100)
            
        prefill后:
            cached_len = 5  (5个token已缓存到KV Cache)
            device_len = 5  (仍然是5个token)
            
        第1次decode后:
            cached_len = 5  (之前的5个token保持缓存)
            device_len = 6  (新增1个生成的token)
            input_ids = [1, 2, 3, 4, 5, 6]  (添加新生成的token)
            
        第2次decode后:
            cached_len = 6  (前6个token已缓存)
            device_len = 7  (又新增1个token)
            input_ids = [1, 2, 3, 4, 5, 6, 7]
    
    属性:
        input_ids (torch.Tensor): 输入token ID序列，存储在CPU内存中
            - 包含原始输入 + 已生成的所有输出token
            - 随着生成过程不断增长
            
        table_idx (int): 在调度器的请求表（RequestTable）中的索引
            - 用于快速查找和管理请求
            
        cached_len (int): 已缓存的token数量
            - 这些token的KV Cache已经计算并存储
            - 下次forward时不需要重新计算
            
        output_len (int): 期望生成的输出token数量
            - 由sampling_params.max_tokens决定
            
        uid (int): 请求的唯一标识符
            - 用于跟踪和调试
            
        sampling_params (SamplingParams): 采样参数配置
            - 控制文本生成的随机性和停止条件
            
        cache_handle (BaseCacheHandle): KV Cache的句柄
            - 指向该请求在KV Cache中的存储位置
            - 由KV Cache管理器（如RadixCacheManager）分配和管理
    
    自动计算的属性:
        device_len (int): 当前处理中的总token数
            - 初始值 = len(input_ids)
            - 每次decode后递增1
            
        max_device_len (int): 最大允许的总token数
            - = len(input_ids) + output_len
            - 达到此值后停止生成
    """
    input_ids: torch.Tensor  # cpu tensor
    table_idx: int
    cached_len: int
    output_len: int
    uid: int
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle

    def __post_init__(self) -> None:
        """
        初始化后的验证和设置
        
        计算派生属性并验证状态的一致性。
        """
        # 确保input_ids在CPU上（为了节省GPU显存）
        assert self.input_ids.is_cpu
        
        # 当前处理中的token总数
        self.device_len = len(self.input_ids)
        
        # 最大允许的token总数（输入 + 输出）
        self.max_device_len = len(self.input_ids) + self.output_len
        
        # 验证状态一致性：
        # 0 <= cached_len < device_len <= max_device_len
        # - cached_len至少为0（没有缓存）
        # - cached_len必须小于device_len（不能缓存比实际更多的token）
        # - device_len不能超过max_device_len（不能生成比预期更多的token）
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len

    @property
    def remain_len(self) -> int:
        """
        剩余可生成的token数量
        
        返回:
            int: 还能生成多少个token
                = max_device_len - device_len
                = 0 表示已达到最大长度，应该停止
        
        示例:
            max_device_len = 105, device_len = 10
            remain_len = 95  (还能生成95个token)
        """
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        """
        本次需要extend（扩展计算）的token数量
        
        在prefill或chunked prefill中，表示有多少新token需要计算KV Cache。
        在decode中，这个值通常是1（每次生成1个token）。
        
        返回:
            int: 需要新计算的token数量
                = device_len - cached_len
                = 已有的token - 已缓存的token
        
        示例:
            Prefill阶段: cached_len=0, device_len=100 → extend_len=100
            Decode阶段: cached_len=100, device_len=101 → extend_len=1
        """
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        """
        完成一次decode步骤
        
        在成功生成一个新token后调用，更新请求状态：
        1. 将cached_len更新到device_len（新生成的token已被缓存）
        2. device_len递增1（准备生成下一个token）
        
        状态变化:
            Before: cached_len=100, device_len=100
            Call: complete_one()
            After: cached_len=100, device_len=101
        
        这个方法在每次decode iteration后由调度器调用。
        """
        # 当前的token已经被计算并缓存
        self.cached_len = self.device_len
        # 为下一个token准备位置
        self.device_len += 1

    def append_host(self, next_token: torch.Tensor) -> None:
        """
        将新生成的token添加到输入序列
        
        在decode阶段，每次生成一个新token后，需要将其添加到input_ids中，
        以便下一次迭代时作为输入的一部分。
        
        参数:
            next_token (torch.Tensor): 新生成的token，shape为[1]的tensor
        
        示例:
            input_ids = [1, 2, 3, 4, 5]
            next_token = tensor([6])
            append_host(next_token)
            → input_ids = [1, 2, 3, 4, 5, 6]
        """
        self.input_ids = torch.cat([self.input_ids, next_token])

    def can_decode(self) -> bool:
        """
        判断是否还能继续生成
        
        检查是否已达到最大长度限制。
        
        返回:
            bool: True表示还能继续生成，False表示应该停止
        
        停止条件:
            - remain_len <= 0: 已达到max_tokens限制
            - 或者生成了EOS token（由调度器单独检查）
        """
        return self.remain_len > 0

    def __repr__(self) -> str:
        """
        返回请求对象的字符串表示，用于调试和日志
        
        显示关键状态信息：table_idx, cached_len, device_len, max_device_len
        """
        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )


@dataclass
class Batch:
    """
    批处理对象（Batch）
    
    将多个请求组合成一个批次进行并行处理，提高GPU利用率。
    这是推理系统中最重要的优化之一：通过批处理，可以在一次GPU调用中处理多个请求。
    
    批处理的两种模式:
        1. Prefill Batch: 处理新请求的输入prompt
           - 每个请求可能有不同长度的输入
           - 计算密集型，需要处理多个token
           
        2. Decode Batch: 为正在生成的请求产生下一个token
           - 每个请求只处理1个token（新生成的）
           - 访存密集型，需要读取大量KV Cache
    
    属性:
        reqs (List[Req]): 实际的有效请求列表
            - 这些是真实用户的请求
            - 数量 = 实际批次大小
            
        phase (Literal["prefill", "decode"]): 当前批次的处理阶段
            - "prefill": 处理输入prompt，计算初始KV Cache
            - "decode": 自回归生成，每次产生一个新token
            
        input_ids (torch.Tensor): 合并后的输入token ID张量
            - 由调度器从各个Req中收集并拼接
            - Prefill: 各个请求的新增token拼接，shape可能是[total_tokens]
            - Decode: 每个请求一个token，shape是[batch_size, 1]
            
        out_loc (torch.Tensor): 输出位置索引
            - 指示每个请求的输出应该写入哪个位置
            - 用于正确地将批次输出分配给各个请求
            
        padded_reqs (List[Req]): 填充后的请求列表
            - 为了使用CUDA Graph，batch size需要是固定的
            - 如果实际请求不足，用dummy请求填充
            - 数量 = CUDA Graph的固定batch size
            
        attn_metadata (BaseAttnMetadata): 注意力计算的元数据
            - 由注意力后端（FlashAttention/FlashInfer）设置
            - 包含注意力计算所需的所有索引和偏移信息
            - 例如：序列长度、KV Cache位置、block表等
    
    工作流程:
        1. 调度器创建Batch对象，指定reqs和phase
        2. 调度器准备input_ids和out_loc
        3. 如果使用CUDA Graph，填充padded_reqs到固定大小
        4. 注意力后端准备attn_metadata
        5. Engine执行forward，使用这些信息进行计算
    
    示例:
        # Prefill Batch
        batch = Batch(
            reqs=[req1, req2, req3],  # 3个新请求
            phase="prefill"
        )
        # req1: 50 tokens, req2: 30 tokens, req3: 20 tokens
        # input_ids shape: [100] (50+30+20 拼接)
        
        # Decode Batch
        batch = Batch(
            reqs=[req1, req2, req3, req4, req5],  # 5个正在生成的请求
            phase="decode"
        )
        # 每个请求生成1个token
        # input_ids shape: [5, 1]
    """
    reqs: List[Req]
    phase: Literal["prefill", "decode"]
    
    # 以下字段由调度器设置（不在__init__中初始化）
    input_ids: torch.Tensor = field(init=False)
    out_loc: torch.Tensor = field(init=False)
    padded_reqs: List[Req] = field(init=False)  # 可能包含一些用于填充的dummy请求
    
    # 此字段由注意力后端设置
    attn_metadata: BaseAttnMetadata = field(init=False)

    @property
    def is_prefill(self) -> bool:
        """
        判断是否为prefill阶段
        
        返回:
            bool: True表示这是prefill batch
        """
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        """
        判断是否为decode阶段
        
        返回:
            bool: True表示这是decode batch
        """
        return self.phase == "decode"

    @property
    def size(self) -> int:
        """
        实际的批次大小（有效请求数量）
        
        返回:
            int: 真实用户请求的数量
        """
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        """
        填充后的批次大小
        
        为了使用CUDA Graph，batch size需要固定。如果实际请求不足，
        会用dummy请求填充到预定义的大小。
        
        返回:
            int: 包含填充后的总请求数量
        
        注意:
            padded_size >= size
            如果不使用CUDA Graph，padded_size == size
        """
        return len(self.padded_reqs)


@dataclass
class Context:
    """
    全局推理上下文（Context）
    
    存储整个推理引擎的全局配置和状态。这是一个单例对象，
    在整个推理过程中共享，允许各个组件访问共同的配置和状态。
    
    设计模式：
        使用全局上下文模式，避免在每个函数调用中传递大量参数。
        类似于Flask中的request context或PyTorch的autograd context。
    
    属性:
        page_size (int): KV Cache的页大小
            - KV Cache被组织成固定大小的"页"（类似OS的分页内存）
            - 每页存储page_size个token的KV Cache
            - 典型值: 16, 32, 64
            - 较大的page_size可以减少管理开销，但可能增加内存碎片
            
        attn_backend (BaseAttnBackend): 注意力计算后端
            - FlashAttention: 适合prefill和通用场景
            - FlashInfer: 针对decode阶段优化
            - 负责准备注意力元数据和执行注意力计算
            
        _batch (Batch | None): 当前正在处理的批次
            - 使用forward_batch上下文管理器设置
            - 在forward期间有效，完成后自动清除
            - None表示当前没有活跃的批次
    
    使用方式:
        # 1. 初始化时设置全局上下文
        ctx = Context(page_size=16, attn_backend=flash_attn)
        set_global_ctx(ctx)
        
        # 2. 在forward时使用上下文管理器
        with ctx.forward_batch(batch):
            # 在此期间，batch是活跃的
            output = model(input)
            # 模型内部可以通过ctx.batch访问当前批次
        # 退出后，batch被清除
        
        # 3. 在模型内部访问全局上下文
        ctx = get_global_ctx()
        current_batch = ctx.batch
        page_size = ctx.page_size
    
    线程安全性:
        注意：当前实现不是线程安全的。每个推理进程（TP rank）
        维护自己的全局上下文，不应在多线程环境中共享。
    """
    page_size: int
    attn_backend: BaseAttnBackend
    _batch: Batch | None = field(default=None, init=False)

    @property
    def batch(self) -> Batch:
        """
        获取当前活跃的批次
        
        返回:
            Batch: 当前正在处理的批次对象
        
        异常:
            AssertionError: 如果在forward_batch上下文之外调用
        
        使用场景:
            在模型的forward方法中，需要访问批次信息（如phase、size等）
            来决定如何处理输入。
        """
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: Batch):
        """
        上下文管理器：设置活跃批次
        
        在处理一个批次时，使用此上下文管理器确保：
        1. 批次被正确设置为活跃状态
        2. 处理完成后自动清理
        3. 防止嵌套（不允许同时处理多个批次）
        
        参数:
            batch (Batch): 要处理的批次
        
        异常:
            AssertionError: 如果尝试嵌套使用（已有活跃批次时再次调用）
        
        示例:
            ctx = get_global_ctx()
            batch = Batch(reqs=[req1, req2], phase="decode")
            
            with ctx.forward_batch(batch):
                # 在此作用域内，batch是活跃的
                logits = engine.forward()
                # engine内部可以通过ctx.batch访问批次信息
            
            # 退出后，batch被清除，ctx.batch再次为None
        """
        # 防止嵌套：如果已有活跃批次，报错
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            # 设置当前批次为活跃状态
            self._batch = batch
            # 执行批次处理（yield返回控制权给调用者）
            yield
        finally:
            # 无论成功还是异常，都清除批次（类似try-finally）
            self._batch = None


# ============================================================================
# 全局上下文管理
# ============================================================================

# 全局变量：存储唯一的Context实例
# 使用module-level变量实现单例模式
_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    """
    设置全局推理上下文
    
    在推理引擎初始化时调用一次，设置整个推理过程共享的上下文。
    
    参数:
        ctx (Context): 要设置的上下文对象
    
    异常:
        AssertionError: 如果全局上下文已经被设置（防止重复设置）
    
    使用场景:
        在Engine.__init__中，创建Context并设置为全局：
        
        class Engine:
            def __init__(self, ...):
                ctx = Context(
                    page_size=self.config.page_size,
                    attn_backend=self.attn_backend
                )
                set_global_ctx(ctx)
    """
    global _GLOBAL_CTX
    # 防止重复设置（每个进程只应设置一次）
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    """
    获取全局推理上下文
    
    在模型和各个模块中调用，访问共享的上下文信息。
    
    返回:
        Context: 全局上下文对象
    
    异常:
        AssertionError: 如果全局上下文未设置（需要先调用set_global_ctx）
    
    使用场景:
        在模型的forward方法中获取上下文：
        
        class QwenModel:
            def forward(self, input_ids):
                ctx = get_global_ctx()
                batch = ctx.batch
                if batch.is_prefill:
                    # prefill逻辑
                else:
                    # decode逻辑
    """
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX

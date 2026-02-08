
from __future__ import annotations

import asyncio
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, overload

from minisgl.utils import UNSET, Unset, init_logger
from openai import AsyncOpenAI as OpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm

logger = init_logger(__name__)


@dataclass(frozen=True)
class BenchmarkTrace:
    """单个请求的 trace 信息"""
    timestamp: float    #请求发送的时间戳 (相对时间)
    message: str        # 请求的 prompt 内容
    output_length: int  # 期望的输出长度 (token 数)
    input_length: int | None = None  # 输入长度 (token 数), 用于统计或 hack


@dataclass(frozen=True)
class BenchOneResult:
    """单个请求的基准测试结果"""
    tics: List[float]  # 时间戳列表: [start, first_token, token_2, ..., end]
    input_len: int
    output_len: int

    def as_json(self) -> List[float]:
        return [self.input_len, self.output_len] + self.tics

    @staticmethod
    def from_json(raw: List[float]) -> BenchOneResult:
        # check raw[0] and raw[1] are integers
        assert raw[0].is_integer() and raw[1].is_integer()
        return BenchOneResult(tics=raw[2:], input_len=int(raw[0]), output_len=int(raw[1]))


@dataclass(frozen=True)
class RawResult:
    """内部使用的原始测试结果"""
    input_len: int | None
    output_len: int
    message: str
    tics: List[float]


@dataclass
class Counter:
    """简单的计数器"""
    current: int = 0
    history_max: int = 0

    def inc(self, n=1):
        self.current += n
        self.history_max = max(self.history_max, self.current)

    def dec(self, n=1):
        self.current -= n
        assert self.current >= 0


@dataclass
class Console:
    """控制台进度条管理器"""
    input_pbar: tqdm   # 发送请求进度
    output_pbar: tqdm  # 完成请求进度
    prefill_pbar: tqdm # Prefill token 进度
    decode_pbar: tqdm  # Decode token 进度
    disabled: bool
    inflight_counter: Counter = field(default_factory=Counter)
    queue_counter: Counter = field(default_factory=Counter)

    def update_input(self, n=1):
        """更新发送请求进度"""
        self.input_pbar.update(n)
        self.input_pbar.refresh()
        self.inflight_counter.inc(n)
        self.queue_counter.inc(n)

    def update_output(self, n=1):
        """更新完成请求进度"""
        self.output_pbar.update(n)
        self.output_pbar.refresh()
        self.inflight_counter.dec(n)

    def update_prefill(self, n=1):
        """更新 Prefill token 进度"""
        self.prefill_pbar.update(n)
        self.prefill_pbar.refresh()
        self.queue_counter.dec(n)

    def update_decode(self, n=1):
        """更新 Decode token 进度"""
        self.decode_pbar.update(n)

    @contextmanager
    def inflight(self, n=1):
        """Context Manager: 记录请求处理生命周期"""
        self.update_input(n)
        yield
        self.update_output(n)

    @contextmanager
    def log_stats(self):
        """Context Manager: 结束时关闭进度条并打印统计信息"""
        yield
        self.input_pbar.close()
        self.output_pbar.close()
        self.prefill_pbar.close()
        self.decode_pbar.close()
        if not self.disabled:
            max_inflight = self.inflight_counter.history_max
            max_queue = self.queue_counter.history_max
            logger.info(f"Max inflight requests: {max_inflight}, Max queued requests: {max_queue}")


@dataclass(frozen=True)
class BenchmarkResult:
    """基准测试结果集合"""
    raw_data: List[BenchOneResult]

    def as_json(self) -> List[List[float]]:
        return [r.as_json() for r in self.raw_data]

    @staticmethod
    def from_json(raw: List[List[float]]) -> BenchmarkResult:
        return BenchmarkResult(raw_data=[BenchOneResult.from_json(r) for r in raw])


def make_console(num_requests: int, sum_output_length: int, use_pbar: bool = True) -> Console:
    """创建并初始化控制台进度条"""
    BAR_FORMAT_0 = (
        "{desc:<10} {percentage:3.0f}%|{bar}|"
        " {n_fmt:>5}/{total_fmt} "
        "[{rate_fmt:>12} {elapsed:>8}/{remaining:<8}]"
    )
    BAR_FORMAT_1 = BAR_FORMAT_0
    n_fmt_align = 5
    prefill_tokens = num_requests
    decode_tokens = sum_output_length - prefill_tokens

    # 动态调整数字对齐宽度
    if len(str(decode_tokens)) > n_fmt_align:
        n_fmt_align = len(str(decode_tokens))
        BAR_FORMAT_0 = BAR_FORMAT_0.replace("{n_fmt:>5}", "{n_fmt:>" + str(n_fmt_align) + "}")
        BAR_FORMAT_1 = BAR_FORMAT_0

    if len(str(prefill_tokens)) < len(str(decode_tokens)):
        old_align_str = "{n_fmt:>" + str(n_fmt_align) + "}"
        n_fmt_align += len(str(decode_tokens)) - len(str(prefill_tokens))
        BAR_FORMAT_0 = BAR_FORMAT_0.replace(old_align_str, "{n_fmt:>" + str(n_fmt_align) + "}")

    disabled = not use_pbar
    input_pbar = tqdm(
        total=num_requests,
        desc="Requests sent",
        position=0,
        bar_format=BAR_FORMAT_0,
        disable=disabled,
    )
    output_pbar = tqdm(
        total=num_requests,
        desc="Requests done",
        position=1,
        bar_format=BAR_FORMAT_0,
        disable=disabled,
    )
    prefill_pbar = tqdm(
        total=prefill_tokens,
        desc="Prefill token",
        position=2,
        bar_format=BAR_FORMAT_0,
        disable=disabled,
    )
    decode_pbar = tqdm(
        total=decode_tokens,
        desc="Decode token ",
        position=3,
        bar_format=BAR_FORMAT_1,
        disable=disabled,
    )
    return Console(
        input_pbar=input_pbar,
        output_pbar=output_pbar,
        prefill_pbar=prefill_pbar,
        decode_pbar=decode_pbar,
        disabled=disabled,
    )


def generate_prompt(tokenizer: Any, n: int) -> str:
    """使用 tokenizer 生成大约 n 个 token 的随机 Prompt。"""
    vocab_size = tokenizer.vocab_size // 2
    token_ids = [random.randint(0, vocab_size) for _ in range(n - 1)]

    for _ in range(64):
        prompt = tokenizer.decode(token_ids)
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) == n:
            return prompt
        if len(token_ids) < n:
            need = n - len(token_ids)
            token_ids.extend([random.randint(0, vocab_size) for _ in range(need)])
        else:
            token_ids = token_ids[:n]

    raise ValueError("Failed to generate a message of the desired length.")


async def benchmark_one(
    client: OpenAI,
    prompt: str,
    output_length: int,
    model: str,
    *,
    pbar: Console | bool = True,
    extra_body: Dict[str, Any] | None = None,
    input_length: int | None = None,  # a hack to force input length
) -> RawResult:
    """
    对单个请求进行基准测试
    
    发送请求并记录以下时刻:
    1. 发送请求前
    2. 收到第一个 token (TTFT)
    3. 收到后续每个 token (TPOT)
    """
    if isinstance(pbar, bool):
        pbar = make_console(1, output_length, use_pbar=pbar)
    with pbar.inflight(1):
        kwargs = {
            "ignore_eos": True,
            "top_k": 1,
        }
        # 这是一个内部 hack 参数，可能用于强制设置输入长度
        if input_length is not None:
            kwargs["input_length_override"] = input_length
        kwargs.update(extra_body or {})  # can override kwargs
        response = await client.chat.completions.create(
            model=model,
            stream=True,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=output_length,
            temperature=0.0,
            extra_body=kwargs,
        )
        tics = [time.perf_counter()]
        async for _ in response:
            tics.append(time.perf_counter())
            if len(tics) == 2:
                # 收到第一个 token，意味着 prefill 完成
                pbar.update_prefill()
            elif len(tics) <= output_length + 1:
                # 收到后续 token，意味着一次 decode 步骤
                pbar.update_decode()
        return RawResult(
            input_len=input_length,
            output_len=output_length,
            message=prompt,
            tics=tics,
        )


async def benchmark_one_batch(
    client: OpenAI,
    prompts: List[str],
    output_lengths: List[int] | int,
    model: str,
    *,
    extra_body: Dict[str, Any] | None = None,
    input_lengths: List[int | None] | None = None,
    pbar: Console | bool = True,
) -> List[RawResult]:
    """并发测试一批请求"""
    if isinstance(output_lengths, int):
        output_lengths = [output_lengths] * len(prompts)
    if isinstance(pbar, bool):
        pbar = make_console(len(prompts), sum(output_lengths), use_pbar=pbar)
    if input_lengths is None:
        l: List[int | None] = [None] * len(prompts)
        input_lengths = l  # work-around for typing bug

    tasks = [
        benchmark_one(
            client=client,
            prompt=prompt,
            output_length=output_length,
            model=model,
            pbar=pbar,
            extra_body=extra_body,
            input_length=input_length,
        )
        for prompt, output_length, input_length in zip(
            prompts, output_lengths, input_lengths, strict=True
        )
    ]
    with pbar.log_stats():
        return await asyncio.gather(*tasks)


async def benchmark_trace(
    client: OpenAI,
    msgs: List[BenchmarkTrace],
    model: str,
    *,
    pbar: Console | bool = True,
) -> List[RawResult]:
    """
    按照 Trace 重放请求负载
    
    模拟真实的请求到达模式 (按 timestamp 发送请求)。
    """
    if isinstance(pbar, bool):
        sum_output_len = sum(msg.output_length for msg in msgs)
        pbar = make_console(len(msgs), sum_output_len, use_pbar=pbar)
    start = time.perf_counter()
    offset = min(msg.timestamp for msg in msgs) - 1

    async def benchmark_timed(msg: BenchmarkTrace):
        target = start + msg.timestamp - offset
        # 等待直到预定的发送时间
        await asyncio.sleep(max(0, target - time.perf_counter()))
        return await benchmark_one(
            client, msg.message, msg.output_length, model, pbar=pbar, input_length=msg.input_length
        )

    tasks = [benchmark_timed(msg) for msg in msgs]
    with pbar.log_stats():
        return await asyncio.gather(*tasks)


@overload
def process_benchmark_results(raw_data: List[RawResult], tokenizer: Any) -> BenchmarkResult: ...


@overload
def process_benchmark_results(raw_data: List[RawResult]) -> None: ...


def process_benchmark_results(
    raw_data: List[RawResult],
    tokenizer: Any = UNSET,
) -> BenchmarkResult | None:
    """
    处理和统计基准测试结果
    
    统计指标:
    - ATTFT (Average Time To First Token): 平均首字符延迟
    - TPOT (Time Per Output Token): 平均每个输出 token 的生成时间
    - E2E (End-to-End Latency): 端到端延迟
    - Throughput: 吞吐量 (token/s, req/s)
    
    打印 P50, P90, P99 分位数值。
    """
    accum_times: List[float] = []
    first_times: List[float] = []
    results = [r.tics for r in raw_data]
    for tics in results:
        deltas: List[float] = []
        for i in range(len(tics) - 1):
            diff = tics[i + 1] - tics[i]
            deltas.append(diff)
        # 首个 token 的时间间隔 (Prefill time)
        first_times.append(deltas[0])
        # 后续 token 的时间间隔 (Decode time)
        accum_times.extend(deltas[1:])

    e2e_times = [tics[-1] - tics[0] for tics in results]
    first_times.sort()
    accum_times.sort()
    e2e_times.sort()

    def _print_stats(times: List[float], scale: float = 1.0) -> Tuple[float, ...]:
        assert len(times) > 0
        return (
            scale * sum(times) / len(times),  # avg
            scale * times[int(len(times) * 0.5)],  # p50
            scale * times[int(len(times) * 0.9)],  # p90
            scale * times[int(len(times) * 0.99)],  # p99
            scale * max(times),  # max
        )

    def _fmt(x: float) -> str:
        if x >= 1000:
            return f"{int(x):>6}"
        elif x >= 10:
            return f"{x:>6.2f}"
        else:
            return f"{x:>6.4f}"

    avg_ttft, p50_ttft, p90_ttft, p99_ttft, max_ttft = _print_stats(first_times, 1000)
    avg_tpot, p50_tpot, p90_tpot, p99_tpot, max_tpot = _print_stats(accum_times, 1000)
    avg_e2e, p50_e2e, p90_e2e, p99_e2e, max_e2e = _print_stats(e2e_times)

    min_time = min(min(r) for r in results)
    max_time = max(max(r) for r in results)
    dur = max_time - min_time
    assert dur > 0, "Duration must be positive"

    num_tokens = sum(len(tic) for tic in results)
    num_requests = len(results)

    logger.info(f"Num requests: #{num_requests}, Num tokens: #{num_tokens}")
    logger.info(
        f"TTFT: {_fmt(avg_ttft)} ms (p50: {_fmt(p50_ttft)} ms, p90: {_fmt(p90_ttft)} ms,"
        f" p99: {_fmt(p99_ttft)} ms, max: {_fmt(max_ttft)} ms)"
    )
    logger.info(
        f"TPOT: {_fmt(avg_tpot)} ms (p50: {_fmt(p50_tpot)} ms, p90: {_fmt(p90_tpot)} ms,"
        f" p99: {_fmt(p99_tpot)} ms, max: {_fmt(max_tpot)} ms)"
    )
    logger.info(
        f"E2E:  {_fmt(avg_e2e) }  s (p50: {_fmt(p50_e2e) }  s, p90: {_fmt(p90_e2e) }  s,"
        f" p99: {_fmt(p99_e2e) }  s, max: {_fmt(max_e2e) }  s)"
    )
    logger.info(f"Duration: {_fmt(dur)} s")
    logger.info(f"Throughput: {_fmt(num_tokens / dur)} token/s, {_fmt(num_requests / dur)} req/s")

    # normalize the time to start from zero
    results = [[r - min_time for r in tics] for tics in results]
    if isinstance(tokenizer, Unset):
        return None

    return BenchmarkResult(
        raw_data=[
            BenchOneResult(
                tics=r.tics,
                input_len=(
                    r.input_len
                    if r.input_len is not None
                    else len(tokenizer.encode(r.message, add_special_tokens=False))
                ),
                output_len=r.output_len,
            )
            for r in raw_data
        ]
    )


def read_qwen_trace(
    file_path: str,
    tokenizer: Any,
    n: int | None = None,
    dummy: bool = False,
) -> List[BenchmarkTrace]:
    """读取 Qwen 格式的 trace 文件"""
    class JSONInput(BaseModel):
        chat_id: int
        parent_chat_id: int
        timestamp: float
        input_length: int
        output_length: int
        type: str  # unused
        turn: int  # unused
        hash_ids: List[int]  # unused

    with open(file_path, "r") as f:
        lines = f.readlines()
        if n is not None:
            lines = lines[:n]
    objs = [JSONInput.model_validate_json(line) for line in lines]
    if dummy:
        prompt = generate_prompt(tokenizer, max(obj.input_length for obj in objs))
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        _get_prompt = lambda obj: tokenizer.decode(ids[: obj.input_length])
    else:
        _get_prompt = lambda obj: generate_prompt(tokenizer, obj.input_length)
    return [
        BenchmarkTrace(
            timestamp=obj.timestamp,
            message=_get_prompt(obj),
            input_length=obj.input_length,
            output_length=obj.output_length,
        )
        for obj in objs
    ]


def read_mooncake_trace(
    file_path: str,
    tokenizer: Any,
    n: int | None = None,
    dummy: bool = False,
) -> List[BenchmarkTrace]:
    """读取 Mooncake 格式的 trace 文件"""
    class JSONInput(BaseModel):
        timestamp: int
        input_length: int
        output_length: int
        hash_ids: List[int]  # unused for now

    with open(file_path, "r") as f:
        lines = f.readlines()
        if n is not None:
            lines = lines[:n]
    objs = [JSONInput.model_validate_json(line) for line in lines]
    if dummy:
        prompt = generate_prompt(tokenizer, max(obj.input_length for obj in objs))
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        _get_prompt = lambda obj: tokenizer.decode(ids[: obj.input_length])
    else:
        _get_prompt = lambda obj: generate_prompt(tokenizer, obj.input_length)
    return [
        BenchmarkTrace(
            timestamp=obj.timestamp / 1000,
            message=_get_prompt(obj),
            input_length=obj.input_length,
            output_length=obj.output_length,
        )
        for obj in objs
    ]


def scale_traces(
    traces: List[BenchmarkTrace],
    scale: float,
) -> List[BenchmarkTrace]:
    """调整 Trace 的时间戳间隔，用于模拟不同程度的并发压力"""
    min_tic = min(trace.timestamp for trace in traces)
    return sorted(
        [
            BenchmarkTrace(
                timestamp=(trace.timestamp - min_tic) * scale,
                message=trace.message,
                input_length=trace.input_length,
                output_length=trace.output_length,
            )
            for trace in traces
        ],
        key=lambda x: x.timestamp,
    )


async def get_model_name(client: OpenAI) -> str:
    """获取服务端模型名称"""
    async for model in client.models.list():
        return model.id
    raise ValueError("No models available")


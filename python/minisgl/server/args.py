
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
from minisgl.distributed import DistributedInfo
from minisgl.scheduler import SchedulerConfig
from minisgl.utils import cached_load_hf_config, init_logger


@dataclass(frozen=True)
class ServerArgs(SchedulerConfig):
    """
    服务器配置参数
    
    继承自 SchedulerConfig，包含了调度器配置以及服务器特有的配置。
    这些参数通常由命令行传入，用于配置整个推理系统的行为。
    """
    server_host: str = "127.0.0.1"
    server_port: int = 1919
    num_tokenizer: int = 0  # Tokenizer 进程数量 (0 表示与 Detokenizer 共享)
    silent_output: bool = False

    @property
    def share_tokenizer(self) -> bool:
        """是否共享 Tokenizer 和 Detokenizer (当 num_tokenizer == 0)"""
        return self.num_tokenizer == 0

    @property
    def zmq_frontend_addr(self) -> str:
        """前端 (API Server) 接收 Tokenizer 消息的 ZMQ IPC 地址"""
        return "ipc:///tmp/minisgl_3" + self._unique_suffix

    @property
    def zmq_tokenizer_addr(self) -> str:
        """Tokenizer 接收前端消息的 ZMQ IPC 地址"""
        if self.share_tokenizer:
            # 如果共享，使用 Detokenizer 的地址
            return self.zmq_detokenizer_addr
        result = "ipc:///tmp/minisgl_4" + self._unique_suffix
        assert result != self.zmq_detokenizer_addr
        return result

    @property
    def tokenizer_create_addr(self) -> bool:
        """是否由 Tokenizer 进程创建 ZMQ Socket (Bind)"""
        return self.share_tokenizer

    @property
    def backend_create_detokenizer_link(self) -> bool:
        """是否由 Backend 创建 Detokenizer 链接"""
        return not self.share_tokenizer

    @property
    def frontend_create_tokenizer_link(self) -> bool:
        """是否由 Frontend 创建 Tokenizer 链接"""
        return not self.share_tokenizer

    @property
    def distributed_addr(self) -> str:
        """分布式通信 (PyNCCL) 的主节点地址"""
        return f"tcp://127.0.0.1:{self.server_port + 1}"


def parse_args(args: List[str], run_shell: bool = False) -> Tuple[ServerArgs, bool]:
    """
    解析命令行参数并返回 ServerArgs
    
    参数:
        args: 命令行参数列表 (例如 sys.argv[1:])
        run_shell: 是否运行在 Shell 模式
    
    返回:
        ServerArgs: 解析后的配置对象
        run_shell: 是否启用 shell模式 (可能由参数 --shell-mode 覆盖)
    """
    from minisgl.attention import validate_backend
    from minisgl.kvcache import SUPPORTED_CACHE_MANAGER

    parser = argparse.ArgumentParser(description="MiniSGL Server Arguments")

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights and activations. 'auto' will use FP16 for FP32/FP16 models and BF16 for BF16 models.",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        "--tp-size",
        type=int,
        default=1,
        help="The tensor parallelism size.",
    )

    parser.add_argument(
        "--max-running-requests",
        type=int,
        dest="max_running_req",
        default=ServerArgs.max_running_req,
        help="The maximum number of running requests.",
    )

    parser.add_argument(
        "--max-seq-len-override",
        type=int,
        default=ServerArgs.max_seq_len_override,
        help="The maximum sequence length override.",
    )

    parser.add_argument(
        "--memory-ratio",
        type=float,
        default=ServerArgs.memory_ratio,
        help="The fraction of GPU memory to use for KV cache.",
    )

    assert ServerArgs.use_dummy_weight == False
    parser.add_argument(
        "--dummy-weight",
        action="store_true",
        dest="use_dummy_weight",
        help="Use dummy weights for testing.",
    )

    assert ServerArgs.use_pynccl == True
    parser.add_argument(
        "--disable-pynccl",
        action="store_false",
        dest="use_pynccl",
        help="Disable PyNCCL for tensor parallelism.",
    )

    parser.add_argument(
        "--host",
        type=str,
        dest="server_host",
        default=ServerArgs.server_host,
        help="The host address for the server.",
    )

    parser.add_argument(
        "--port",
        type=int,
        dest="server_port",
        default=ServerArgs.server_port,
        help="The port number for the server to listen on.",
    )

    parser.add_argument(
        "--cuda-graph-max-bs",
        "--graph",
        type=int,
        default=ServerArgs.cuda_graph_max_bs,
        help="The maximum batch size for CUDA graph capture. None means auto-tuning based on the GPU memory.",
    )

    parser.add_argument(
        "--num-tokenizer",
        "--tokenizer-count",
        type=int,
        default=ServerArgs.num_tokenizer,
        help="The number of tokenizer processes to launch. 0 means the tokenizer is shared with the detokenizer.",
    )

    parser.add_argument(
        "--max-prefill-length",
        "--max-extend-length",
        type=int,
        dest="max_extend_tokens",
        default=ServerArgs.max_extend_tokens,
        help="Chunk Prefill maximum chunk size in tokens.",
    )

    parser.add_argument(
        "--num-pages",
        "--num-tokens",
        dest="num_page_override",
        type=int,
        default=ServerArgs.num_page_override,
        help="Set the maximum number of pages for KVCache.",
    )

    parser.add_argument(
        "--attention-backend",
        "--attn",
        type=validate_backend,
        default=ServerArgs.attention_backend,
        help="The attention backend to use. If two backends are specified,"
        " the first one is used for prefill and the second one for decode.",
    )

    parser.add_argument(
        "--cache-type",
        type=str,
        default=ServerArgs.cache_type,
        choices=SUPPORTED_CACHE_MANAGER.supported_names(),
        help="The KV cache management strategy.",
    )

    parser.add_argument(
        "--shell-mode",
        action="store_true",
        help="Run the server in shell mode.",
    )

    # Parse arguments
    kwargs = parser.parse_args(args).__dict__.copy()

    # resolve some arguments
    run_shell |= kwargs.pop("shell_mode")
    if run_shell:
        # Shell 模式下，强制单 batch 和禁用部分输出，以便于交互
        kwargs["cuda_graph_max_bs"] = 1
        kwargs["max_running_req"] = 1
        kwargs["silent_output"] = True

    if kwargs["model_path"].startswith("~"):
        kwargs["model_path"] = os.path.expanduser(kwargs["model_path"])

    DTYPE_MAP = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if (dtype_str := kwargs["dtype"]) != "auto":
        kwargs["dtype"] = DTYPE_MAP[dtype_str]
    else:
        # 自动推断 data type
        dtype_or_str = cached_load_hf_config(kwargs["model_path"]).dtype
        if isinstance(dtype_or_str, str):
            kwargs["dtype"] = DTYPE_MAP[dtype_or_str]
        else:
            kwargs["dtype"] = dtype_or_str

    # 初始化分布式信息 (TP)
    kwargs["tp_info"] = DistributedInfo(0, kwargs["tensor_parallel_size"])
    del kwargs["tensor_parallel_size"]

    result = ServerArgs(**kwargs)
    logger = init_logger(__name__)
    logger.info(f"Parsed arguments:\n{result}")
    return result, run_shell

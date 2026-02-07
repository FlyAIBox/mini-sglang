
from __future__ import annotations

from dataclasses import dataclass, field

from minisgl.engine import EngineConfig


def _get_pid_suffix() -> str:
    """获取基于 PID 的后缀，确保 IPC 地址唯一"""
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    """
    调度器配置
    
    集成 EngineConfig，并添加调度器特有的配置项。
    
    属性:
        max_extend_tokens: 最大扩展 token 数 (Prefill 阶段一次最多处理多少 token)
        cache_type: Cache 管理器类型 (e.g., "radix")
        offline_mode: 是否为离线模式 (不使用网络通信)
    """
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        """后端接收地址 (Tokenizers -> Scheduler)"""
        return "ipc:///tmp/minisgl_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        """Detokenizer 接收地址 (Scheduler -> Detokenizer)"""
        return "ipc:///tmp/minisgl_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        """调度器广播地址 (Rank 0 -> Other Ranks)"""
        return "ipc:///tmp/minisgl_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        """单次 Forward 最大处理长度"""
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        """是否由后端创建到 Detokenizer 的连接"""
        return True

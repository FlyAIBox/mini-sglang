
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

_LOG_LEVEL = None


def init_logger(
    name: str,
    suffix: str = "",
    *,
    strip_file: bool = True,
    level: str | None = None,
    use_pid: bool | None = None,
    use_tp_rank: bool | None = None,
):
    """
    初始化 Logger (带颜色和格式化)
    
    统一的日志初始化函数，支持:
    - 自动彩色输出 (基于日志级别)
    - 格式化的时间戳
    - 自动附加 PID 和 TP Rank 信息
    - 避免重复添加 Handler
    - 防止日志传播到 root logger (避免重复打印)
    """
    import logging
    import os
    import sys

    global _LOG_LEVEL
    if _LOG_LEVEL is None:
        LEVEL_MAP = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

        level = level or os.getenv("LOG_LEVEL", "").upper()
        _LOG_LEVEL = LEVEL_MAP.get(level, logging.INFO)

    if strip_file:
        suffix = os.path.basename(suffix)

    if suffix:
        suffix = f"|{suffix}"

    if use_pid is None:
        use_pid = os.getenv("LOG_PID", "0").lower() in ("1", "true", "yes")

    if use_pid:
        pid = os.getpid()
        suffix = f"|pid={pid}{suffix}"

    tp_info = None

    # Color formatter class
    class ColorFormatter(logging.Formatter):
        """带颜色的自定义日志格式化器"""

        # ANSI color codes
        COLORS = {
            "DEBUG": "\033[36m",  # Cyan (青色)
            "INFO": "\033[32m",  # Green (绿色)
            "WARNING": "\033[33m",  # Yellow (黄色)
            "ERROR": "\033[31m",  # Red (红色)
            "CRITICAL": "\033[35m",  # Magenta (洋红)
        }
        RESET = "\033[0m"
        BOLD = "\033[1m"

        def format(self, record):
            from minisgl.distributed import try_get_tp_info

            # 格式化时间戳: [YYYY-MM-DD|HH:MM:SS|pid=1234]
            timestamp = self.formatTime(record, "[%Y-%m-%d|%H:%M:%S{suffix}]")
            nonlocal tp_info
            tp_info = tp_info or try_get_tp_info()
            # 如果是分布式环境，自动附加 Rank 信息
            if tp_info is not None and use_tp_rank is not False:
                real_suffix = f"{suffix}|core|rank={tp_info.rank}"
            else:
                real_suffix = suffix
            timestamp = timestamp.format(suffix=real_suffix)

            # 获取日志级别颜色
            level_color = self.COLORS.get(record.levelname, "")

            # 格式化消息体
            colored_level = f"{level_color}{record.levelname:<8}{self.RESET}"
            message = record.getMessage()

            # 最终格式: [timestamp] LEVEL message
            return f"{self.BOLD}{timestamp}{self.RESET} {colored_level} {message}"

    logger = logging.getLogger(name)
    logger.setLevel(_LOG_LEVEL)

    # 清除现有的 handlers，避免重复输出
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    formatter = ColorFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # 禁止传播到 root logger
    logger.propagate = False

    def _call_rank0(msg, *args, _which, **kwargs):
        """仅在 Rank 0 打印日志的辅助函数"""
        from minisgl.distributed import get_tp_info

        nonlocal tp_info
        tp_info = tp_info or get_tp_info()
        assert tp_info is not None, "TP info not set yet"
        # 只有主进程才打印
        if tp_info.is_primary():
            getattr(logger, _which)(msg, *args, **kwargs)

    if TYPE_CHECKING:

        class WrapperLogger(logging.Logger):
            """自定义 Logger 类型提示，包含 rank0 方法"""

            def info_rank0(self, msg, *args, **kwargs): ...
            def warning_rank0(self, msg, *args, **kwargs): ...
            def debug_rank0(self, msg, *args, **kwargs): ...
            def critical_rank0(self, msg, *args, **kwargs): ...

        return WrapperLogger(name)
    else:
        # 动态绑定 rank0 方法
        logger.info_rank0 = partial(_call_rank0, _which="info")
        logger.debug_rank0 = partial(_call_rank0, _which="debug")
        logger.critical_rank0 = partial(_call_rank0, _which="critical")
        logger.warning_rank0 = partial(_call_rank0, _which="warning")
        return logger

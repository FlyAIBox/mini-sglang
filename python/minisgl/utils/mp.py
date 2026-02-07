
from __future__ import annotations

from typing import Callable, Dict, Generic, TypeVar

import msgpack
import zmq
import zmq.asyncio

T = TypeVar("T")


class ZmqPushQueue(Generic[T]):
    """
    ZMQ Push Queue (同步版本)
    
    单向消息发送队列，用于进程间通过 ZMQ 通信。
    使用 msgpack 进行序列化。
    """
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def put(self, obj: T):
        """发送对象到队列"""
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        self.socket.send(event, copy=False)

    def stop(self):
        """关闭队列和上下文"""
        self.socket.close()
        self.context.term()


class ZmqAsyncPushQueue(Generic[T]):
    """
    ZMQ Push Queue (异步版本)
    
    用于 asyncio 环境下的非阻塞消息发送。
    """
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    async def put(self, obj: T):
        """异步发送对象到队列"""
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        await self.socket.send(event, copy=False)

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqPullQueue(Generic[T]):
    """
    ZMQ Pull Queue (同步版本)
    
    单向消息接收队列。
    """
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder

    def get(self) -> T:
        """从队列接收并反序列化对象 (阻塞)"""
        event = self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def get_raw(self) -> bytes:
        """接收原始字节数据"""
        return self.socket.recv()

    def decode(self, raw: bytes) -> T:
        """手动反序列化原始数据"""
        return self.decoder(msgpack.unpackb(raw, raw=False))

    def empty(self) -> bool:
        """检查队列是否为空"""
        return self.socket.poll(timeout=0) == 0

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqAsyncPullQueue(Generic[T]):
    """
    ZMQ Pull Queue (异步版本)
    
    用于 asyncio 环境下的异步消息接收。
    """
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder

    async def get(self) -> T:
        """异步接收并反序列化对象"""
        event = await self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqPubQueue(Generic[T]):
    """
    ZMQ Publish Queue
    
    广播模式发送者，一对多通信。
    """
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def put_raw(self, raw: bytes):
        """广播原始字节数据"""
        self.socket.send(raw, copy=False)

    def put(self, obj: T):
        """广播序列化对象"""
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        self.socket.send(event, copy=False)

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqSubQueue(Generic[T]):
    """
    ZMQ Subscribe Queue (同步版本)
    
    广播模式订阅者。
    """
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        # 订阅所有消息
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.decoder = decoder

    def get(self) -> T:
        """从订阅接收消息 (阻塞)"""
        event = self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def empty(self) -> bool:
        return self.socket.poll(timeout=0) == 0

    def stop(self):
        self.socket.close()
        self.context.term()

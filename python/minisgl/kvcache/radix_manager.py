
"""
Radix Tree KV Cache 管理器

本模块实现了基于 Radix Tree (前缀树) 的 KV Cache 管理机制。
这是 SGLang/Mini-SGLang 高效复用 KV Cache 的核心组件。

设计理念：
- 将 Token 序列映射为树路径，相同的 Token 前缀共享相同的路径节点。
- 每个节点存储一段连续的 KV Cache 物理索引。
- 通过前缀匹配 (Prefix Matching) 快速找到可复用的 Cache。
- 采用 LRU (Least Recently Used) 策略管理显存，自动驱逐最久未使用的节点。
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from .base import BaseCacheHandle, BaseCacheManager, SizeInfo


class RadixTreeNode:
    """
    Radix Tree 节点
    
    表示前缀树中的一个节点，对应一段连续的 Token 序列及其 KV Cache。
    
    属性:
        children (Dict[int, RadixTreeNode]): 子节点映射 {next_token_id: child_node}。
        _parent (RadixTreeNode): 父节点引用。
        _key (torch.Tensor): 该节点存储的 Token 序列 (GPU Tensor)。
        _value (torch.Tensor): 对应的 KV Cache 物理页索引 (GPU Tensor)。
        ref_count (int): 引用计数。表示当前有多少个活跃请求正在使用该节点。
                        ref_count > 0 的节点受到保护，不可被驱逐。
        timestamp (int): 最后访问时间戳，用于 LRU 驱逐策略。
    """
    counter: int = 0

    def __init__(self, tic: int | None = None) -> None:
        self.children: Dict[int, RadixTreeNode] = {}
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0
        self.uuid = RadixTreeNode.counter
        RadixTreeNode.counter += 1
        self.timestamp = tic or time.monotonic_ns()

        # 这些字段稍后通过 set_key_value 设置
        self._key: torch.Tensor
        self._value: torch.Tensor
        self._length: int

    def set_key_value(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """设置节点的 Token 序列 Key 和 KV Cache 索引 Value"""
        assert len(key) == len(value)
        self._key = key
        self._value = value
        self._length = len(key)

    def set_parent(self, parent: RadixTreeNode) -> None:
        """设置父节点，并维护父节点的 children 映射"""
        self._parent = parent
        # 以 key 的第一个 token 作为在父节点 children 中的索引键
        parent.children[int(self._key[0].item())] = self

    @property
    def length(self) -> int:
        return self._length

    @property
    def parent(self) -> RadixTreeNode:
        assert self._parent is not None
        return self._parent

    @property
    def value(self) -> torch.Tensor:
        return self._value

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        """
        计算输入序列与当前节点 Key 的匹配长度
        
        使用自定义 CUDA kernel (fast_compare_key) 进行并行比较，
        返回第一个不匹配的位置索引。
        """
        from minisgl.kernel import fast_compare_key

        # compare key and input_ids, find the first diff
        return fast_compare_key(self._key, input_ids)

    def _split_at(self, pos: int) -> RadixTreeNode:
        """
        在指定位置分裂节点 (Split)
        
        应用场景：
        当一个新的请求与当前节点部分匹配（但不是完全匹配）时，需要将当前节点从不匹配的位置切开。
        前半部分保留为父节点，后半部分成为子节点，新请求的分支将成为另一个子节点。
        
        Before Split:
            Parent -> [A, B, C, D] (Current Node) -> Children
            
        Split at pos=2 (after B):
            Parent -> [A, B] (New Parent) -> [C, D] (Original Node) -> Children
        
        Args:
            pos: 分裂位置 (Split Position)
            
        Returns:
            RadixTreeNode: 分裂出的上半部分节点 (New Parent)
        """
        assert 0 < pos < self.length
        parent = self.parent

        # 创建新节点作为上半部分 [0:pos]
        new_node = RadixTreeNode(self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count # 继承引用计数

        # 修改当前节点为下半部分 [pos:]
        self.set_key_value(self._key[pos:], self._value[pos:])
        self.set_parent(new_node) # 当前节点挂载到新节点下

        return new_node

    def __lt__(self, other: RadixTreeNode) -> bool:
        # 用于优先队列 (heapq) 比较，按时间戳排序以实现 LRU
        return self.timestamp < other.timestamp


@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    """
    Radix Cache 句柄
    
    请求持有的对象，用于引用树中的某个节点。
    scheduler 通过此句柄跟踪请求占用的 KV Cache 资源。
    """
    node: RadixTreeNode


class RadixCacheManager(BaseCacheManager):
    """
    基于 Radix Tree 的 KV Cache 管理器
    
    核心功能：
    1. 自动前缀缓存 (Automatic Prefix Caching): 缓存已生成的 KV Cache。
    2. 动态树维护: 处理节点的插入、查找、分裂。
    3. 内存回收 (Eviction): 当显存不足时，基于 LRU 策略驱逐节点。
    
    状态维护：
    - evictable_size: 当前处于非活跃状态（引用计数为0）、可被回收的 Token 总数。
    - protected_size: 当前被活跃请求引用、受保护不可回收的 Token 总数。
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        super().__init__()
        # 根节点不存储任何数据，始终存在
        self.root_node = RadixTreeNode()
        self.root_node.ref_count = 1  # root is always protected
        self.evictable_size = 0
        self.protected_size = 0

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """
        锁定/解锁缓存资源
        
        - Lock (unlock=False): 请求开始使用该节点，引用计数 +1，状态转为 Protected。
        - Unlock (unlock=True): 请求结束使用，引用计数 -1。如果降为0，状态转为 Evictable。
        
        注意：锁定操作会递归向上影响路径上的所有父节点。
        """
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        if unlock:
            while not node.is_root():
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0:
                    # 引用归零，变为可回收
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                node = node.parent
        else:
            while not node.is_root():
                if node.ref_count == 0:
                    # 原本是 0，现在被引用，变为受保护
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
                node = node.parent

    def match_prefix(self, input_ids: torch.Tensor) -> Tuple[RadixCacheHandle, torch.Tensor]:
        """
        前缀匹配 (Prefix Matching)
        
        在树中寻找与 input_ids 匹配的最长前缀路径。
        
        Returns:
            Tuple: (匹配末端的节点句柄, 路径上所有节点的 KV Cache 索引拼接)
        """
        node, prefix_len = self._walk(input_ids)
        if prefix_len == 0:
            assert node.is_root() and node is self.root_node and prefix_len == 0
            return RadixCacheHandle(prefix_len, node), self.empty_tensor
        
        # 回溯路径，收集所有父节点的 KV Cache 索引
        value_list: List[torch.Tensor] = []
        matched_node = node
        while not node.is_root():
            value_list.append(node.value)
            node = node.parent
        value_list.reverse()
        
        return RadixCacheHandle(prefix_len, matched_node), torch.cat(value_list)

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> int:
        """
        前缀插入 (Prefix Insertion)
        
        将新生成的 KV Cache (input_ids + indices) 插入到树中。
        通常发生在请求处理完成，或者 Prefill 阶段完成时。
        
        流程：
        1. 找到最长匹配路径 (Walk)。
        2. 如果有未匹配的后缀，创建新节点挂载到匹配节点下。
        """
        node, prefix_len = self._walk(input_ids)
        assert prefix_len <= len(input_ids)
        
        # 如果有新内容需要插入
        if prefix_len < len(input_ids):
            new_node = RadixTreeNode()
            # 设置新节点内容
            new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:].clone())
            new_node.set_parent(node) # 挂载
            self.evictable_size += new_node.length # 新节点初始引用为0，是可回收的
            
        return prefix_len

    def _walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        """
        树遍历核心逻辑
        
        沿着边遍历树，直到无法匹配更多 Token。
        如果在边的中间停止匹配，会对该边对应的节点执行 Split 操作。
        同时更新节点的时间戳 (LRU)。
        """
        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns()

        while prefix_len < indice_len:
            # 1. 根据当前 Token 查找子节点
            this_id = int(input_ids[prefix_len].item())
            if this_id not in node.children:
                return node, prefix_len # 无匹配路径

            node = node.children[this_id]

            # 2. 比较子节点的 Key 与输入序列
            match_len = node.get_match_len(input_ids[prefix_len:])
            prefix_len += match_len

            # 3. 处理部分匹配 (Partial Match) -> Split
            if match_len != node.length:
                # 只匹配了节点的一部分，需要分裂节点
                node = node._split_at(match_len)
                return node, prefix_len

            # 4. 更新访问时间 (LRU)
            node.timestamp = tic

        # 完全匹配到某个节点边界
        return node, prefix_len

    def evict(self, size: int) -> torch.Tensor:
        """
        显存回收 (Eviction)
        
        当显存不足时调用。使用 LRU 策略优先驱逐最近最少使用的节点。
        
        Args:
            size: 需要释放的 Token 数量目标
            
        Returns:
            torch.Tensor: 被释放的所有 KV Cache 物理页索引
        """
        if size == 0:
            return self.empty_tensor
        assert (
            size <= self.evictable_size
        ), f"Cannot evict {size}, only {self.evictable_size} is evictable"

        # 1. 收集所有可驱逐的叶子节点
        leave_nodes = self._collect_leave_nodes_for_evict()
        # 构建最小堆，按时间戳排序
        heapq.heapify(leave_nodes)
        
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size:
            assert (
                leave_nodes
            ), f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
            
            # 2. 弹出最久未访问的节点
            node = heapq.heappop(leave_nodes)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            
            # 3. 记录回收的索引
            evicted_size += node.length
            evicted_indices.append(node.value)
            self.evictable_size -= node.length
            
            # 4. 从树中移除并断开连接
            parent = node.parent
            del parent.children[int(node._key[0].item())]
            
            # 5. 检查父节点
            # 如果父节点因此变成了叶子节点，且没有被引用，则它也成为可驱逐对象
            if parent.is_leaf() and parent.ref_count == 0:
                # 根节点除外
                if not parent.is_root():
                    heapq.heappush(leave_nodes, parent)

        return torch.cat(evicted_indices)

    def _collect_leave_nodes_for_evict(self) -> List[RadixTreeNode]:
        """BFS 收集所有引用计数为 0 的叶子节点"""
        nodes: List[RadixTreeNode] = [self.root_node]
        leave_nodes: List[RadixTreeNode] = []

        while len(nodes) > 0:
            node = nodes.pop()
            if node.is_leaf():
                if node.ref_count == 0:
                    leave_nodes.append(node)
            else:
                for child in node.children.values():
                    nodes.append(child)

        return leave_nodes

    def reset(self) -> None:
        raise NotImplementedError("RadixManager.reset is not implemented")

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(
            evictable_size=self.evictable_size,
            protected_size=self.protected_size,
        )

    def check_integrity(self) -> None:
        pass

from typing import Dict, Callable, Any
import functools
from collections import defaultdict
import enum
import heapq
import time
from typing import DefaultDict, Dict, List, Optional, Tuple

import numpy as np

from vllm.block import PhysicalTokenBlock
from vllm.core.lora_cost_manager import LoraCostManager
from vllm.logger import init_logger
from vllm.utils import Device

logger = init_logger(__name__)

# 用于存储每个函数的调用时间统计
function_stats = {
    "calls": {},      # 调用次数
    "total_time": {},  # 总时间
    "max_time": {},   # 最长单次调用时间
    "min_time": {},   # 最短单次调用时间
}


def time_tracker(func: Callable) -> Callable:
    """装饰器：跟踪函数调用时间"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # 记录开始时间
        start_time = time.time()

        # 调用原函数
        result = func(*args, **kwargs)

        # 计算执行时间
        execution_time = time.time() - start_time

        # 更新统计信息
        func_name = func.__name__
        if func_name not in function_stats["calls"]:
            function_stats["calls"][func_name] = 0
            function_stats["total_time"][func_name] = 0
            function_stats["max_time"][func_name] = 0
            function_stats["min_time"][func_name] = float('inf')

        function_stats["calls"][func_name] += 1
        function_stats["total_time"][func_name] += execution_time
        function_stats["max_time"][func_name] = max(
            function_stats["max_time"][func_name], execution_time)
        function_stats["min_time"][func_name] = min(
            function_stats["min_time"][func_name], execution_time)

        return result

    return wrapper

def reset_function_stats():
    """重置所有函数调用统计信息"""
    global function_stats
    function_stats = {
        "calls": {},
        "total_time": {},
        "max_time": {},
        "min_time": {}
    }
    print("函数调用统计信息已重置")

def print_function_stats():
    """打印所有被跟踪函数的调用统计信息"""
    print("\n--- 函数调用统计 ---")
    print(f"{'函数名':<20} {'调用次数':<10} {'总时间(秒)':<15} {'平均时间(秒)':<15} {'最长时间(秒)':<15} {'最短时间(秒)':<15}")
    print("-" * 90)

    total_all_functions = 0
    for func_name in sorted(function_stats["calls"].keys()):
        calls = function_stats["calls"][func_name]
        total = function_stats["total_time"][func_name]
        avg = total / calls if calls > 0 else 0
        max_t = function_stats["max_time"][func_name]
        min_t = function_stats["min_time"][func_name] if function_stats["min_time"][func_name] != float(
            'inf') else 0

        print(
            f"{func_name:<20} {calls:<10} {total:<15.6f} {avg:<15.6f} {max_t:<15.6f} {min_t:<15.6f}")
        total_all_functions += total

    print("-" * 90)
    print(f"所有函数总执行时间: {total_all_functions:.6f} 秒")


class kvCacheProgressStatus(enum.Enum):
    """Status of a SequenceData in RTC."""

    # 正在RUNNING
    RUNNING = enum.auto()
    # STABLY_EVICTED
    STABLE = enum.auto()
    # PENDING
    PENDING = enum.auto()


class NodeType(enum.Enum):
    ROOT = enum.auto()
    LORA = enum.auto()
    KV = enum.auto()


class TreeNode:
    def __init__(self,
                 parent: Optional["TreeNode"],
                 nodeType: NodeType,
                 progressStatus: kvCacheProgressStatus,
                 height: int):
        self.parent = parent
        self.node_type = nodeType
        self.status = progressStatus
        self.height = height
        self.last_access_time = time.time()
        self.used_time = 1
        self.device: Device


class RootTreeNode(TreeNode):
    def __init__(self):
        super().__init__(None, NodeType.ROOT, kvCacheProgressStatus.RUNNING, 0)
        self.device = Device.GPU
        self.children: Dict[int, "TreeNode"] = {}


class LoRATreeNode(TreeNode):
    def __init__(self, parent: TreeNode, progressStatus: kvCacheProgressStatus, key: int, value: List[PhysicalTokenBlock], device: Device):
        super().__init__(parent, NodeType.LORA, progressStatus, 1)  # 高度固定为1
        self.key = key
        self.value: List[PhysicalTokenBlock] = value
        self.children: Dict[Tuple, "TreeNode"] = {}
        self.device = device


class KVTreeNode(TreeNode):
    def __init__(self, parent: TreeNode, progressStatus: kvCacheProgressStatus, height: int, key: Tuple, value: PhysicalTokenBlock):
        super().__init__(parent, NodeType.KV, progressStatus, height)
        self.key = key
        self.value = value
        self.children: Dict[Tuple, "TreeNode"] = {}
        self.device = value.device


class ELoRATree:
    ######### api function#########
    @time_tracker
    def __init__(self):
        self.root_node = RootTreeNode()
        self.kv_key_node_dict: Dict[Tuple[int, int], KVTreeNode] = {}
        self.lora_cost_manager = LoraCostManager()
        self.low_lora = 1

    def match_prefix(self, lora_id: int, block_keys: List[Tuple]) -> List[TreeNode]:
        # assert lora_id in self.root_node.children
        # lora_node = self.root_node.children[lora_id]
        # nodes: List[TreeNode] = [lora_node]
        # self.lora_cost_manager.record_access(str(lora_id))
        # self._match_prefix_helper(lora_node, block_keys, nodes)
        # return nodes
        self.lora_cost_manager.record_access(str(lora_id))
        lora_node: TreeNode = self.find_lora_node(lora_id)
        ret_nodes: List[TreeNode] = [lora_node]
        for key in block_keys:
            if key in self.kv_key_node_dict:
                ret_nodes.append(self.kv_key_node_dict[key])
            else:
                break
        for node in ret_nodes:
            node.last_access_time = time.time()
            node.used_time += 1
        return ret_nodes

    @time_tracker
    def insert_lora(self, progressStatus: kvCacheProgressStatus, key: int, value: List[PhysicalTokenBlock], device: Device):
        lora_node: LoRATreeNode = LoRATreeNode(
            self.root_node, progressStatus, key, value, device)
        lora_id = lora_node.key
        assert lora_id not in self.root_node.children
        self.root_node.children[lora_id] = lora_node

    @time_tracker
    def insert_kv(self, lora_id: int, keys: List[Tuple], insert_blocks: List[PhysicalTokenBlock]):
        self.lora_cost_manager.record_access(str(lora_id))
        blocks = insert_blocks.copy()
        assert len(keys) == len(blocks)
        assert lora_id in self.root_node.children
        if not keys:
            return
        match_nodes: List[TreeNode] = self.match_prefix(lora_id, keys.copy())
        assert match_nodes
        match_node: TreeNode = match_nodes.pop(0)
        insert_key: Tuple
        while len(match_nodes) > 0:
            match_node = match_nodes.pop(0)
            insert_key = keys.pop(0)
            blocks.pop(0)
            assert match_node.key == insert_key
        parent_node: TreeNode = match_node
        while len(keys) > 0:
            insert_key = keys.pop(0)
            insert_block = blocks.pop(0)
            assert insert_key not in parent_node.children
            current_node = KVTreeNode(
                parent_node, kvCacheProgressStatus.RUNNING, parent_node.height+1, insert_key, insert_block)
            self.kv_key_node_dict[insert_key] = current_node
            parent_node.children[insert_key] = current_node
            parent_node = current_node
    

    @time_tracker
    def evict(self, num_nodes: int, device: Device) -> List[TreeNode]:
        self.low_lora = self.lora_cost_manager.calculate_low_lora()
        leaves = self._collect_leaves_for_evict(device)
        heap_items = [(self.calculate_node_cost(node), node)
                      for node in leaves]
        heapq.heapify(heap_items)
        ret_nodes: List[TreeNode] = []
        evicted_block_num = 0
        while evicted_block_num < num_nodes and heap_items:
            _, current_node = heapq.heappop(heap_items)
            if current_node == self.root_node:
                logger.info("only root node left")
                break
            assert self.check_conditions(
                current_node, device, ret_nodes) == True
            ret_nodes.append(current_node)
            if current_node.node_type == NodeType.KV:
                evicted_block_num += 1
            if current_node.node_type == NodeType.LORA:
                evicted_block_num += len(current_node.value)
            parent_node = current_node.parent

            if self.check_conditions(parent_node, device, ret_nodes):
                heapq.heappush(
                    heap_items, (self.calculate_node_cost(parent_node), parent_node))
        return ret_nodes

    @time_tracker
    def promote(self, num_nodes: int, device: Device) -> List[TreeNode]:
        candidates = self._collect_candidates_for_promote(device)

        # 使用优先队列存储节点及其优先级
        # 通过对calculate_node_cost的结果取负，实现相反的排序逻辑
        heap_items = []
        for node in candidates:
            # 获取成本元组
            cost = self.calculate_node_cost(node)
            # 将成本元组中的每个元素取负，实现排序逻辑反转
            inverted_cost = tuple(-x for x in cost)
            heap_items.append((inverted_cost, node))

        heapq.heapify(heap_items)

        ret_nodes: List[TreeNode] = []
        promoted_block_num = 0
        while promoted_block_num < num_nodes and heap_items:
            _, current_node = heapq.heappop(heap_items)
            ret_nodes.append(current_node)
            if current_node.node_type == NodeType.KV:
                promoted_block_num += 1
            if current_node.node_type == NodeType.LORA:
                promoted_block_num += len(current_node.value)
            # 处理子节点
            for child in current_node.children.values():
                if (child.device == device and
                        child.status == kvCacheProgressStatus.STABLE):
                    cost = self.calculate_node_cost(child)
                    inverted_cost = tuple(-x for x in cost)
                    heapq.heappush(heap_items, (inverted_cost, child))
        return ret_nodes

    @time_tracker
    def remove(self, remove_keys: List[Tuple]) -> List[Tuple]:
        return_keys: List[Tuple] = []
        for key in remove_keys:
            node = self.find(key)
            del self.kv_key_node_dict[key]
            assert node is not None
            if len(node.children) == 0:
                remove_res = self.remove_helper(node)
                assert remove_res == True
                return_keys.append(key)
        return return_keys

    @time_tracker
    def update_lora(self,
                    key: int,
                    progressStatus: kvCacheProgressStatus = None,
                    value: List[PhysicalTokenBlock] = None,
                    device: Device = None):
        assert key in self.root_node.children
        lora_node: LoRATreeNode = self.root_node.children[key]
        if progressStatus:
            lora_node.status = progressStatus
        if value:
            lora_node.value = value
        if device:
            lora_node.device = device

    @time_tracker
    def update_kv(self, key: Tuple, block: PhysicalTokenBlock = None, status: kvCacheProgressStatus = None):
        kv_node = self.find(key)
        assert kv_node
        if block:
            kv_node.value = block
        if status:
            kv_node.status = status
        kv_node.device = kv_node.value.device

    @time_tracker
    def update_last_block(self, old_block: PhysicalTokenBlock, new_block: PhysicalTokenBlock):
        # self.print()
        node = self.find((old_block.block_hash, old_block.num_hashed_tokens))
        assert node is not None
        assert len(node.children) == 0
        node.value = new_block
        node.device = new_block.device
        _key = (new_block.block_hash, new_block.num_hashed_tokens)
        node.key = _key
        parent_node = node.parent
        del parent_node.children[(
            old_block.block_hash, old_block.num_hashed_tokens)]
        del self.kv_key_node_dict[(
            old_block.block_hash, old_block.num_hashed_tokens)]
        parent_node.children[_key] = node
        self.kv_key_node_dict[_key] = node

    ######### helper function#########
    def remove_helper(self, node: TreeNode) -> bool:
        if node is None or node is self.root_node:
            return False
        assert len(node.children) == 0
        parent = node.parent
        if parent is None:
            return False
        # 从父节点的children中移除该节点
        # 需要找到对应的key
        for key, child in parent.children.items():
            if child is node:
                del parent.children[key]
                break
        # 清理节点的引用
        node.parent = None
        node.children.clear()
        return True

    def _match_prefix_helper(self, node: TreeNode, key: List[Tuple], nodes: List[TreeNode]):
        node.last_access_time = time.time()
        node.used_time += 1
        for _key, _child in node.children.items():
            if _key == key[0] and not (
                _child.status == kvCacheProgressStatus.RUNNING
                and _child.device == Device.CPU
            ):
                nodes.append(_child)
                key.pop(0)
                if len(key) == 0:
                    break
                self._match_prefix_helper(_child, key, nodes)
                break

    def _collect_leaves_for_evict(self, device: Device):
        # dfs遍历叶子节点
        # 叶子节点的要求：
        # 1. 没有在指定设备上的子节点
        # 2. 节点的物理token块在指定device上
        # 3. 节点处于STABLE状态
        ret_list: List[TreeNode] = []

        def dfs_(cur_node: TreeNode):
            if device == Device.CPU:
                num_child = len(cur_node.children)
            else:
                num_child = 0
                for x in cur_node.children.values():
                    if x.device == device:
                        num_child += 1
            if (
                num_child == 0
                and cur_node.value
                and cur_node.status
                and cur_node.device is device
                and cur_node.status is kvCacheProgressStatus.STABLE
            ):
                if cur_node.node_type == NodeType.LORA and device == Device.CPU:
                    return
                assert self.check_conditions(cur_node, device, [])
                ret_list.append(cur_node)
                return

            for x in cur_node.children.values():
                dfs_(x)

        dfs_(self.root_node)
        return ret_list

    # 找到从根节点往下第一个device为目标device的节点

    def _collect_candidates_for_promote(self, device: Device) -> List[TreeNode]:
        ret_list: List[TreeNode] = []

        def dfs_(cur_node: TreeNode):
            if (
                cur_node is not self.root_node
                and cur_node.device == device
            ):
                if cur_node.status == kvCacheProgressStatus.STABLE:
                    ret_list.append(cur_node)
                return
            else:
                for child in cur_node.children.values():
                    dfs_(child)

        dfs_(self.root_node)
        return ret_list
    @time_tracker
    def calculate_node_cost(self, node: TreeNode) -> tuple:
        # 默认实现保持与原来相同的行为（LRU）
        # 可以根据需要组合多种因素，如访问时间、使用次数、高度等
        low_lora = self.low_lora
        now_lora = 0
        for _, lora_node in self.root_node.children.items():
            if lora_node.device == Device.GPU:
                now_lora += 1
        lora_eval = max(1, low_lora/now_lora)
        if node.node_type == NodeType.LORA:
            cost_i = len(node.value)
        else:
            cost_i = 1
        prob_i = node.used_time
        current_time = time.time()
        t_i = 0.0000000000001*(current_time-node.last_access_time)
        time_i = 1-(1 / (1 + np.exp(-t_i)))
        retain_eval = cost_i*prob_i*time_i
        eval_i = lora_eval*retain_eval
        cost = (
            eval_i,
            node.last_access_time,  # 最近访问时间（越早访问越容易被驱逐）
            node.used_time,         # 使用次数（使用次数越少越容易被驱逐）
            -node.height            # 高度（高度越小越容易被驱逐）
        )
        return cost

    def check_conditions(
        self,
        current_node: TreeNode,
        device: Device,
        ret_nodes: List[TreeNode],
    ) -> bool:
        if current_node.device != device:
            return False
        if current_node.status != kvCacheProgressStatus.STABLE:
            return False
        if device == Device.CPU:
            # 不能删除LoRA
            if current_node.node_type == NodeType.LORA:
                return False
            # 子节点必须是已经在要被remove的序列里
            for _, node in current_node.children.items():
                if node not in ret_nodes:
                    return False
        if device == Device.GPU:
            # 检查其子节点要不已经是CPU,要不在将要change_device的序列里
            for _, node in current_node.children.items():
                if node not in ret_nodes and node.device == device:
                    return False
        return True

    def find(self, key: Tuple) -> Optional[TreeNode]:
        if key in self.kv_key_node_dict:
            return self.kv_key_node_dict[key]
        return

    def find_lora_node(self, lora_id: int) -> Optional[LoRATreeNode]:
        if lora_id in self.root_node.children:
            return self.root_node.children[lora_id]
        return None

    def find_helper(
        self, current_node: TreeNode, key: Tuple, node_list: List[TreeNode]
    ):
        if current_node is not self.root_node and current_node.key == key:
            node_list.append(current_node)
        for _, child_node in current_node.children.items():
            self.find_helper(child_node, key, node_list)

    def print(self):
        x = 1
        # visualize_tree(self.root_node)
        # 在程序结束时调用
        # print_function_stats()


def visualize_tree(node: TreeNode, level: int = 0, prefix: str = "", is_last: bool = True):
    """
    可视化ELoRATree的结构

    Args:
        node: 当前节点
        level: 当前层级
        prefix: 用于显示树结构的前缀字符串
        is_last: 是否是父节点的最后一个子节点
    """
    # 打印当前节点
    if node.node_type == NodeType.ROOT:
        print("Root (Device: GPU)")
    else:
        # 根据节点类型显示不同的信息
        node_info = ""
        if node.node_type == NodeType.LORA:
            lora_node = node  # type: LoRATreeNode
            blocks_count = len(lora_node.value) if lora_node.value else 0
            node_info = f"LoRA ID: {lora_node.key}, Blocks: {blocks_count}"
        elif node.node_type == NodeType.KV:
            kv_node = node  # type: KVTreeNode
            node_info = f"KV Key: {kv_node.key}, Block: {kv_node.value.block_number if kv_node.value else 'None'}"

        # 通用信息
        status_str = node.status.name if node.status else "None"
        device_str = node.device.name if hasattr(node, 'device') else "None"
        access_time = f"{time.time() - node.last_access_time:.2f}s ago"

        print(f"{prefix}{'└── ' if is_last else '├── '}{node_info}")
        print(f"{prefix}{'    ' if is_last else '│   '}Status: {status_str}, Device: {device_str}, Height: {node.height}")
        print(f"{prefix}{'    ' if is_last else '│   '}Used: {node.used_time} times, Last access: {access_time}")

    # 为子节点创建新的前缀
    if hasattr(node, 'children'):
        children = list(node.children.items())

        # 处理所有子节点
        for i, (key, child) in enumerate(children):
            child_is_last = i == len(children) - 1
            new_prefix = prefix + ("    " if is_last else "│   ")
            # 递归处理子节点
            visualize_tree(child, level + 1, new_prefix, child_is_last)

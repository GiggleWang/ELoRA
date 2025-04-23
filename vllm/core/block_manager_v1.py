"""A block manager that manages token blocks."""
from abc import ABC, abstractmethod
import copy
from itertools import count, takewhile
import logging
import math
from os.path import commonprefix
import threading
import time
from typing import Dict, List, Optional, Set, Tuple

from vllm.block import BlockTable, PhysicalTokenBlock
from vllm.core.color_print import ColorPrint, print_block_table, print_cached_blocks, print_free_pool
from vllm.core.elora_tree import ELoRATree, NodeType, print_function_stats, reset_function_stats
from vllm.core.evictor import EvictionPolicy, Evictor, make_evictor
from vllm.core.evictor_manager import ContainResult, EvictorManager, TransferLocation
from vllm.core.free_block_pool import FreeBlockPool
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.core.elora_tree import TreeNode, kvCacheProgressStatus
from vllm.logger import init_logger
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device

logger = init_logger(__name__)


class BlockAllocatorBase(ABC):
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    @abstractmethod
    def __init__(self,
                 device: Device,
                 block_size: int,
                 num_blocks: int,
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU):
        pass

    @abstractmethod
    def allocate(self,
                 block_hash: Optional[int] = None,
                 num_hashed_tokens: int = 0) -> PhysicalTokenBlock:
        pass

    @abstractmethod
    def free(self, block: PhysicalTokenBlock) -> None:
        pass

    @abstractmethod
    def get_num_free_blocks(self) -> int:
        pass

    @abstractmethod
    def contains_block(self, block_hash: int) -> bool:
        pass

    @abstractmethod
    def update_hash(self, block_hash: int, block: PhysicalTokenBlock):
        pass


class CachedBlockAllocator(BlockAllocatorBase):
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    def __init__(self,
                 device: Device,
                 block_size: int,
                 num_blocks: int,
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU) -> None:
        self.device = device
        self.block_size = block_size
        self.cached_blocks: Dict[int, PhysicalTokenBlock] = {}
        self.free_block_pool: FreeBlockPool = FreeBlockPool(num_blocks)
        self.default_hash_ctr = count()
        self.num_blocks = num_blocks

    def do_clean(self):
        self.cached_blocks.clear()
        self.free_block_pool.do_clean()
        self.default_hash_ctr = count()

    def add_free_block(self, free_block_num: int):
        self.free_block_pool.add_free_num(free_block_num)

    def allocate_block(self, block_hash: int,
                       num_hashed_tokens: int) -> PhysicalTokenBlock:
        block_num = self.free_block_pool.get_free_num()
        assert block_num != None
        block = PhysicalTokenBlock(device=self.device,
                                   block_number=block_num,
                                   block_size=self.block_size,
                                   block_hash=block_hash,
                                   num_hashed_tokens=num_hashed_tokens)
        return block

    def allocate(self,
                 block_hash: Optional[int] = None,
                 num_hashed_tokens: int = 0) -> PhysicalTokenBlock:
        if block_hash is None:
            block_hash = next(self.default_hash_ctr)
        if block_hash not in self.cached_blocks:
            write_to_file("allocate", block_hash, True)
            self.cached_blocks[block_hash] = self.allocate_block(
                block_hash, num_hashed_tokens)
        block = self.cached_blocks[block_hash]
        assert block.block_hash == block_hash
        block.ref_count += 1
        return block

    # 如果返回True,则需要移入evcitor
    # 反之则不需要
    def free(self, block: PhysicalTokenBlock) -> bool:
        if block.ref_count == 0:
            raise ValueError(f"Double free! {block} is already freed.")
        block.ref_count -= 1
        if block.ref_count == 0:
            write_to_file("free", block.block_hash, False)
            del self.cached_blocks[block.block_hash]
            return True
        return False

    def get_num_used_blocks(self) -> int:
        return len(self.cached_blocks)

    def get_num_free_blocks(self) -> int:
        raise NotImplementedError("This function is not allowed to be called.")

    def contains_block(self, block_hash: int) -> bool:
        return block_hash in self.cached_blocks

    def update_hash(self, block_hash: int, block: PhysicalTokenBlock):
        # Update the hash of block and the cached_blocks dictionary.
        assert not self.contains_block(block_hash)
        old_hash = block.block_hash
        block.block_hash = block_hash
        write_to_file("update_hash", old_hash, True)
        write_to_file("update_hash", block_hash, False)
        del self.cached_blocks[old_hash]
        self.cached_blocks[block_hash] = block


class UncachedBlockAllocator(BlockAllocatorBase):
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    def __init__(
        self,
        device: Device,
        block_size: int,
        num_blocks: int,
    ) -> None:
        self.device = device
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Initialize the free blocks.
        self.free_blocks: BlockTable = []
        for i in range(num_blocks):
            block = PhysicalTokenBlock(device=device,
                                       block_number=i,
                                       block_size=block_size,
                                       block_hash=-1,
                                       num_hashed_tokens=0)
            self.free_blocks.append(block)

    def allocate(self,
                 block_hash: Optional[int] = None,
                 num_hashed_tokens: int = 0) -> PhysicalTokenBlock:
        if not self.free_blocks:
            raise ValueError("Out of memory! No free blocks are available.")
        block = self.free_blocks.pop()
        block.ref_count = 1
        return block

    def free(self, block: PhysicalTokenBlock) -> None:
        if block.ref_count == 0:
            raise ValueError(f"Double free! {block} is already freed.")
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block)

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def contains_block(self, block_hash: int) -> bool:
        raise NotImplementedError(
            "Invalid codepath for uncached block allocator.")

    def update_hash(self, block_hash: int, block: PhysicalTokenBlock):
        raise NotImplementedError(
            "Invalid codepath for uncached block allocator.")


class BlockSpaceManagerV1(BlockSpaceManager):
    """Manages the mapping between logical and physical token blocks."""

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
        enable_caching: bool = False,
    ) -> None:
        with open('/root/vllm/vllm/core/cached_blocks.txt', 'w') as file:
            pass
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks
        ColorPrint.print(
            f"num_total_gpu_blocks:{self.num_total_gpu_blocks}", color="green")

        if enable_caching and sliding_window is not None:
            raise NotImplementedError(
                "Sliding window is not allowed with prefix caching enabled!")

        self.block_sliding_window = None
        if sliding_window is not None:
            assert sliding_window % block_size == 0, (sliding_window,
                                                      block_size)
            self.block_sliding_window = sliding_window // block_size

        self.watermark = watermark
        assert watermark >= 0.0

        self.enable_caching = enable_caching

        self.watermark_blocks = int(0.2 * num_gpu_blocks)

        if self.enable_caching:
            logger.info("Automatic prefix caching is enabled.")
            self.gpu_allocator = CachedBlockAllocator(Device.GPU, block_size,
                                                      num_gpu_blocks)
            self.cpu_allocator = CachedBlockAllocator(Device.CPU, block_size,
                                                      num_cpu_blocks)
            self.evictor_manager = EvictorManager()
        else:
            self.gpu_allocator = UncachedBlockAllocator(
                Device.GPU, block_size, num_gpu_blocks)
            self.cpu_allocator = UncachedBlockAllocator(
                Device.CPU, block_size, num_cpu_blocks)
        # Mapping: seq_id -> BlockTable.
        self.block_tables: Dict[int, BlockTable] = {}
        # Mapping: seq_id -> (transfer_id,waiting_blocks)
        self.waiting_blocks: Dict[int, Tuple[str, BlockTable]] = {}
        self.pending_num = count()
        self.pending_map: Dict[str,
                               Dict[PhysicalTokenBlock, PhysicalTokenBlock]] = {}
        self.low_percent = 0.6
        self.high_percent = 0.8
        self.satisfied_percent = 0.7
        self.elora_tree = ELoRATree()
        lora_num = 100
        self.lora_ids = list(range(0, lora_num + 1))
        self.block_num_per_lora = 3
        self.init_lora()
        # swap_id -> (list(lora_id) ,swap_type_id)
        # 1.gpu满了，换出到cpu
        # 2.gpu空了，换入到gpu
        # 3.需要时发现在cpu，换入到gpu
        self.swap_id_lora_id_type_map: Dict[str, Tuple[List[int], int]] = {}
        self.hash_helper: Dict[int, int] = {}
        self.gpu_hit_num = 0
        self.cpu_hit_num = 0
        # elora_id -> state_num
        # 0:RUNNING
        # 1:STABLE
        # 2:CPU
        self.elora_state: Dict[int, int] = {}

    def init_lora(self):
        for lora_id in self.lora_ids:
            block_list: List[PhysicalTokenBlock] = []
            for _ in range(self.block_num_per_lora):
                block = self.gpu_allocator.allocate()
                block_list.append(block)
                free_res = self.gpu_allocator.free(block)
                assert free_res == True
                self.evictor_manager.allocate(ContainResult._GPU, [block])
            self.elora_tree.insert_lora(
                kvCacheProgressStatus.STABLE, lora_id, block_list, Device.GPU)

    def generate_pending_id(self) -> str:
        """ 生成格式如'PENDING_1'的递增ID """
        current_id = next(self.pending_num)  # 获取并自增序号
        return f"PENDING_{current_id}"

    def deal_finished_events(self, in_finished: List[str], out_finished: List[str]) -> List[int]:
        ret_int: List[int] = []
        for finished_id in in_finished:
            # ColorPrint.print(f"finished:{finished_id}")
            cpu_gpu_map = self.pending_map[finished_id]
            lora_blocks: List[PhysicalTokenBlock] = []
            if finished_id in self.swap_id_lora_id_type_map:
                lora_list, type_int = self.swap_id_lora_id_type_map[finished_id]
                for lora_id in lora_list:
                    # ColorPrint.print(f"lora {lora_id}::进到位", color="cyan")
                    lora_node = self.elora_tree.find_lora_node(lora_id)
                    assert lora_node
                    lora_blocks += lora_node.value
                if type_int == 1:
                    for lora_id in lora_list:
                        self.elora_tree.update_lora(
                            lora_id, kvCacheProgressStatus.STABLE, device=Device.CPU)
                if type_int == 2:
                    for lora_id in lora_list:
                        self.elora_tree.update_lora(
                            lora_id, kvCacheProgressStatus.STABLE, device=Device.GPU)
                if type_int == 3:
                    for lora_id in lora_list:
                        self.elora_tree.update_lora(
                            lora_id, kvCacheProgressStatus.RUNNING, device=Device.GPU)
            update_keys: List[Tuple] = [
                (gpu_block.block_hash, gpu_block.num_hashed_tokens)
                for _, gpu_block in cpu_gpu_map.items()
                if gpu_block not in lora_blocks
            ]
            no_lora_map: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
            lora_map: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
            for cpu_block, gpu_block in cpu_gpu_map.items():
                if gpu_block not in lora_blocks:
                    no_lora_map[cpu_block] = gpu_block
                else:
                    lora_map[cpu_block] = gpu_block
            for cpu_block, gpu_block in lora_map.items():
                self.evictor_manager.cpu_evictor.handle_pending_out(
                    cpu_block.block_hash)
                self.evictor_manager.gpu_evictor.handle_pending_in(
                    gpu_block.block_hash)
                # self.evictor_manager.gpu_evictor.remove(gpu_block.block_hash)
                self.cpu_allocator.add_free_block(cpu_block.block_number)
            for key in update_keys:
                self.elora_tree.update_kv(
                    key, status=kvCacheProgressStatus.STABLE)
            flag: bool = any(
                finished_id == block_info[0] for block_info in self.waiting_blocks.values())
            for cpu_block, gpu_block in no_lora_map.items():
                self.evictor_manager.cpu_evictor.handle_pending_out(
                    cpu_block.block_hash)
                self.evictor_manager.gpu_evictor.handle_pending_in(
                    gpu_block.block_hash)
                self.cpu_allocator.add_free_block(cpu_block.block_number)
                if flag:
                    # 从GPU_EVICTOR 转移到 GPU_RUNNING
                    block_hash = gpu_block.block_hash
                    free_res = self.evictor_manager.free(
                        ContainResult._GPU, [block_hash])
                    block = free_res[0]
                    block.ref_count = 0
                    self.gpu_allocator.cached_blocks[block_hash] = block
                    self.elora_tree.update_kv(key=(
                        block.block_hash, block.num_hashed_tokens), status=kvCacheProgressStatus.RUNNING)
                    write_to_file("deal_finished_events", block_hash, True)
            if flag:
                waiting_result = {
                    key: value
                    for key, value in self.waiting_blocks.items()
                    if value[0] == finished_id
                }
                if len(waiting_result) != 1:
                    raise ValueError(
                        f"Expected exactly 1 matching key, but found {len(waiting_result)}")
                target_key = list(waiting_result.keys())[0]
                target_str, target_table = waiting_result[target_key]
                assert target_str == finished_id
                ret_int.append(target_key)
            # if finished_id[:3] == "now":
            #     ColorPrint.print(
            #         f"because of {finished_id}, set waiting = true")

        for finished_id in out_finished:
            # ColorPrint.print(f"finished:{finished_id}")
            lora_blocks: List[PhysicalTokenBlock] = []
            if finished_id in self.swap_id_lora_id_type_map:
                lora_list, type_int = self.swap_id_lora_id_type_map[finished_id]
                for lora_id in lora_list:
                    # ColorPrint.print(f"lora {lora_id}::出到位", color="green")
                    lora_node = self.elora_tree.find_lora_node(lora_id)
                    assert lora_node
                    lora_blocks += lora_node.value
                if type_int == 1:
                    for lora_id in lora_list:
                        self.elora_tree.update_lora(
                            lora_id, kvCacheProgressStatus.STABLE, device=Device.CPU)
                if type_int == 2:
                    for lora_id in lora_list:
                        self.elora_tree.update_lora(
                            lora_id, kvCacheProgressStatus.STABLE, device=Device.GPU)
                if type_int == 3:
                    for lora_id in lora_list:
                        self.elora_tree.update_lora(
                            lora_id, kvCacheProgressStatus.STABLE, device=Device.GPU)
            gpu_cpu_map = self.pending_map[finished_id]
            no_lora_map: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
            lora_map: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
            for gpu_block, cpu_block in gpu_cpu_map.items():
                if cpu_block not in lora_blocks:
                    no_lora_map[gpu_block] = cpu_block
                else:
                    lora_map[gpu_block] = cpu_block
            for gpu_block, cpu_block in lora_map.items():
                self.evictor_manager.gpu_evictor.handle_pending_out(
                    gpu_block.block_hash)
                self.evictor_manager.cpu_evictor.handle_pending_in(
                    cpu_block.block_hash)
                # self.evictor_manager.cpu_evictor.remove(cpu_block.block_hash)
                self.gpu_allocator.add_free_block(gpu_block.block_number)

            update_keys: List[Tuple] = [
                (cpu_block.block_hash, cpu_block.num_hashed_tokens)
                for _, cpu_block in gpu_cpu_map.items()
                if cpu_block not in lora_blocks
            ]
            for key in update_keys:
                self.elora_tree.update_kv(
                    key, status=kvCacheProgressStatus.STABLE)
            for gpu_block, cpu_block in no_lora_map.items():
                self.evictor_manager.gpu_evictor.handle_pending_out(
                    gpu_block.block_hash)
                self.evictor_manager.cpu_evictor.handle_pending_in(
                    cpu_block.block_hash)
                self.gpu_allocator.add_free_block(gpu_block.block_number)
            # if finished_id[:3] == "now":
                # ColorPrint.print(
                #     f"because of {finished_id}, set waiting = False")
        return ret_int

    def can_allocate(self, seq_group: SequenceGroup) -> AllocStatus:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        num_required_blocks = len(seq.logical_token_blocks)

        if self.block_sliding_window is not None:
            num_required_blocks = min(num_required_blocks,
                                      self.block_sliding_window)
        if self.enable_caching:
            num_free_gpu_blocks = self.num_total_gpu_blocks - \
                self.gpu_allocator.get_num_used_blocks()
            already_have_blocks = 0
            for logical_idx in range(num_required_blocks):
                block_hash = seq.hash_of_block(logical_idx)
                if self.gpu_allocator.contains_block(block_hash):
                    already_have_blocks += 1
            num_required_blocks -= already_have_blocks
        else:
            num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()

        # Use watermark to avoid frequent cache eviction.
        if (self.num_total_gpu_blocks - num_required_blocks <
                self.watermark_blocks):
            return AllocStatus.NEVER
        if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def allocate(self, seq_group: SequenceGroup) -> Optional[Tuple[str, Dict[int, int]]]:
        # 如果有返回值,那么意味着需要从CPU换入
        # NOTE: Here we assume that all sequences in the group have the same
        # prompt.
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        # Allocate new physical token blocks that will store the prompt tokens.
        num_prompt_blocks = len(seq.logical_token_blocks)
        cpu_gpu_map: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        block_table: BlockTable = []
        # TODO:后续注意第一个元素是lora
        insert_blocks: List[PhysicalTokenBlock] = []
        insert_keys: List[Tuple] = []
        _gpu_tuples: List[Tuple] = []
        cpu_to_gpu_blocks: List[PhysicalTokenBlock] = []
        lora_id = seq.elora_int_id
        lora_node = self.elora_tree.find_lora_node(lora_id)
        special_flag = False
        flag = False
        assert lora_node
        if lora_node.device == Device.GPU:
            if lora_node.value[0].block_hash in self.gpu_allocator.cached_blocks:
                lora_blocks = lora_node.value
                for block in lora_blocks:
                    self.gpu_allocator.cached_blocks[block.block_hash].ref_count += 1
            if self.evictor_manager.contains_block(lora_node.value[0].block_hash) == ContainResult._GPU:
                self.elora_tree.update_lora(
                    lora_id, kvCacheProgressStatus.RUNNING)
                lora_blocks = lora_node.value
                lora_hashs = [block.block_hash for block in lora_blocks]
                self.evictor_manager.free(ContainResult._GPU, lora_hashs)
                for block in lora_blocks:
                    self.gpu_allocator.cached_blocks[block.block_hash] = block
                    block.ref_count = 1
            if lora_node.status == kvCacheProgressStatus.PENDING:
                special_flag = True
                flag = True
        else:
            # 说明在CPU上,要实现从cpu换到gpu
            flag = True
            # ColorPrint.print(f"swap in lora:::{lora_node.key}", color="red")
            blocks: List[PhysicalTokenBlock] = lora_node.value
            new_blocks: List[PhysicalTokenBlock] = []
            for cpu_block in blocks:
                gpu_block = self.swap_evictor(True, cpu_block)
                cpu_gpu_map[cpu_block] = gpu_block
                new_blocks.append(gpu_block)
            self.elora_tree.update_lora(
                lora_id, kvCacheProgressStatus.PENDING, new_blocks, Device.GPU)

        for logical_idx in range(num_prompt_blocks):
            if (self.block_sliding_window is not None
                    and logical_idx >= self.block_sliding_window):
                block = block_table[logical_idx % self.block_sliding_window]
                # Set the reference counts of the token blocks.
                block.ref_count = seq_group.num_seqs()
            elif self.enable_caching:
                block_hash = seq.hash_of_block(logical_idx)
                if block_hash in self.hash_helper:
                    assert self.hash_helper[block_hash] == seq.elora_int_id
                else:
                    self.hash_helper[block_hash] = seq.elora_int_id
                num_hashed_tokens = seq.num_hashed_tokens_of_block(logical_idx)
                contain_res = self.evictor_manager.contains_block(block_hash)
                if contain_res is ContainResult._NOT:
                    _write_to_file("1")
                    block = self.gpu_allocator.allocate(
                        block_hash, num_hashed_tokens)
                if contain_res is ContainResult._GPU:
                    self.gpu_hit_num += 1
                    free_res = self.evictor_manager.free(
                        ContainResult._GPU, [block_hash])
                    block = free_res[0]
                    assert block.ref_count == 0
                    block.ref_count += 1
                    self.gpu_allocator.cached_blocks[block_hash] = block
                    write_to_file("block_manager_allocate", block_hash, True)
                    _gpu_tuples.append(
                        (block.block_hash, block.num_hashed_tokens))
                if contain_res is ContainResult._CPU:
                    self.cpu_hit_num += 1
                    cpu_block: PhysicalTokenBlock = self.evictor_manager.cpu_evictor.get_block(
                        block_hash)
                    if cpu_block in cpu_gpu_map:
                        continue
                    block: PhysicalTokenBlock = self.swap_evictor(
                        True, cpu_block)
                    assert block.device is Device.GPU
                    cpu_gpu_map[cpu_block] = block
                    cpu_to_gpu_blocks.append(block)
                insert_blocks.append(block)
                insert_keys.append((block_hash, num_hashed_tokens))
            else:
                block = self.gpu_allocator.allocate()
                # Set the reference counts of the token blocks.
                block.ref_count = seq_group.num_seqs()
            block_table.append(block)
        if cpu_to_gpu_blocks:
            for cpu_to_gpu_block in cpu_to_gpu_blocks:
                self.elora_tree.update_kv(
                    (cpu_to_gpu_block.block_hash, cpu_to_gpu_block.num_hashed_tokens), cpu_to_gpu_block, kvCacheProgressStatus.PENDING)
        if _gpu_tuples:
            for gpu_tuple in _gpu_tuples:
                self.elora_tree.update_kv(
                    gpu_tuple, status=kvCacheProgressStatus.RUNNING)
        if insert_blocks and insert_keys:
            self.elora_tree.insert_kv(lora_id, insert_keys, insert_blocks)
        if cpu_gpu_map or special_flag:
            # 说明需要进行swap in
            pending_id: str = self.generate_pending_id()
            for block in block_table:
                block.ref_count -= 1
            if flag:
                self.swap_id_lora_id_type_map[pending_id] = ([lora_id], 3)
            # waiting_blocks代表等待换入之后可以直接进入RUNNING队列的blocks
            # print(f"{pending_id}:::swap in")
            self.waiting_blocks[seq.seq_id] = (pending_id, block_table.copy())
            self.pending_map[pending_id] = cpu_gpu_map
            id_map: Dict[int, int] = {}
            for key, value in cpu_gpu_map.items():
                id_map[key.block_number] = value.block_number
            return (pending_id, id_map)
        # Assign the block table for each sequence.
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            # print_block_table(block_table)
            # print_free_pool(self.gpu_allocator.free_block_pool.free_queue)
            self.block_tables[seq.seq_id] = block_table.copy()

    def can_append_slot(self, seq_group: SequenceGroup) -> bool:
        # Simple heuristic: If there is at least one free block
        # for each sequence, we can append.
        if self.enable_caching:
            num_free_gpu_blocks = self.num_total_gpu_blocks - \
                self.gpu_allocator.get_num_used_blocks()
        else:
            num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)
        return num_seqs <= num_free_gpu_blocks

    def _promote_last_block(
        self,
        seq: Sequence,
        last_block: PhysicalTokenBlock,
    ) -> PhysicalTokenBlock:
        # 这个block已经满了
        assert self.enable_caching

        # Compute a new hash for the block so that it can be shared by other
        # Sequences
        new_hash = seq.hash_of_block(len(seq.logical_token_blocks) - 1)
        if new_hash in self.hash_helper:
            assert self.hash_helper[new_hash] == seq.elora_int_id
        else:
            self.hash_helper[new_hash] = seq.elora_int_id
        # if new_hash is already in the cached table, then free last_block
        # and return the cached version
        # FIXME:(yixiao) 这里可以加一下查询块是不是在CPUevictor中
        if self.gpu_allocator.contains_block(new_hash):
            # 说明可以复用
            free_res = self.gpu_allocator.free(last_block)
            if free_res:
                self.elora_tree.remove(
                    [(last_block.block_hash, last_block.num_hashed_tokens)])
            _write_to_file("2")
            return self.gpu_allocator.allocate(new_hash)
        if self.evictor_manager.contains_block(new_hash) is ContainResult._GPU:
            # 说明在GPU的evictor里,转移到gpu-allocator的dict来
            self.gpu_allocator.free(last_block)
            free_res = self.evictor_manager.free(
                ContainResult._GPU, [new_hash])
            block = free_res[0]
            assert block.ref_count == 0
            block.ref_count += 1
            self.gpu_allocator.cached_blocks[new_hash] = block
            write_to_file("promote_last_block", new_hash, True)
            self.elora_tree.update_kv(
                (last_block.block_hash, last_block.num_hashed_tokens), status=kvCacheProgressStatus.RUNNING)
            return block
        else:
            old_block = copy.deepcopy(last_block)
            self.gpu_allocator.update_hash(new_hash, last_block)
            new_block = last_block
            self.elora_tree.update_last_block(old_block, new_block)
            return last_block

    def _is_last_block_full(
        self,
        seq: Sequence,
    ) -> bool:
        token_ids_len = len(seq.data.get_token_ids())
        return token_ids_len > 0 and token_ids_len % seq.block_size == 0

    def _maybe_promote_last_block(
        self,
        seq: Sequence,
        last_block: PhysicalTokenBlock,
    ) -> PhysicalTokenBlock:
        if self._is_last_block_full(seq):
            return self._promote_last_block(seq, last_block)
        else:
            return last_block

    def _allocate_last_physical_block(
        self,
        seq: Sequence,
    ) -> PhysicalTokenBlock:
        # Called before a new block is appended.
        # This is in charge of allocating a new physical block (to be appended).

        # None if the last block is not full. Otherwise, we set it to the
        # content hash.
        if not self.enable_caching:
            return self.gpu_allocator.allocate()
        block_hash: Optional[int] = None
        if (self._is_last_block_full(seq)):
            block_hash = seq.hash_of_block(len(seq.logical_token_blocks) - 1)
            if block_hash in self.hash_helper:
                assert self.hash_helper[block_hash] == seq.elora_int_id
            else:
                self.hash_helper[block_hash] = seq.elora_int_id
        num_hashed_tokens = seq.num_hashed_tokens_of_block(
            len(seq.logical_token_blocks) - 1)

        # num_hashed_tokens is used to compute future hashes
        # (e.g. in the hashing function, it is used to ask the sequence for
        # prefix tokens)
        if block_hash is not None and self.evictor_manager.contains_block(block_hash) is ContainResult._GPU:
            free_res = self.evictor_manager.free(
                ContainResult._GPU, [block_hash])
            block = free_res[0]
            assert block.ref_count == 0
            block.ref_count += 1
            self.gpu_allocator.cached_blocks[block_hash] = block
            write_to_file("allocate_last_physical_block", block_hash, True)
            self.elora_tree.update_kv(
                (block.block_hash, block.num_hashed_tokens), status=kvCacheProgressStatus.RUNNING)
            return block
        new_block = self.gpu_allocator.allocate(block_hash, num_hashed_tokens)

        # If the block has is None, then the block is not full.
        # If the block is not full, then we expect it to have a refcount of 1.
        if block_hash is None:
            assert new_block.ref_count == 1
        return new_block

    def append_slot(
        self,
        seq: Sequence,
    ) -> Optional[Tuple[int, int]]:
        """Allocate a physical slot for a new token."""
        logical_blocks = seq.logical_token_blocks
        block_table = self.block_tables[seq.seq_id]
        # If we need to allocate a new physical block
        if len(block_table) < len(logical_blocks):
            # 如果理论需要的block数量比实际已经用的多,说明最新的需要新建一个block
            # Currently this code only supports adding one physical block
            assert len(block_table) == len(logical_blocks) - 1

            if (self.block_sliding_window
                    and len(block_table) >= self.block_sliding_window):
                # reuse a block
                block_table.append(block_table[len(block_table) %
                                               self.block_sliding_window])
            else:
                # The sequence has a new logical block.
                # Allocate a new physical block.
                new_block = self._allocate_last_physical_block(seq)
                block_table.append(new_block)
                block_keys: List[Tuple] = [
                    (block.block_hash, block.num_hashed_tokens) for block in block_table]
                self.elora_tree.insert_kv(
                    seq.elora_int_id, block_keys, block_table)
                return None

        # We want to append the token to the last physical block.
        last_block = block_table[-1]
        assert last_block.device == Device.GPU
        if last_block.ref_count == 1:
            # Not shared with other sequences. Appendable.
            # 直接修改即可
            if self.enable_caching:
                # If the last block is now complete, we may reuse an old block
                # to save memory.
                maybe_new_block = self._maybe_promote_last_block(
                    seq, last_block)
                block_table[-1] = maybe_new_block
                # block_keys: List[Tuple] = [
                #     (block.block_hash, block.num_hashed_tokens) for block in block_table]
                # self.elora_tree.insert_kv2(
                #     seq.elora_int_id, block_keys, block_table)
            return None
        else:
            # The last block is shared with other sequences.
            # Copy on Write: Allocate a new block and copy the tokens.
            new_block = self._allocate_last_physical_block(seq)

            block_table[-1] = new_block
            block_keys: List[Tuple] = [
                (block.block_hash, block.num_hashed_tokens) for block in block_table]
            self.elora_tree.insert_kv(
                seq.elora_int_id, block_keys, block_table)
            self.gpu_allocator.free(last_block)
            if self.enable_caching:
                if not self.gpu_allocator.contains_block(last_block.block_hash):
                    self.elora_tree.remove(
                        [(last_block.block_hash, last_block.num_hashed_tokens)])
                    self.gpu_allocator.add_free_block(last_block.block_number)
            return last_block.block_number, new_block.block_number

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        # NOTE: fork does not allocate a new physical block.
        # Thus, it is always safe from OOM.
        src_block_table = self.block_tables[parent_seq.seq_id]
        self.block_tables[child_seq.seq_id] = src_block_table.copy()
        # When using a sliding window, blocks will be eventually reused.
        # In this case the block tables will contain repeated blocks.
        # When forking, we must make sure that each block's `ref_count`
        # is only incremented by one, so we deduplicate them by wrapping
        # them in a set.
        for block in set(src_block_table):
            block.ref_count += 1

    def _get_physical_blocks(
            self, seq_group: SequenceGroup) -> List[PhysicalTokenBlock]:
        # NOTE: Here, we assume that the physical blocks are only shared by
        # the sequences in the same group.
        blocks: Set[PhysicalTokenBlock] = set()
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue
            blocks.update(self.block_tables[seq.seq_id])
        return list(blocks)

    def can_swap_in(self, seq_group: SequenceGroup) -> bool:
        blocks = self._get_physical_blocks(seq_group)
        num_swapped_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
        num_free_blocks = len(self.gpu_allocator.free_block_pool.free_queue)
        # NOTE: Conservatively, we assume that every sequence will allocate
        # at least one free block right after the swap-in.
        # NOTE: This should match the logic in can_append_slot().
        num_required_blocks = len(blocks) + num_swapped_seqs
        return num_free_blocks - num_required_blocks >= self.watermark_blocks

    def swap_in(self, seq_group: SequenceGroup) -> Dict[int, int]:
        # CPU block -> GPU block.
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]

            for cpu_block in block_table:
                if cpu_block in mapping:
                    gpu_block = mapping[cpu_block]
                    gpu_block.ref_count += 1
                else:
                    _write_to_file("3")
                    gpu_block = self.gpu_allocator.allocate(
                        cpu_block.block_hash, cpu_block.num_hashed_tokens)
                    mapping[cpu_block] = gpu_block
                new_block_table.append(gpu_block)
                # Free the CPU block swapped in to GPU.
                self.cpu_allocator.free(cpu_block)
                if self.enable_caching:
                    self.cpu_allocator.add_free_block(cpu_block.block_hash)
            self.block_tables[seq.seq_id] = new_block_table

        block_number_mapping = {
            cpu_block.block_number: gpu_block.block_number
            for cpu_block, gpu_block in mapping.items()
        }
        return block_number_mapping

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        blocks = self._get_physical_blocks(seq_group)
        return len(blocks) <= len(self.cpu_allocator.free_block_pool.free_queue)

    def swap_out(self, seq_group: SequenceGroup) -> Dict[int, int]:
        # GPU block -> CPU block.
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]

            for gpu_block in block_table:
                if gpu_block in mapping:
                    cpu_block = mapping[gpu_block]
                    cpu_block.ref_count += 1
                else:
                    cpu_block = self.cpu_allocator.allocate(
                        gpu_block.block_hash, gpu_block.num_hashed_tokens)
                    mapping[gpu_block] = cpu_block
                new_block_table.append(cpu_block)
                # Free the GPU block swapped out to CPU.
                self.gpu_allocator.free(gpu_block)
                if self.enable_caching:
                    self.gpu_allocator.add_free_block(gpu_block.block_hash)
            self.block_tables[seq.seq_id] = new_block_table

        block_number_mapping = {
            gpu_block.block_number: cpu_block.block_number
            for gpu_block, cpu_block in mapping.items()
        }
        return block_number_mapping

    def _free_block_table(self, block_table: BlockTable) -> None:
        # 首先将block_table的每个元素输出到block_table.txt文件中
        # when using a sliding window, each seq will only use up
        # to `self.block_sliding_window` blocks. When freeing
        # the block table, we must make sure to not free blocks more
        # than once. If no sliding window is used, there is no block
        # reuse in the block table, so we must free all blocks.
        blocks_to_free = (block_table[-self.block_sliding_window:]
                          if self.block_sliding_window is not None else
                          block_table)
        for block in set(blocks_to_free):
            if block.device == Device.GPU:
                if self.enable_caching:
                    free_res = self.gpu_allocator.free(block)
                    if free_res:
                        self.evictor_manager.allocate(
                            ContainResult._GPU, [block])
                        self.elora_tree.update_kv(
                            (block.block_hash, block.num_hashed_tokens), status=kvCacheProgressStatus.STABLE)
                else:
                    self.gpu_allocator.free(block)
            else:
                self.cpu_allocator.free(block)
                if self.enable_caching:
                    if not self.cpu_allocator.contains_block(block.block_hash):
                        self.evictor_manager.allocate(
                            ContainResult._CPU, [block])
                        self.elora_tree.update_kv(
                            (block.block_hash, block.num_hashed_tokens), status=kvCacheProgressStatus.STABLE)
        # self.elora_tree.print()

    def free(self, seq: Sequence) -> None:
        # print(f"free:::{seq.seq_id}")
        lora_node = self.elora_tree.find_lora_node(seq.elora_int_id)
        free_results = []
        for block in lora_node.value:
            result = self.gpu_allocator.free(block)
            free_results.append(result)
        if free_results[0]:
            all_true = all(free_results)
            assert all_true
            self.evictor_manager.allocate(
                ContainResult._GPU, lora_node.value.copy())
            self.elora_tree.update_lora(
                seq.elora_int_id, progressStatus=kvCacheProgressStatus.STABLE)
        if seq.seq_id not in self.block_tables:
            # Already freed or haven't been scheduled yet.
            return
        block_table = self.block_tables[seq.seq_id]
        self._free_block_table(block_table)
        del self.block_tables[seq.seq_id]

    def reset(self) -> None:
        for block_table in self.block_tables.values():
            self._free_block_table(block_table)
        self.block_tables.clear()

    def get_block_table(self, seq: Sequence) -> List[int]:
        block_table = self.block_tables[seq.seq_id]
        return [block.block_number for block in block_table]

    def get_num_free_gpu_blocks(self) -> int:
        if self.enable_caching:
            return self.num_total_gpu_blocks-self.gpu_allocator.get_num_used_blocks()
        return self.gpu_allocator.get_num_free_blocks()

    def get_num_free_cpu_blocks(self) -> int:
        if self.enable_caching:
            return self.num_total_cpu_blocks-self.cpu_allocator.get_num_used_blocks()
        return self.cpu_allocator.get_num_free_blocks()

    def access_all_blocks_in_seq(
        self,
        seq: Sequence,
        access_time: float,
    ) -> None:
        if self.enable_caching:
            # Update the last accessed time of all the blocks accessed
            # in this step.
            block_table = self.block_tables[seq.seq_id]
            for block in block_table:
                block.last_accessed = access_time

    def compute_full_blocks_in_seq(self, seq: Sequence):
        # self.elora_tree.print()
        # print(seq.seq_id)
        if seq.seq_id not in self.block_tables:
            return
        max_full_block = seq.get_len() // self.block_size - 1
        block_table = self.block_tables[seq.seq_id]
        if max_full_block == -1:
            return
        for i in reversed(range(max_full_block)):
            if block_table[i].computed:
                break
            block_table[i].computed = True

    def get_all_computed_blocks(self, seq: Sequence) -> List[int]:
        if seq.seq_id not in self.block_tables:
            return []
        block_table = self.block_tables[seq.seq_id]
        # NOTE We exclude the last block to avoid the case where the entire
        # prompt is cached. This would cause erroneous behavior in model
        # runner.
        return [
            b.block_number
            for b in takewhile(lambda b: b.computed, block_table[:-1])
        ]

    def get_common_computed_block_ids(self, seqs: List[Sequence]) -> List[int]:
        """Return the block ids that are common for a given sequence group.

        Used in prefill (can skip prefill of some blocks).
        """
        # Can return non-empty result only with prefix caching enabled.
        if not self.enable_caching:
            return []

        ids_list = [self.get_all_computed_blocks(seq) for seq in seqs]
        return commonprefix([ids for ids in ids_list if ids != []])

    def mark_blocks_as_computed(self, seq_group: SequenceGroup):
        if self.enable_caching:
            for seq in seq_group.seqs_dict.values():
                self.compute_full_blocks_in_seq(seq)

    def swap_evictor(self, flag: bool, block: PhysicalTokenBlock) -> PhysicalTokenBlock:
        # flag = true: from cpu -> gpu
        # flag = false:from gpu -> cpu
        if flag:
            # assert self.evictor_manager.contains_block(
            #     block.block_hash) == ContainResult._CPU
            self.evictor_manager.cpu_evictor.add_pending_out(block.block_hash)
            _write_to_file("4")
            gpu_block: PhysicalTokenBlock = self.gpu_allocator.allocate(
                block.block_hash, block.num_hashed_tokens)
            self.gpu_allocator.free(gpu_block)
            self.evictor_manager.gpu_evictor.add_pending_in(gpu_block)
            return gpu_block
        else:
            # assert self.evictor_manager.contains_block(
            #     block.block_hash) == ContainResult._GPU
            # self.check_cpu_full()
            self.evictor_manager.gpu_evictor.add_pending_out(block.block_hash)
            cpu_block: PhysicalTokenBlock = self.cpu_allocator.allocate(
                block.block_hash, block.num_hashed_tokens)
            self.cpu_allocator.free(cpu_block)
            self.evictor_manager.cpu_evictor.add_pending_in(cpu_block)
            return cpu_block

    def check_gpu_full(self) -> Optional[Tuple[bool, str, Dict[int, int]]]:
        # bool = true: from cpu -> gpu
        # bool = false:from gpu -> cpu
        gpu_high = self.high_percent * self.num_total_gpu_blocks
        gpu_low = self.low_percent * self.num_total_gpu_blocks
        gpu_satisfied = self.satisfied_percent * self.num_total_gpu_blocks
        gpu_used = self.num_total_gpu_blocks - \
            len(self.gpu_allocator.free_block_pool.free_queue)
        if gpu_used > gpu_high:
            # gpu -> cpu
            gpu_cpu_map: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
            swap_out_num: int = math.ceil(gpu_used-gpu_satisfied)
            swap_out_nodes = self.elora_tree.evict(swap_out_num, Device.GPU)
            lora_node_list: List[PhysicalTokenBlock] = []
            block_list: List[PhysicalTokenBlock] = []
            for node in swap_out_nodes:
                if node.node_type == NodeType.LORA:
                    lora_node_list.append(node)
                else:
                    block_list.append(node.value)
            if lora_node_list:
                for lora_node in lora_node_list:
                    lora_id: int = lora_node.key
                    blocks: List[PhysicalTokenBlock] = lora_node.value
                    new_blocks: List[PhysicalTokenBlock] = []
                    for gpu_block in blocks:
                        cpu_block = self.swap_evictor(False, gpu_block)
                        gpu_cpu_map[gpu_block] = cpu_block
                        new_blocks.append(cpu_block)
                    self.elora_tree.update_lora(
                        lora_id, kvCacheProgressStatus.PENDING, new_blocks, Device.CPU)
            to_update_blocks: List[PhysicalTokenBlock] = []
            for gpu_block in block_list:
                assert gpu_block not in gpu_cpu_map
                cpu_block: PhysicalTokenBlock = self.swap_evictor(
                    False, gpu_block)
                assert cpu_block.device is Device.CPU
                gpu_cpu_map[gpu_block] = cpu_block
                to_update_blocks.append(cpu_block)
            if to_update_blocks:
                for update_block in to_update_blocks:
                    self.elora_tree.update_kv(
                        (update_block.block_hash, update_block.num_hashed_tokens), update_block, kvCacheProgressStatus.PENDING)
            if gpu_cpu_map:
                key_str = "now_"+self.generate_pending_id()
                if lora_node_list:
                    lora_id_list = [_node.key for _node in lora_node_list]
                    self.swap_id_lora_id_type_map[key_str] = (lora_id_list, 1)
                self.pending_map[key_str] = gpu_cpu_map
                block_number_mapping = {
                    gpu_block.block_number: cpu_block.block_number
                    for gpu_block, cpu_block in gpu_cpu_map.items()
                }
                # print(f"{key_str}:::gpu used too much, let us try to move some out")
                return (False, key_str, block_number_mapping)

        if gpu_used < gpu_low:
            # cpu -> gpu
            cpu_gpu_map: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
            swap_in_num: int = math.ceil(gpu_satisfied-gpu_used)
            swap_in_nodes = self.elora_tree.promote(swap_in_num, Device.CPU)
            lora_node_list: List[PhysicalTokenBlock] = []
            block_list: List[PhysicalTokenBlock] = []
            for node in swap_in_nodes:
                if node.node_type == NodeType.LORA:
                    lora_node_list.append(node)
                else:
                    block_list.append(node.value)
            if lora_node_list:
                for lora_node in lora_node_list:
                    lora_id: int = lora_node.key
                    # ColorPrint.print(
                    #     f"swap out lora:::{lora_id}", color="blue")
                    blocks: List[PhysicalTokenBlock] = lora_node.value
                    new_blocks: List[PhysicalTokenBlock] = []
                    for cpu_block in blocks:
                        gpu_block = self.swap_evictor(True, cpu_block)
                        cpu_gpu_map[cpu_block] = gpu_block
                        new_blocks.append(gpu_block)
                    self.elora_tree.update_lora(
                        lora_id, kvCacheProgressStatus.PENDING, new_blocks, Device.GPU)
            to_update_blocks: List[PhysicalTokenBlock] = []
            for cpu_block in block_list:
                assert cpu_block not in cpu_gpu_map
                gpu_block: PhysicalTokenBlock = self.swap_evictor(
                    True, cpu_block)
                assert gpu_block.device == Device.GPU
                cpu_gpu_map[cpu_block] = gpu_block
                to_update_blocks.append(gpu_block)
            if to_update_blocks:
                for block in to_update_blocks:
                    self.elora_tree.update_kv(
                        (block.block_hash, block.num_hashed_tokens), block, kvCacheProgressStatus.PENDING)

            if cpu_gpu_map:
                key_str = "now_"+self.generate_pending_id()
                if lora_node_list:
                    lora_id_list = [_node.key for _node in lora_node_list]
                    self.swap_id_lora_id_type_map[key_str] = (lora_id_list, 2)
                self.pending_map[key_str] = cpu_gpu_map
                block_number_mapping = {
                    cpu_block.block_number: gpu_block.block_number
                    for cpu_block, gpu_block in cpu_gpu_map.items()
                }
                # print("gpu used too little, let us try to move some in")
                return (True, key_str, block_number_mapping)

    def check_cpu_full(self):
        cpu_used = self.num_total_cpu_blocks - \
            len(self.cpu_allocator.free_block_pool.free_queue)
        cpu_high = self.high_percent * self.num_total_cpu_blocks
        cpu_satisfied = self.satisfied_percent * self.num_total_cpu_blocks
        if cpu_used > cpu_high:
            removed_block_num = math.ceil(cpu_used - cpu_satisfied)  # 想上取整
            node_list: List[TreeNode] = self.elora_tree.evict(
                removed_block_num, Device.CPU)
            lora_node_list: List[PhysicalTokenBlock] = []
            block_hash_list: List[int] = []
            block_node_list: List[PhysicalTokenBlock] = []
            for node in node_list:
                if node.node_type == NodeType.LORA:
                    lora_node_list.append(node)
                else:
                    block_node_list.append(node)
            remove_list: List[Tuple] = (node.key for node in block_node_list)
            for node in block_node_list:
                self.cpu_allocator.add_free_block(
                    node.value.block_number)
            freed_list = self.elora_tree.remove(remove_list)
            for item in freed_list:
                block_hash_list.append(item[0])
            self.evictor_manager.free(ContainResult._CPU, block_hash_list)

            assert self.num_total_cpu_blocks - \
                len(self.cpu_allocator.free_block_pool.free_queue) <= cpu_satisfied

    def do_clean(self):
        self.gpu_allocator.do_clean()
        self.cpu_allocator.do_clean()
        self.evictor_manager.do_clean()
        self.block_tables.clear()
        # Mapping: seq_id -> (transfer_id,waiting_blocks)
        self.waiting_blocks.clear()
        self.pending_num = count()
        self.pending_map.clear()
        self.elora_tree = ELoRATree()
        lora_num = 100
        self.lora_ids = list(range(0, lora_num + 1))
        self.block_num_per_lora = 3
        self.init_lora()
        self.swap_id_lora_id_type_map.clear()
        self.hash_helper.clear()
        ColorPrint.print(f"gpu_hit_num:{self.gpu_hit_num}", color="red")
        ColorPrint.print(f"cpu_hit_num:{self.cpu_hit_num}", color="red")
        self.gpu_hit_num = 0
        self.cpu_hit_num = 0
        print_function_stats()
        reset_function_stats()

    def check_full(self) -> Optional[Tuple[bool, str, Dict[int, int]]]:
        self.check_cpu_full()
        return self.check_gpu_full()


def write_to_file(prefix: str, hash: int, in_or_not: bool):
    #     with open('/root/vllm/vllm/core/cached_blocks.txt', 'a') as file:
    #         file.write(f"{prefix}: {hash}")
    #         if in_or_not:
    #             file.write(" IN\n\n")
    #         else:
    #             file.write(" OUT\n\n")
    #     x = 1
    pass


def _write_to_file(word: str):
    # with open('/root/vllm/vllm/core/cached_blocks.txt', 'a') as file:
    #     file.write(f"{word }")
    pass

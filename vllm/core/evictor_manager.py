from enum import Enum
from typing import Dict, Optional
from vllm.block import PhysicalTokenBlock
from vllm.block import BlockTable
from vllm.core.block import block_table

from vllm.core.evictor import EvictionPolicy, Evictor, make_evictor
from vllm.utils import Device


class ContainResult(Enum):
    _NOT = 0
    _GPU = 1
    _CPU = 2


class TransferLocation(Enum):
    CPU_EVCITOR = 0
    CPU_RUNNING = 1
    GPU_EVICTOR = 2
    GPU_RUNNING = 3


class EvictorManager:
    def __init__(self):
        self.cpu_evictor: Evictor = make_evictor(EvictionPolicy.LRU)
        self.gpu_evictor: Evictor = make_evictor(EvictionPolicy.LRU)
    def do_clean(self):
        self.cpu_evictor.do_clean()
        self.gpu_evictor.do_clean()

    def allocate(self, type: ContainResult, blocks: list[PhysicalTokenBlock]):
        # 传入进来的blocks应该是已经在对应的device allocate好的块
        assert type != ContainResult._NOT
        if type is ContainResult._CPU:
            for block in blocks:
                assert block.device == Device.CPU
                self.cpu_evictor.add(block)
        else:
            for block in blocks:
                assert block.device == Device.GPU
                self.gpu_evictor.add(block)

    def free(self, type: ContainResult, block_hashs: list[int]) -> BlockTable:
        assert type != ContainResult._NOT
        res: BlockTable = []
        if type is ContainResult._CPU:
            for block_hash in block_hashs:
                block = self.cpu_evictor.remove(block_hash)
                res.append(block)
        else:
            for block_hash in block_hashs:
                block = self.gpu_evictor.remove(block_hash)
                res.append(block)
        return res

    def get_num_used_blocks(self, type: ContainResult) -> int:
        assert type != ContainResult._NOT
        if type is ContainResult._CPU:
            return self.cpu_evictor.num_blocks
        else:
            return self.gpu_evictor.num_blocks

    def contains_block(self, block_hash: int) -> ContainResult:
        if self.gpu_evictor.__contains__(block_hash):
            return ContainResult._GPU
        if self.cpu_evictor.__contains__(block_hash):
            return ContainResult._CPU
        return ContainResult._NOT

    def evict(self, type: ContainResult) -> PhysicalTokenBlock:
        assert type != ContainResult._NOT
        if type is ContainResult._CPU:
            return self.cpu_evictor.evict()
        else:
            return self.gpu_evictor.evict()
        
    def get_next_evicted_block(self, type: ContainResult) -> PhysicalTokenBlock:
        assert type != ContainResult._NOT
        if type is ContainResult._CPU:
            return self.cpu_evictor.get_next_evicted_block()
        else:
            return self.gpu_evictor.get_next_evicted_block()
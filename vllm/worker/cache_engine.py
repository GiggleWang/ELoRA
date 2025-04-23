"""CacheEngine class for managing the KV cache."""
from typing import Dict, List, Tuple

import torch
from torch.cuda import Event
from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.core.color_print import ColorPrint
from vllm.core.elora_tree import time_tracker
from vllm.logger import init_logger
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, is_pin_memory_available

logger = init_logger(__name__)


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        self.attn_backend = get_attn_backend(model_config.dtype)

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache(self.num_gpu_blocks, "cuda")
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")

        # Initialize the stream for caching operations.
        self.swap_in_stream = torch.cuda.Stream()
        self.swap_out_stream = torch.cuda.Stream()

        # Initialize the events for stream synchronization.
        self.swap_in_events: Dict[str, Event] = {}
        self.swap_out_events: Dict[str, Event] = {}

        self.completed_in_events: List[str] = []
        self.completed_out_events: List[str] = []

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_heads, self.head_size)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []
        for _ in range(self.num_layers):
            kv_cache.append(
                torch.empty(kv_cache_shape,
                            dtype=self.dtype,
                            pin_memory=pin_memory,
                            device=device))
        return kv_cache

    def swap_in_(self, src_to_dst: Dict[int, int]) -> None:
        for i in range(self.num_layers):
            self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                          src_to_dst)

    def swap_out_(self, src_to_dst: Dict[int, int]) -> None:
        for i in range(self.num_layers):
            self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)

    # 异步进行swap_in
    def swap_in(self, src_to_dst: Dict[int, int], key: str) -> None:
        # print("swap_in::::::::::::::::::::::::::::::::::::::::::::::::::::")
        if key == "":
            self.swap_in_(src_to_dst)
        else:
            with torch.cuda.stream(self.swap_in_stream):
                for i in range(self.num_layers):
                    self.attn_backend.swap_blocks(
                        self.cpu_cache[i], self.gpu_cache[i], src_to_dst)
                event = Event()
                event.record()
            self.swap_in_events[key] = event
        # if key[:3] == "now":
        #     ColorPrint.print(f"must wait until {key} stopped", "red")
        #     self.wait_for_swap_events([key])

    # 异步进行swap_out
    def swap_out(self, src_to_dst: Dict[int, int], key: str) -> None:
        # print("swap_out::::::::::::::::::::::::::::::::::::::::::::::::::::")
        if key == "":
            self.swap_out_(src_to_dst)
        else:
            with torch.cuda.stream(self.swap_out_stream):
                for i in range(self.num_layers):
                    self.attn_backend.swap_blocks(
                        self.gpu_cache[i], self.cpu_cache[i], src_to_dst)
                event = Event()
                event.record()
            self.swap_out_events[key] = event
        # if key[:3] == "now":
        #     ColorPrint.print(f"must wait until {key} stopped1", "red")
        #     self.wait_for_swap_events([key])
        #     ColorPrint.print(f"must wait until {key} stopped2", "red")

    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

    # 将某些events直接同步处理
    # @time_tracker
    def wait_for_swap_events(self, swap_events_to_wait: List[str]) -> None:
        for key in swap_events_to_wait:
            if key in self.swap_in_events:
                event = self.swap_in_events.pop(key)
                event.synchronize()
                ColorPrint.print(f"{key} stopped!", "blue")
                self.completed_in_events.append(key)
            if key in self.swap_out_events:
                event = self.swap_out_events.pop(key)
                event.synchronize()
                ColorPrint.print(f"{key} stopped!", "blue")
                self.completed_out_events.append(key)

    # 检查并收集已完成的数据交换事件
    def check_finished_events(self) -> Tuple[List[str], List[str]]:
        _in_finished_ids: List[str] = []
        _out_finished_ids: List[str] = []
        # _in
        for key, event in self.swap_in_events.items():
            if event.query():
                _in_finished_ids.append(key)
            else:
                break
        for key in _in_finished_ids:
            self.swap_in_events.pop(key)

        # _out
        for key, event in self.swap_out_events.items():
            if event.query():
                _out_finished_ids.append(key)
            else:
                break
        for key in _out_finished_ids:
            self.swap_out_events.pop(key)

        _in_finished_ids += self.completed_in_events.copy()
        _out_finished_ids += self.completed_out_events.copy()
        self.completed_in_events.clear()
        self.completed_out_events.clear()
        return (_in_finished_ids, _out_finished_ids)

    @staticmethod
    def get_cache_block_size(
        block_size: int,
        cache_dtype: str,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        if cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        dtype_size = _get_dtype_size(dtype)
        return dtype_size * total


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()

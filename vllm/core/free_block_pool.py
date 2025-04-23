from collections import deque
from typing import Optional


class FreeBlockPool:
    def __init__(self, max_block_num: int):
        self.max_block_num = max_block_num
        self.free_queue = deque()
        # 初始化队列，放入0到max_block_num-1的数字
        for i in range(max_block_num):
            self.free_queue.append(i)
    def do_clean(self):
        self.free_queue.clear()
        for i in range(self.max_block_num):
            self.free_queue.append(i)

    def get_free_num(self) -> Optional[int]:
        if self.free_queue:
            return self.free_queue.popleft()

    def add_free_num(self, free_num: int):
        self.free_queue.append(free_num)

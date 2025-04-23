import time
from collections import defaultdict
from typing import Dict, List, Tuple


class LoraCostManager:
    """
    管理LoRA访问记录并计算Low_lora值的成本管理器
    """
    
    def __init__(self, time_window: float = 5.0):
        """
        初始化LoRA成本管理器
        
        Args:
            time_window: 记录保留的时间窗口（秒），默认为5秒
        """
        # 存储访问记录的列表，每个记录为 (timestamp, lora_id)
        self.access_records: List[Tuple[float, str]] = []
        # 时间窗口（秒）
        self.time_window = time_window
        # 上次清理的时间
        self.last_cleanup_time = time.time()
    
    def record_access(self, lora_id: str) -> None:
        """
        记录一次LoRA访问，不进行时间检查，直接插入
        
        Args:
            lora_id: 被访问的LoRA的ID
        """
        current_time = time.time()
        # 添加新的访问记录，不进行清理
        self.access_records.append((current_time, lora_id))
        
        # 每100条记录或者距离上次清理超过10秒时进行一次清理
        # 这样可以避免列表无限增长，同时不会频繁清理影响性能
        if len(self.access_records) % 100 == 0 or (current_time - self.last_cleanup_time > 10):
            self._clean_old_records(current_time)
            self.last_cleanup_time = current_time
    
    def _clean_old_records(self, current_time: float) -> None:
        """
        清理时间窗口外的旧记录
        
        Args:
            current_time: 当前时间戳
        """
        cutoff_time = current_time - self.time_window
        # 过滤掉过期的记录
        self.access_records = [(ts, lora_id) for ts, lora_id in self.access_records 
                               if ts >= cutoff_time]
    def _clean_old_records(self, current_time: float) -> None:
        """
        清理时间窗口外的旧记录
        通过找到第一个在时间窗口内的记录的索引，然后直接截断列表
        
        Args:
            current_time: 当前时间戳
        """
        cutoff_time = current_time - self.time_window
        
        # 从后向前查找第一个过期的记录
        # 因为记录是按时间顺序添加的，所以一旦找到一个过期记录，其前面的都过期
        for i in range(len(self.access_records) - 1, -1, -1):
            if self.access_records[i][0] < cutoff_time:
                # 找到第一个过期记录，截断列表，只保留i+1及之后的记录
                self.access_records = self.access_records[i + 1:]
                return
        
        # 如果没有过期记录，则不需要操作

    def calculate_low_lora(self) -> float:
        """
        计算Low_lora值，即系统当前所需的LoRA数量估计
        从后往前查找时间窗口内的数据，并清理过期数据
        
        Returns:
            Low_lora值
        """
        current_time = time.time()
        cutoff_time = current_time - self.time_window
        
        # 从后向前找到第一个过期的记录索引
        valid_start_idx = 0
        for i in range(len(self.access_records) - 1, -1, -1):
            if self.access_records[i][0] < cutoff_time:
                valid_start_idx = i + 1
                break
        
        # 只处理有效的记录
        valid_records = self.access_records[valid_start_idx:]
        
        # 清理过期记录
        if valid_start_idx > 0:
            self.access_records = valid_records
            self.last_cleanup_time = current_time
        
        # 如果没有有效记录，直接返回0
        bs = len(valid_records)
        if bs == 0:
            return 0.0
        
        # 计算每个LoRA的访问次数
        lora_counts = defaultdict(int)
        for _, lora_id in valid_records:
            lora_counts[lora_id] += 1
        
        # 计算访问概率
        prob_cache = {lora_id: count / bs for lora_id, count in lora_counts.items()}
        
        # 计算公式 Low_lora = Σ (1 - (1 - prob_i)^BS)
        low_lora = 0.0
        for lora_id, prob in prob_cache.items():
            # 计算在当前批次中至少有一个请求使用该LoRA的概率
            fe_i = 1.0 - (1.0 - prob) ** bs
            low_lora += fe_i
        
        return low_lora

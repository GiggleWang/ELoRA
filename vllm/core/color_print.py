from typing import Dict
from vllm.block import BlockTable, PhysicalTokenBlock


class ColorPrint:
    # ANSI 转义序列颜色代码
    COLORS = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'reset': '\033[0m',  # 重置所有属性

        # 明亮色版本
        'bright_black': '\033[90m',
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
    }

    # 背景色
    BG_COLORS = {
        'bg_black': '\033[40m',
        'bg_red': '\033[41m',
        'bg_green': '\033[42m',
        'bg_yellow': '\033[43m',
        'bg_blue': '\033[44m',
        'bg_magenta': '\033[45m',
        'bg_cyan': '\033[46m',
        'bg_white': '\033[47m',
    }

    # 样式
    STYLES = {
        'bold': '\033[1m',
        'dim': '\033[2m',
        'italic': '\033[3m',
        'underline': '\033[4m',
        'blink': '\033[5m',
        'reverse': '\033[7m',
    }

    @classmethod
    def print(cls, text, color=None, bg_color=None, style=None, end='\n', type="normal"):
        """
        支持直接使用f-string的彩色打印
        """
        format_str = ''
        if style and style in cls.STYLES:
            format_str += cls.STYLES[style]
        if color and color in cls.COLORS:
            format_str += cls.COLORS[color]
        if bg_color and bg_color in cls.BG_COLORS:
            format_str += cls.BG_COLORS[bg_color]

        print(f"{format_str}{text}{cls.COLORS['reset']}", end=end)


def print_block_table(block_table: BlockTable) -> None:
    """
    打印 BlockTable 的完整内容，包含所有字段信息

    Args:
        block_table: List[PhysicalTokenBlock] 类型的区块表
    """
    if not block_table:
        ColorPrint.print("Empty BlockTable", color="yellow")
        return

    # 打印表头
    ColorPrint.print("\n=== Block Table Status ===",
                     color="cyan", style="bold")
    ColorPrint.print(
        f"{'Index':^6} | {'Device':^8} | {'Block#':^7} | {'BlockSize':^9} | "
        f"{'BlockHash':^10} | {'#Tokens':^7} | {'RefCount':^8} | "
        f"{'LastAccess':^10} | {'Computed':^8}",
        color="bright_blue",
        style="bold"
    )
    ColorPrint.print("-" * 90, color="bright_blue")

    # 打印每个块的信息
    for idx, block in enumerate(block_table):
        if block is None:
            ColorPrint.print(
                f"{idx:^6} | {'None':^70}",
                color="bright_black"
            )
            continue

        # 选择颜色：已计算的块使用绿色，未计算的块使用黄色，引用计数为0的块使用暗色
        color = "bright_green" if block.computed else "yellow"
        if block.ref_count == 0:
            color = "bright_black"

        ColorPrint.print(
            f"{idx:^6} | {block.device:^8} | {block.block_number:^7} | "
            f"{block.block_size:^9} | {block.block_hash:^10} | "
            f"{block.num_hashed_tokens:^7} | {block.ref_count:^8} | "
            f"{block.last_accessed:^10} | {block.computed!s:^8}",
            color=color
        )

    # 打印统计信息
    total_blocks = len([b for b in block_table if b is not None])
    computed_blocks = len(
        [b for b in block_table if b is not None and b.computed])
    active_blocks = len(
        [b for b in block_table if b is not None and b.ref_count > 0])

    ColorPrint.print("\n=== Statistics ===", color="cyan", style="bold")
    ColorPrint.print(f"Total Blocks: {total_blocks}", color="bright_white")
    ColorPrint.print(
        f"Computed Blocks: {computed_blocks}", color="bright_green")
    ColorPrint.print(f"Active Blocks: {active_blocks}", color="bright_yellow")


def print_free_pool(free_queue, color="green"):
    blocks = list(free_queue)
    # ColorPrint.print(f"Pool size: {len(blocks)}, Blocks: {blocks}",color=color)


def print_cached_blocks(dict:Dict[int, PhysicalTokenBlock]):
    _table:BlockTable = []
    for _,block in dict.items():
        _table.append(block)
    
    # print_block_table(_table)
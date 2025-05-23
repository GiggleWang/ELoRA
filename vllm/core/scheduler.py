import enum
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple, Union

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.color_print import ColorPrint
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.core.policy import PolicyFactory
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceStatus)

logger = init_logger(__name__)


class PreemptionMode(enum.Enum):
    """Preemption modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    """
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()


# seq_group: SequenceGroup to schedule.
# token_chunk_size: The number of prefill tokens to be processed in the next
# step.
@dataclass
class ScheduledSequenceGroup:
    # A sequence group that's scheduled.
    seq_group: SequenceGroup
    # The total chunk size (number of tokens) to process for next iteration.
    # 1 for decoding. Same as prompt tokens for prefill, but if prefill is
    # chunked, it can be smaller than that.
    token_chunk_size: int


class SchedulerOutputs:

    def __init__(
        self,
        scheduled_seq_groups: Iterable[ScheduledSequenceGroup],
        prompt_run: bool,
        num_batched_tokens: int,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        ignored_seq_groups: List[SequenceGroup],
        swap_id: str = ""
    ) -> None:
        """A list of sequence groups to be scheduled as a single batch.

        Args:
            scheduled_seq_groups: A tuple of scheduled sequence group and its
                token chunk size.
            prompt_run: True if all sequence groups are in prefill phase.
                If False, all sequence groups are in decoding phase.
            num_batched_tokens: Total number of batched tokens.
            blocks_to_swap_in: Blocks to swap in. Dict of CPU -> GPU block
                number.
            blocks_to_swap_out: Blocks to swap out. Dict of GPU -> CPU block
                number.
            blocks_to_copy: Blocks to copy. Source to a list of dest blocks.
            ignored_seq_groups: Sequence groups that are going to be ignored.
        """
        # A tuple of scheduled sequence group and its chunk size.
        self.scheduled_seq_groups: ScheduledSequenceGroup = scheduled_seq_groups
        # True if all sequence groups are in prefill phase. If False, all
        # sequence groups are in decoding phase.
        self.prompt_run: bool = prompt_run
        # Total number of batched tokens.
        self.num_batched_tokens: int = num_batched_tokens
        # Blocks to swap in. Dict of CPU -> GPU block number.
        self.blocks_to_swap_in: Dict[int, int] = blocks_to_swap_in
        # Blocks to swap out. Dict of GPU -> CPU block number.
        self.blocks_to_swap_out: Dict[int, int] = blocks_to_swap_out
        # Blocks to copy. Source to a list of dest blocks.
        self.blocks_to_copy: Dict[int, List[int]] = blocks_to_copy
        # Sequence groups that are going to be ignored.
        self.ignored_seq_groups: List[SequenceGroup] = ignored_seq_groups
        self.swap_id = swap_id
        # Swap in and swap out should never happen at the same time.
        assert not (blocks_to_swap_in and blocks_to_swap_out)

        self.num_loras: int = len(self.lora_requests)
        if self.num_loras > 0:
            self._sort_by_lora_ids()

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (not self.scheduled_seq_groups and not self.blocks_to_swap_in
                and not self.blocks_to_swap_out and not self.blocks_to_copy)

    def _sort_by_lora_ids(self) -> bool:
        self.scheduled_seq_groups = sorted(
            self.scheduled_seq_groups,
            key=lambda g: (g.seq_group.lora_request.elora_id, g.seq_group.request_id))

    @property
    def lora_requests(self) -> Set[LoRARequest]:
        return {g.seq_group.lora_request for g in self.scheduled_seq_groups}


class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        # Note for LoRA scheduling: the current policy is extremely
        # simple and NOT fair. It can lead to starvation of some
        # LoRAs. This should be improved in the future.
        self.lora_config = lora_config

        self.prompt_limit = min(self.scheduler_config.max_model_len,
                                self.scheduler_config.max_num_batched_tokens)

        # Instantiate the scheduling policy.
        self.policy = PolicyFactory.get_policy(policy_name="fcfs")

        BlockSpaceManagerImpl = BlockSpaceManager.get_block_space_manager_class(
            version="v2" if self.scheduler_config.
            use_v2_block_manager else "v1")

        # Create the block space manager.
        self.block_manager = BlockSpaceManagerImpl(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=self.cache_config.enable_prefix_caching)

        # Sequence groups in the WAITING state.
        self.waiting: Deque[SequenceGroup] = deque()
        # Sequence groups in the RUNNING state.
        self.running: Deque[SequenceGroup] = deque()
        # Sequence groups in the SWAPPED state.
        self.swapped: Deque[SequenceGroup] = deque()
        # Sequence groups in the PENDING state.
        self.pending: List[SequenceGroup] = []
        # Time at previous scheduling step
        self.prev_time = 0.0
        # Did we schedule a prompt at previous step?
        self.prev_prompt = False
        # Latency of the last prompt step
        self.last_prompt_latency = 0.0

    def deal_finished_events(self, in_finished: List[str], out_finished: List[str]):
        seq_list = self.block_manager.deal_finished_events(
            in_finished, out_finished)
        for seq_id in seq_list:
            found_seq_group = next(
                (sg for sg in self.pending
                 if sg.seqs_dict and any(seq.seq_id == seq_id for seq in sg.seqs_dict.values())),
                None
            )
            self.pending.remove(found_seq_group)
            self.waiting.appendleft(found_seq_group)
            for seq in found_seq_group.seqs_dict.values():
                seq.status = SequenceStatus.WAITING 


    @property
    def lora_enabled(self) -> bool:
        return bool(self.lora_config)

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts a sequence group with the given ID.

        Check if the sequence group with the given ID
            is present in any of the state queue.
        If present, remove the sequence group from the state queue.
            Also, if any of the sequences in the sequence group is not finished,
                free the sequence with status `FINISHED_ABORTED`.
        Otherwise, do nothing.

        Args:
            request_id: The ID(s) of the sequence group to abort.
        """
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)
        for state_queue in [self.waiting, self.running, self.swapped]:
            aborted_groups: List[SequenceGroup] = []
            for seq_group in state_queue:
                if not request_ids:
                    # Using 'break' here may add two extra iterations,
                    # but is acceptable to reduce complexity .
                    break
                if seq_group.request_id in request_ids:
                    # Appending aborted group into pending list.
                    aborted_groups.append(seq_group)
                    request_ids.remove(seq_group.request_id)
            for aborted_group in aborted_groups:
                # Remove the sequence group from the state queue.
                state_queue.remove(aborted_group)
                for seq in aborted_group.get_seqs():
                    if seq.is_finished():
                        continue
                    seq.status = SequenceStatus.FINISHED_ABORTED
                    self.free_seq(seq)

    def has_unfinished_seqs(self) -> bool:
        return self.waiting or self.running or self.swapped or self.pending

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    def _schedule(self) -> SchedulerOutputs:
        if self.block_manager.enable_caching:
            check_full_res = self.block_manager.check_full()
            if check_full_res is not None:
                # bool = true: from cpu -> gpu
                # bool = false:from gpu -> cpu
                flag, key_str, xpu_xpu_dic = check_full_res
                if flag:
                    scheduler_outputs = SchedulerOutputs(
                        scheduled_seq_groups=[],
                        prompt_run=True,
                        num_batched_tokens=0,
                        blocks_to_swap_in=xpu_xpu_dic,
                        blocks_to_swap_out={},
                        blocks_to_copy={},
                        ignored_seq_groups={},
                        swap_id=key_str
                    )
                    return scheduler_outputs
                else:
                    scheduler_outputs = SchedulerOutputs(
                        scheduled_seq_groups=[],
                        prompt_run=True,
                        num_batched_tokens=0,
                        blocks_to_swap_in={},
                        blocks_to_swap_out=xpu_xpu_dic,
                        blocks_to_copy={},
                        ignored_seq_groups={},
                        swap_id=key_str
                    )
                    return scheduler_outputs

        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        # Fix the current time.
        now = time.time()
        swap_id: str = ""

        # Join waiting sequences if possible.
        if not self.swapped:
            ignored_seq_groups: List[SequenceGroup] = []
            scheduled: List[SequenceGroup] = []
            # The total number of sequences on the fly, including the
            # requests in the generation phase.
            num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                                for seq_group in self.running)
            curr_loras = set(
                seq_group.lora_int_id
                for seq_group in self.running) if self.lora_enabled else None

            # Optimization: We do not sort the waiting queue since the preempted
            # sequence groups are added to the front and the new sequence groups
            # are added to the back.
            leftover_waiting_sequences = deque()
            num_batched_tokens = 0
            while self._passed_delay(now) and self.waiting:
                seq_group = self.waiting[0]
                waiting_seqs = seq_group.get_seqs(
                    status=SequenceStatus.WAITING)
                assert len(waiting_seqs) == 1, (
                    "Waiting sequence group should have only one prompt "
                    "sequence.")
                # get_len includes output tokens if the request has been
                # preempted.
                num_prefill_tokens = waiting_seqs[0].get_len()
                if num_prefill_tokens > self.prompt_limit:
                    logger.warning(
                        f"Input prompt ({num_prefill_tokens} tokens) is too "
                        f"long and exceeds limit of {self.prompt_limit}")
                    for seq in waiting_seqs:
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.popleft()
                    continue

                # If the sequence group cannot be allocated, stop.
                # 相关:查找是否可以allocate
                can_allocate = self.block_manager.can_allocate(seq_group)
                if can_allocate == AllocStatus.LATER:
                    break
                elif can_allocate == AllocStatus.NEVER:
                    logger.warning(
                        f"Input prompt ({num_prefill_tokens} tokens) is too "
                        f"long and exceeds the capacity of block_manager")
                    for seq in waiting_seqs:
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.popleft()
                    continue
                # 说明是OK
                # lora_int_id = 0
                # if self.lora_enabled:
                #     lora_int_id = seq_group.lora_int_id
                #     if (lora_int_id > 0 and lora_int_id not in curr_loras
                #             and len(curr_loras) >= self.lora_config.max_loras):
                #         # We don't have a space for another LoRA, so
                #         # we ignore this request for now.
                #         leftover_waiting_sequences.appendleft(seq_group)
                #         self.waiting.popleft()
                #         continue

                # If the number of batched tokens exceeds the limit, stop.
                # 累计当前批次的 token 数量。如果超出调度器的 max_num_batched_tokens 限制，则停止调度。
                num_batched_tokens += num_prefill_tokens
                if (num_batched_tokens >
                        self.scheduler_config.max_num_batched_tokens):
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                # 当前运行的序列数量。如果加上新序列组的数量后超过 max_num_seqs 限制，则停止调度。
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    break

                # if lora_int_id > 0:
                #     curr_loras.add(lora_int_id)
                # 分配资源并将序列组移入RUNNING队列:
                self.waiting.popleft()
                # 相关:_allocate seq_group
                allocate_res = self._allocate(seq_group)
                if allocate_res is not None:
                    # 说明需要有block进行swap in
                    swap_id, swap_in_dict = allocate_res
                    blocks_to_swap_in = swap_in_dict
                    self.pending.append(seq_group)
                    break
                else:
                    self.running.append(seq_group)
                    num_curr_seqs += num_new_seqs
                    scheduled.append(
                        ScheduledSequenceGroup(
                            seq_group=seq_group,
                            token_chunk_size=num_prefill_tokens))
            self.waiting.extendleft(leftover_waiting_sequences)
            if scheduled or ignored_seq_groups or swap_id != "":
                self.prev_prompt = True
                scheduler_outputs = SchedulerOutputs(
                    scheduled_seq_groups=scheduled,
                    prompt_run=True,
                    num_batched_tokens=num_batched_tokens,
                    blocks_to_swap_in=blocks_to_swap_in,
                    blocks_to_swap_out=blocks_to_swap_out,
                    blocks_to_copy=blocks_to_copy,
                    ignored_seq_groups=ignored_seq_groups,
                    swap_id=swap_id
                )
                return scheduler_outputs

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        self.running = self.policy.sort_by_priority(now, self.running)

        # Reserve new token slots for the running sequence groups.
        running: Deque[SequenceGroup] = deque()
        preempted: List[SequenceGroup] = []
        while self.running:
            seq_group = self.running.popleft()
            # 相关:can_append_slot
            while not self.block_manager.can_append_slot(seq_group):
                if self.running:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = self.running.pop()
                    self._preempt(victim_seq_group, blocks_to_swap_out)
                    preempted.append(victim_seq_group)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq_group, blocks_to_swap_out)
                    preempted.append(seq_group)
                    break
            else:
                # Append new slots to the sequence group.
                self._append_slot(seq_group, blocks_to_copy)
                running.append(seq_group)
        self.running = running

        # Swap in the sequence groups in the SWAPPED state if possible.
        self.swapped = self.policy.sort_by_priority(now, self.swapped)
        if not preempted:
            num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                                for seq_group in self.running)
            # curr_loras = set(
            #     seq_group.lora_request.elora_id
            #     for seq_group in self.running) if self.lora_enabled else None

            leftover_swapped = deque()

            while self.swapped:
                seq_group = self.swapped[0]
                # lora_int_id = 0
                # if self.lora_enabled:
                #     lora_int_id = seq_group.lora_int_id
                #     if (lora_int_id > 0 and lora_int_id not in curr_loras
                #             and len(curr_loras) >= self.lora_config.max_loras):
                #         # We don't have a space for another LoRA, so
                #         # we ignore this request for now.
                #         leftover_swapped.appendleft(seq_group)
                #         self.swapped.popleft()
                #         continue

                # If the sequence group cannot be swapped in, stop.
                if not self.block_manager.can_swap_in(seq_group):
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    break

                # if lora_int_id > 0:
                #     curr_loras.add(lora_int_id)
                self.swapped.popleft()
                self._swap_in(seq_group, blocks_to_swap_in)
                self._append_slot(seq_group, blocks_to_copy)
                num_curr_seqs += num_new_seqs
                self.running.append(seq_group)

            self.swapped.extendleft(leftover_swapped)

        # Each sequence in the generation phase only takes one token slot.
        # Therefore, the number of batched tokens is equal to the number of
        # sequences in the RUNNING state.
        num_batched_tokens = sum(
            seq_group.num_seqs(status=SequenceStatus.RUNNING)
            for seq_group in self.running)

        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=[
                ScheduledSequenceGroup(seq_group=running_group,
                                       token_chunk_size=1)
                for running_group in self.running
            ],
            prompt_run=False,
            num_batched_tokens=num_batched_tokens,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=[],
        )
        return scheduler_outputs

    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        scheduler_outputs = self._schedule()
        now = time.time()

        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
            seq_group = scheduled_seq_group.seq_group
            token_chunk_size = scheduled_seq_group.token_chunk_size
            seq_group.maybe_set_first_scheduled_time(now)

            # seq_id -> SequenceData
            seq_data: Dict[int, SequenceData] = {}
            # seq_id -> physical block numbers
            block_tables: Dict[int, List[int]] = {}

            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)
                self.block_manager.access_all_blocks_in_seq(seq, now)

            common_computed_block_nums = (
                self.block_manager.get_common_computed_block_ids(
                    seq_group.get_seqs(status=SequenceStatus.RUNNING)))

            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=scheduler_outputs.prompt_run,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=block_tables,
                token_chunk_size=token_chunk_size,
                lora_request=seq_group.lora_request,
                computed_block_nums=common_computed_block_nums,
                state=seq_group.state,
                # `multi_modal_data` will only be present for the 1st comm
                # between engine and worker.
                # the subsequent comms can still use delta, but
                # `multi_modal_data` will be None.
                multi_modal_data=seq_group.multi_modal_data
                if scheduler_outputs.prompt_run else None,
            )
            seq_group_metadata_list.append(seq_group_metadata)

        # Now that the batch has been created, we can assume all blocks in the
        # batch will have been computed before the next scheduling invocation.
        # This is because the engine assumes that a failure in model execution
        # will crash the vLLM instance / will not retry.
        for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
            self.block_manager.mark_blocks_as_computed(
                scheduled_seq_group.seq_group)

        return seq_group_metadata_list, scheduler_outputs

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.block_manager.fork(parent_seq, child_seq)

    def free_seq(self, seq: Sequence) -> None:
        """Free a sequence from a block table."""
        self.block_manager.free(seq)

    def free_finished_seq_groups(self) -> None:
        self.running = deque(seq_group for seq_group in self.running
                             if not seq_group.is_finished())

    def _allocate(self, seq_group: SequenceGroup) -> Optional[Tuple[str, Dict[int, int]]]:
        allocate_res = self.block_manager.allocate(seq_group)
        if allocate_res is None:
            for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
                seq.status = SequenceStatus.RUNNING
        else:
            for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
                seq.status = SequenceStatus.PENDING
            return allocate_res

    def _append_slot(
        self,
        seq_group: SequenceGroup,
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            # 相关:append_slot
            ret = self.block_manager.append_slot(seq)
            if ret is not None:
                src_block, dst_block = ret
                if src_block in blocks_to_copy:
                    blocks_to_copy[src_block].append(dst_block)
                else:
                    blocks_to_copy[src_block] = [dst_block]

    def _preempt(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
        preemption_mode: Optional[PreemptionMode] = None,
    ) -> None:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not currently supported. In
        # such a case, we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        if preemption_mode is None:
            if seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP
        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            raise AssertionError("Invalid preemption mode.")

    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.free_seq(seq)
            seq.reset_state_for_recompute()
        # NOTE: For FCFS, we insert the preempted sequence group to the front
        # of the waiting queue.
        self.waiting.appendleft(seq_group)

    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        self._swap_out(seq_group, blocks_to_swap_out)
        self.swapped.append(seq_group)

    def _swap_in(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_in: Dict[int, int],
    ) -> None:
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.RUNNING

    def _swap_out(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED

    def _passed_delay(self, now: float) -> bool:
        if self.prev_prompt:
            self.last_prompt_latency = now - self.prev_time
        self.prev_time, self.prev_prompt = now, False
        # Delay scheduling prompts to let waiting queue fill up
        if self.scheduler_config.delay_factor > 0 and self.waiting:
            earliest_arrival_time = min(
                [e.metrics.arrival_time for e in self.waiting])
            passed_delay = (
                (now - earliest_arrival_time) >
                (self.scheduler_config.delay_factor * self.last_prompt_latency)
                or not self.running)
        else:
            passed_delay = True
        return passed_delay
    def do_clean(self):
        self.block_manager.do_clean()

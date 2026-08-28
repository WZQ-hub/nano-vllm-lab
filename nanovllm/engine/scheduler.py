from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.num_spec_tokens = config.num_spec_tokens if config.speculative_model else 0

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining) # 完成prefill和chunked prefill的处理
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq, num_tokens=self.num_spec_tokens):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = self.num_spec_tokens + 1
                seq.is_prefill = False
                self.block_manager.may_append(seq, num_tokens=self.num_spec_tokens)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    # def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
    #     for seq, token_id in zip(seqs, token_ids):
    #         self.block_manager.hash_blocks(seq)
    #         seq.num_cached_tokens += seq.num_scheduled_tokens
    #         seq.num_scheduled_tokens = 0
    #         if is_prefill and seq.num_cached_tokens < seq.num_tokens:
    #             continue
    #         seq.append_token(token_id)
    #         if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
    #             seq.status = SequenceStatus.FINISHED
    #             self.block_manager.deallocate(seq)
    #             self.running.remove(seq)

    def _truncate(self, seq: Sequence, tokens: list[int]) -> list[int]:
        room = seq.max_tokens - seq.num_completion_tokens
        assert room > 0
        tokens = tokens[:room]
        if not seq.ignore_eos and self.eos in tokens:
            tokens = tokens[:tokens.index(self.eos) + 1]
        return tokens


    def postprocess(self, seqs: list[Sequence], token_ids: list[list[int]], is_prefill: bool): # after run
        k = self.num_spec_tokens
        for seq, tokens in zip(seqs, token_ids):
            old_cached = seq.num_cached_tokens
            if is_prefill:
                self.block_manager.hash_blocks(seq, old_cached, old_cached + seq.num_scheduled_tokens)
                seq.num_cached_tokens += seq.num_scheduled_tokens
                seq.num_scheduled_tokens = 0
                if seq.num_cached_tokens < seq.num_tokens:
                    continue
                seq.append_token(tokens[0])
            else:
                L = len(seq)
                tokens = self._truncate(seq, tokens)
                seq.num_cached_tokens += len(tokens)
                seq.draft_cached_tokens = min(L + k - 1, seq.num_cached_tokens)
                seq.draft_token_ids.clear()
                seq.num_scheduled_tokens = 0
                for token_id in tokens:
                    seq.append_token(token_id)
                self.block_manager.hash_blocks(seq, old_cached, seq.num_cached_tokens)
            if (not seq.ignore_eos and tokens[-1] == self.eos) or seq.num_completion_tokens >= seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

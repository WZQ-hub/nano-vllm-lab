from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class SpeculativeDecoder:
    """
    Speculative Decoding implementation using nano-vllm's ModelRunner.

    Uses a small draft model to generate K tokens speculatively, then verifies
    them with the target model in parallel.
    """

    def __init__(
        self,
        target_model_runner: ModelRunner,
        draft_model_path: str,
        tokenizer,  # transformers.AutoTokenizer
        num_speculative_tokens: int = 5,
    ):
        """
        Args:
            target_model_runner: The main (large) model's ModelRunner
            draft_model_path: Path to the draft (small) model
            tokenizer: Shared tokenizer for both models
            num_speculative_tokens: Number of tokens to generate speculatively (K)
        """
        self.target_runner = target_model_runner
        self.tokenizer = tokenizer
        self.num_speculative_tokens = num_speculative_tokens

        # Initialize draft model with its own ModelRunner
        draft_config = Config(
            model=draft_model_path,
            max_num_batched_tokens=target_model_runner.config.max_num_batched_tokens,
            max_num_seqs=1,  # Speculative decoding works with single sequence
            max_model_len=target_model_runner.config.max_model_len,
            gpu_memory_utilization=0.3,  # Draft model uses less memory
            tensor_parallel_size=1,
            enforce_eager=True,  # Simpler execution for draft
        )
        self.draft_runner = ModelRunner(draft_config, rank=0, event=[])

        self.eos_token_id = tokenizer.eos_token_id

    def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> dict:
        """
        Generate text using speculative decoding.

        Args:
            prompt: Input text prompt
            sampling_params: Sampling parameters (temperature, max_tokens)

        Returns:
            Dictionary with 'text' and 'token_ids' keys
        """
        # Tokenize prompt
        prompt_token_ids = self.tokenizer.encode(prompt)

        # Create sequences for draft and target
        draft_seq = Sequence(prompt_token_ids, sampling_params)
        target_seq = Sequence(prompt_token_ids, sampling_params)

        # Allocate blocks for both sequences
        self._allocate_blocks(draft_seq, self.draft_runner)
        self._allocate_blocks(target_seq, self.target_runner)

        # Prefill stage: process the prompt
        self._prefill(draft_seq, self.draft_runner)
        self._prefill(target_seq, self.target_runner)

        num_generated = 0
        max_tokens = sampling_params.max_tokens

        # Main speculative decoding loop
        while num_generated < max_tokens:
            # Step 1: Generate K draft tokens
            draft_tokens = self._draft_generate(draft_seq, self.num_speculative_tokens)

            if len(draft_tokens) == 0:
                break

            # Step 2: Verify draft tokens with target model
            num_accepted = self._verify_and_accept(
                target_seq,
                draft_seq,
                draft_tokens
            )

            num_generated += num_accepted

            # Check for EOS
            if target_seq.last_token == self.eos_token_id and not sampling_params.ignore_eos:
                break

            # If no tokens accepted, we need at least one token from target
            if num_accepted == 0:
                # Rollback draft sequence to match target
                self._rollback_draft(draft_seq, target_seq)

        # Decode output
        output_token_ids = target_seq.completion_token_ids
        output_text = self.tokenizer.decode(output_token_ids)

        return {
            "text": output_text,
            "token_ids": output_token_ids
        }

    def _allocate_blocks(self, seq: Sequence, _runner: ModelRunner):
        """Allocate KV cache blocks for a sequence."""
        num_blocks_needed = seq.num_blocks
        # Simple allocation: assign blocks sequentially
        # In production, this should use BlockManager
        seq.block_table = list(range(num_blocks_needed))

    def _prefill(self, seq: Sequence, runner: ModelRunner):
        """Run prefill stage for a sequence."""
        seq.num_scheduled_tokens = seq.num_prompt_tokens
        runner.run([seq], is_prefill=True)
        seq.num_cached_tokens = seq.num_prompt_tokens
        seq.is_prefill = False

    def _draft_generate(self, seq: Sequence, num_tokens: int) -> list[int]:
        """
        Generate num_tokens draft tokens using the draft model.

        Returns:
            List of generated token IDs
        """
        draft_tokens = []

        for _ in range(num_tokens):
            # Ensure we have enough blocks
            if len(seq.block_table) < seq.num_blocks:
                new_blocks_needed = seq.num_blocks - len(seq.block_table)
                next_block = len(seq.block_table)
                seq.block_table.extend(range(next_block, next_block + new_blocks_needed))

            # Run one decode step
            seq.num_scheduled_tokens = 1
            token_ids = self.draft_runner.run([seq], is_prefill=False)

            if token_ids is None:
                break

            token_id = token_ids[0]
            seq.append_token(token_id)
            seq.num_cached_tokens = len(seq)
            draft_tokens.append(token_id)

            # Stop if EOS generated
            if token_id == self.eos_token_id:
                break

        return draft_tokens

    def _verify_and_accept(
        self,
        target_seq: Sequence,
        draft_seq: Sequence,
        draft_tokens: list[int]
    ) -> int:
        """
        Verify draft tokens with target model and accept matching ones.

        Returns:
            Number of accepted tokens
        """
        if len(draft_tokens) == 0:
            return 0

        # Prepare target sequence for verification
        # We need to run target model on draft_tokens to get its predictions
        original_len = len(target_seq)

        # Temporarily add all draft tokens to target sequence
        for token_id in draft_tokens:
            target_seq.append_token(token_id)
            if len(target_seq.block_table) < target_seq.num_blocks:
                target_seq.block_table.append(len(target_seq.block_table))

        # Run target model to get predictions for all positions
        target_seq.num_scheduled_tokens = len(draft_tokens)

        # Get target model predictions
        # Note: We need to collect logits for each position
        target_predictions = []

        # Run decode steps one by one to collect predictions
        temp_seq = Sequence(target_seq.token_ids[:original_len], SamplingParams(temperature=target_seq.temperature))
        temp_seq.block_table = target_seq.block_table[:temp_seq.num_blocks]
        temp_seq.num_cached_tokens = original_len
        temp_seq.is_prefill = False

        for draft_token in draft_tokens:
            if len(temp_seq.block_table) < temp_seq.num_blocks:
                temp_seq.block_table.append(len(temp_seq.block_table))

            temp_seq.num_scheduled_tokens = 1
            token_ids = self.target_runner.run([temp_seq], is_prefill=False)

            if token_ids is None:
                break

            target_pred = token_ids[0]
            target_predictions.append(target_pred)
            temp_seq.append_token(draft_token)  # Use draft token to continue
            temp_seq.num_cached_tokens = len(temp_seq)

        # Compare predictions with draft tokens
        num_accepted = 0
        for draft_token, target_pred in zip(draft_tokens, target_predictions):
            if draft_token == target_pred:
                num_accepted += 1
            else:
                # Mismatch found: accept tokens up to this point
                # and use target's prediction instead
                num_accepted += 1
                # Rollback target sequence
                target_seq.token_ids = target_seq.token_ids[:original_len + num_accepted]
                target_seq.token_ids[-1] = target_pred
                target_seq.last_token = target_pred
                target_seq.num_tokens = len(target_seq.token_ids)
                break
        else:
            # All draft tokens accepted, add one more from target prediction
            if len(target_predictions) == len(draft_tokens):
                # Accept all draft tokens
                target_seq.token_ids = target_seq.token_ids[:original_len + num_accepted]
                target_seq.last_token = target_seq.token_ids[-1]
                target_seq.num_tokens = len(target_seq.token_ids)

        target_seq.num_cached_tokens = len(target_seq)

        return num_accepted

    def _rollback_draft(self, draft_seq: Sequence, target_seq: Sequence):
        """Rollback draft sequence to match target sequence."""
        draft_seq.token_ids = target_seq.token_ids.copy()
        draft_seq.last_token = target_seq.last_token
        draft_seq.num_tokens = target_seq.num_tokens
        draft_seq.num_cached_tokens = target_seq.num_cached_tokens
        draft_seq.block_table = target_seq.block_table.copy()

    def cleanup(self):
        """Cleanup resources."""
        self.draft_runner.exit()

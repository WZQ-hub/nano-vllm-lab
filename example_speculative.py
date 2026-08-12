import os
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.sampling_params import SamplingParams
from nanovllm.speculative_decoding import SpeculativeDecoder


def main():
    # Model paths
    target_model_path = os.path.expanduser("./huggingface/Qwen3-8B/")
    draft_model_path = os.path.expanduser("./huggingface/Qwen3-0.6B/")

    # Load tokenizer (shared between target and draft)
    tokenizer = AutoTokenizer.from_pretrained(target_model_path)

    # Initialize target model runner
    target_config = Config(
        model=target_model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_seqs=1,
        gpu_memory_utilization=0.6,
    )
    target_runner = ModelRunner(target_config, rank=0, event=[])

    # Initialize speculative decoder
    decoder = SpeculativeDecoder(
        target_model_runner=target_runner,
        draft_model_path=draft_model_path,
        tokenizer=tokenizer,
        num_speculative_tokens=5,  # Generate 5 draft tokens per iteration
    )

    # Prepare prompt
    prompt = "introduce yourself"
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=256,
        ignore_eos=False,
    )

    # Generate with speculative decoding
    print(f"Prompt: {prompt!r}")
    print("\nGenerating with speculative decoding...")

    output = decoder.generate(prompt, sampling_params)

    print(f"\nCompletion: {output['text']!r}")
    print(f"\nGenerated {len(output['token_ids'])} tokens")

    # Cleanup
    decoder.cleanup()
    target_runner.exit()


if __name__ == "__main__":
    main()

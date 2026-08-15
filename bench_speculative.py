import os
import time
from random import randint, seed
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams
from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.spec_decoding.speculative_decoding import SpeculativeDecoder


def benchmark_standard(llm, prompts, sampling_params):
    """Benchmark standard generation (no speculative decoding)."""
    print("\n" + "="*60)
    print("Benchmarking STANDARD generation (baseline)")
    print("="*60)

    # Warmup
    llm.generate(["Warmup"], SamplingParams(max_tokens=10), use_tqdm=False)

    # Actual benchmark
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    elapsed_time = time.time() - start_time

    # Calculate stats
    total_tokens = sum(len(output["token_ids"]) for output in outputs)
    throughput = total_tokens / elapsed_time

    print(f"\n📊 Standard Generation Results:")
    print(f"  Total output tokens: {total_tokens}")
    print(f"  Time: {elapsed_time:.2f}s")
    print(f"  Throughput: {throughput:.2f} tok/s")

    return {
        "total_tokens": total_tokens,
        "time": elapsed_time,
        "throughput": throughput,
    }


def benchmark_speculative(target_runner, draft_model_path, tokenizer, prompts, sampling_params_list):
    """Benchmark speculative decoding."""
    print("\n" + "="*60)
    print("Benchmarking SPECULATIVE DECODING")
    print("="*60)

    # Initialize speculative decoder
    decoder = SpeculativeDecoder(
        target_model_runner=target_runner,
        draft_model_path=draft_model_path,
        tokenizer=tokenizer,
        num_speculative_tokens=5,
    )

    # Warmup
    warmup_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    decoder.generate(warmup_prompt, SamplingParams(max_tokens=10))

    # Actual benchmark
    outputs = []
    start_time = time.time()

    for i, (prompt, sp) in enumerate(zip(prompts, sampling_params_list)):
        print(f"Processing {i+1}/{len(prompts)}...", end="\r")
        output = decoder.generate(prompt, sp)
        outputs.append(output)

    elapsed_time = time.time() - start_time

    # Calculate stats
    total_tokens = sum(len(output["token_ids"]) for output in outputs)
    throughput = total_tokens / elapsed_time

    print(f"\n📊 Speculative Decoding Results:")
    print(f"  Total output tokens: {total_tokens}")
    print(f"  Time: {elapsed_time:.2f}s")
    print(f"  Throughput: {throughput:.2f} tok/s")

    # Cleanup
    decoder.cleanup()

    return {
        "total_tokens": total_tokens,
        "time": elapsed_time,
        "throughput": throughput,
    }


def main():
    seed(0)

    # Configuration
    num_prompts = 10  # Use fewer prompts for more realistic comparison
    target_model_path = os.path.expanduser("~/huggingface/Qwen3-8B/")
    draft_model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    print("\n🚀 Speculative Decoding Benchmark")
    print(f"Target Model: {target_model_path}")
    print(f"Draft Model: {draft_model_path}")
    print(f"Number of prompts: {num_prompts}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(target_model_path)

    # Prepare prompts
    test_prompts = [
        "Explain the theory of relativity in simple terms.",
        "Write a short story about a robot learning to paint.",
        "List the top 10 programming languages and their use cases.",
        "What are the key differences between Python and JavaScript?",
        "Describe the process of photosynthesis.",
        "How does a neural network work?",
        "What is the capital of France and its history?",
        "Explain quantum computing to a 10-year-old.",
        "What are the benefits of meditation?",
        "Describe the water cycle in nature.",
    ]

    prompts = test_prompts[:num_prompts]

    # Apply chat template
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    # Sampling parameters
    sampling_params_list = [
        SamplingParams(
            temperature=0.6,
            max_tokens=randint(100, 256),
            ignore_eos=False,
        )
        for _ in range(num_prompts)
    ]

    # Single sampling params for standard benchmark
    sampling_params_uniform = SamplingParams(
        temperature=0.6,
        max_tokens=200,
        ignore_eos=False,
    )

    # ========================================
    # Benchmark 1: Standard Generation
    # ========================================
    llm = LLM(
        target_model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_seqs=1,
        gpu_memory_utilization=0.6,
    )

    standard_results = benchmark_standard(llm, prompts, sampling_params_uniform)

    # ========================================
    # Benchmark 2: Speculative Decoding
    # ========================================
    # Get the model runner from LLMEngine
    target_runner = llm.model_runner

    speculative_results = benchmark_speculative(
        target_runner,
        draft_model_path,
        tokenizer,
        prompts,
        sampling_params_list,
    )

    # ========================================
    # Comparison
    # ========================================
    print("\n" + "="*60)
    print("📈 PERFORMANCE COMPARISON")
    print("="*60)

    speedup = standard_results["time"] / speculative_results["time"]
    throughput_improvement = (speculative_results["throughput"] / standard_results["throughput"] - 1) * 100

    print(f"\n{'Method':<25} {'Time (s)':<12} {'Throughput (tok/s)':<20}")
    print("-" * 60)
    print(f"{'Standard Generation':<25} {standard_results['time']:<12.2f} {standard_results['throughput']:<20.2f}")
    print(f"{'Speculative Decoding':<25} {speculative_results['time']:<12.2f} {speculative_results['throughput']:<20.2f}")
    print("-" * 60)
    print(f"\n🎯 Speedup: {speedup:.2f}x")
    print(f"🎯 Throughput improvement: {throughput_improvement:+.1f}%")

    if speedup > 1.5:
        print("\n✅ Speculative decoding is working well!")
    elif speedup > 1.0:
        print("\n⚠️  Moderate speedup. Consider tuning num_speculative_tokens.")
    else:
        print("\n❌ Speculative decoding is slower. Check acceptance rate and model compatibility.")

    # Cleanup
    llm.exit()


if __name__ == "__main__":
    main()

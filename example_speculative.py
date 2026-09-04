import argparse
import os
from time import perf_counter

from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


TARGET_PATH = os.environ.get("TARGET_MODEL", "./huggingface/Qwen3-8B/")
DRAFT_PATH = os.environ.get("DRAFT_MODEL", "./huggingface/Qwen3-0.6B/")

PROMPTS = [
    "introduce yourself",
    "list all prime numbers within 100",
]


def build_prompts(tokenizer, prompts):
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=TARGET_PATH)
    parser.add_argument("--draft", default=DRAFT_PATH)
    parser.add_argument("-k", "--num-spec-tokens", type=int, default=4)
    parser.add_argument("--no-spec", action="store_true",
                        help="不加载 draft 模型，跑纯 target 逐 token 解码作为基线")
    parser.add_argument("--batch", action="store_true",
                        help="一次跑多条 prompt(每步产出的统计会变成批平均)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--enforce-eager", action="store_true", default=True)
    parser.add_argument("--cudagraph", dest="enforce_eager", action="store_false",
                        help="关掉 enforce_eager, 用 cudagraph")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    prompts = build_prompts(tokenizer, PROMPTS if args.batch else PROMPTS[:1])
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    k = 0 if args.no_spec else args.num_spec_tokens
    llm = LLM(
        args.target,
        speculative_model=None if args.no_spec else args.draft,
        num_spec_tokens=k,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=1,
    )

    for prompt in prompts:
        llm.add_request(prompt, sampling_params)

    # 手动驱动 step 循环（而不是 llm.generate），为的是数出 decode 步数
    outputs = {}
    decode_steps = 0
    t = perf_counter()
    while not llm.is_finished():
        finished, num_tokens = llm.step()
        if num_tokens <= 0:            # num_tokens > 0 表示这是一个 prefill 步
            decode_steps += 1
        for seq_id, token_ids in finished:
            outputs[seq_id] = token_ids
    elapsed = perf_counter() - t

    outputs = [outputs[seq_id] for seq_id in sorted(outputs)]

    for prompt, token_ids in zip(prompts, outputs):
        print("\n" + "=" * 60)
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {tokenizer.decode(token_ids)!r}")

    total_tokens = sum(len(token_ids) for token_ids in outputs)
    per_step = total_tokens / decode_steps if decode_steps else 0.0

    print("\n" + "=" * 60)
    print(f"mode            : {'baseline (no spec)' if args.no_spec else f'speculative, k={k}'}")
    print(f"generated tokens: {total_tokens}")
    print(f"decode steps    : {decode_steps}")
    print(f"tokens / step   : {per_step:.2f}")
    if not args.no_spec and decode_steps:
        print(f"accept rate     : {max(per_step - 1, 0) / k:.1%}   (平均接受草稿数 / k)")
    print(f"elapsed         : {elapsed:.2f}s   ({total_tokens / elapsed:.1f} tok/s)")


if __name__ == "__main__":
    main()

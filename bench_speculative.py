import argparse
import os
import traceback
from time import perf_counter

import torch.multiprocessing as mp


TARGET_PATH = os.environ.get("TARGET_MODEL", "./huggingface/Qwen3-8B/")
DRAFT_PATH = os.environ.get("DRAFT_MODEL", "./huggingface/Qwen3-0.6B/")

# 用真实文本，不用随机 token id：接受率取决于草稿模型能不能预测目标模型的输出，
# 喂随机 token 会把两个模型都推到分布外，测出来的接受率没有参考价值。
PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a short story about a robot learning to paint.",
    "List the top 10 programming languages and their use cases.",
    "What are the key differences between Python and JavaScript?",
    "Describe the process of photosynthesis.",
    "How does a neural network work?",
    "Explain quantum computing to a 10-year-old.",
    "Describe the water cycle in nature.",
]


def _worker(queue, args, k):
    """在子进程里跑一个配置。进程退出时显存自动释放。"""
    try:
        from transformers import AutoTokenizer
        from nanovllm import LLM, SamplingParams

        tokenizer = AutoTokenizer.from_pretrained(args.target)
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for i in range(args.num_seqs)
        ]
        # ignore_eos + 固定 max_tokens：让每条序列产出的 token 数确定，吞吐可比
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            ignore_eos=True,
        )

        llm = LLM(
            args.target,
            speculative_model=None if k == 0 else args.draft,
            num_spec_tokens=k,
            enforce_eager=args.enforce_eager,
            tensor_parallel_size=1,
            max_num_seqs=args.num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

        # warmup：触发 dynamo 编译 + cudagraph replay，否则首轮编译开销会算进计时
        llm.generate(["Benchmark: "], SamplingParams(max_tokens=16, ignore_eos=True), use_tqdm=False)

        for prompt in prompts:
            llm.add_request(prompt, sampling_params)

        outputs = {}
        decode_steps = prefill_steps = prefill_tokens = seq_steps = 0
        t = perf_counter()
        while not llm.is_finished():
            finished, num_tokens = llm.step()
            if num_tokens > 0:                  # >0 表示 prefill 步
                prefill_steps += 1
                prefill_tokens += num_tokens
            else:
                decode_steps += 1
                seq_steps += -num_tokens    # llm_engine 在 decode 分支返回 -len(seqs)
            for seq_id, token_ids in finished:
                outputs[seq_id] = token_ids
        elapsed = perf_counter() - t

        total_tokens = sum(len(ids) for ids in outputs.values())
        queue.put(dict(
            ok=True, k=k,
            total_tokens=total_tokens,
            decode_steps=decode_steps,
            seq_steps=seq_steps,
            prefill_steps=prefill_steps,
            prefill_tokens=prefill_tokens,
            num_blocks=len(llm.scheduler.block_manager.blocks),
            elapsed=elapsed,
        ))
    except Exception:
        queue.put(dict(ok=False, k=k, err=traceback.format_exc()))


def run_one(args, k):
    ctx = mp.get_context("spawn")           # 必须 spawn：fork 会把父进程的 CUDA 状态带坏
    queue = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(queue, args, k))
    proc.start()
    result = queue.get()                    # 先 get 再 join，避免队列缓冲写满时死锁
    proc.join()
    if proc.exitcode != 0 and result.get("ok"):
        result = dict(ok=False, k=k, err=f"子进程异常退出，exitcode={proc.exitcode}")
    return result


def solve_alpha(tokens_per_step, k):
    """从 E[tokens/step] = 1 + a(1-a^k)/(1-a) 反解逐 token 接受率，二分即可。"""
    target = tokens_per_step - 1
    if target <= 0:
        return 0.0
    lo, hi = 0.0, 1.0 - 1e-9
    for _ in range(60):
        mid = (lo + hi) / 2
        expected = mid * (1 - mid ** k) / (1 - mid)
        if expected < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=TARGET_PATH)
    parser.add_argument("--draft", default=DRAFT_PATH)
    parser.add_argument("-k", "--num-spec-tokens", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6],
                        help="要扫的 k 值列表，会自动带上 k=0 作为基线")
    parser.add_argument("--num-seqs", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096,
                        help="warmup 的激活峰值按它算，调小能给 KV cache 腾出预算")
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    configs = [0] + [k for k in args.num_spec_tokens if k > 0]

    print(f"target : {args.target}")
    print(f"draft  : {args.draft}")
    print(f"batch  : {args.num_seqs} seqs x {args.max_tokens} tokens, temperature={args.temperature}")
    print(f"backend: {'eager' if args.enforce_eager else 'cudagraph'}\n")

    results = []
    for k in configs:
        label = "baseline" if k == 0 else f"k={k}"
        print(f"[{label}] running ...", flush=True)
        res = run_one(args, k)
        if not res["ok"]:
            print(f"[{label}] FAILED\n{res['err']}")
            continue
        res["throughput"] = res["total_tokens"] / res["elapsed"]
        res["per_step"] = res["total_tokens"] / res["seq_steps"] if res["seq_steps"] else 0.0
        results.append(res)
        print(f"[{label}] {res['elapsed']:.2f}s  {res['throughput']:.1f} tok/s  "
              f"blocks={res['num_blocks']}  prefill: {res['prefill_steps']} 步 / "
              f"{res['prefill_tokens']} tok\n", flush=True)

    if not results:
        return

    baseline = next((r for r in results if r["k"] == 0), None)

    print("=" * 74)
    print(f"{'config':<10} {'time(s)':>9} {'tok/s':>10} {'speedup':>9} {'tok/seq/step':>13} "
          f"{'alpha':>8} {'blocks':>7} {'reprefill':>10}")
    print("-" * 90)
    for r in results:
        label = "baseline" if r["k"] == 0 else f"k={r['k']}"
        speedup = f"{r['throughput'] / baseline['throughput']:.2f}x" if baseline else "-"
        alpha = f"{solve_alpha(r['per_step'], r['k']):.3f}" if r["k"] else "-"
        # 正常情况下每条序列只 prefill 一次；多出来的都是被 preempt 后重跑的
        reprefill = r["prefill_tokens"] / max(r["total_tokens"], 1)
        print(f"{label:<10} {r['elapsed']:>9.2f} {r['throughput']:>10.1f} {speedup:>9} "
              f"{r['per_step']:>13.2f} {alpha:>8} {r['num_blocks']:>7} {reprefill:>9.2f}x")
    print("=" * 90)
    print("tok/seq/step = 每条序列每个 decode step 产出的 token 数。baseline 恒为 1.00，")
    print("               spec 上限 k+1。这才是判断 spec 有没有生效的指标")
    print("alpha     = 逐 token 接受率，理论上不随 k 变化；若随 k 明显漂移说明记账或索引有问题")
    print("reprefill = prefill token 数 / 产出 token 数。健康值 <0.2；接近或超过 1 说明")
    print("            KV block 不够，序列被反复 preempt 后从头重跑，这才是拖慢的主因")


if __name__ == "__main__":
    main()

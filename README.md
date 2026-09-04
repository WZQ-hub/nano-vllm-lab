English | [简体中文](README.zh-CN.md)

# nano-vllm-lab

A lightweight vLLM implementation built from scratch, extended with speculative decoding.

This is a fork of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm). The upstream engine is kept intact; this repo adds draft-model speculative
decoding and the benchmarks for it.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Compact implementation in Python
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.
* 🧠 **Speculative decoding** - Draft-model assisted decoding with configurable speculation length

## Installation

```bash
git clone https://github.com/WZQ-hub/nano-vllm-lab.git
cd nano-vllm-lab
pip install -e .
```

> Note: this repo is a fork of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).
> Installing the upstream package instead will **not** give you speculative decoding —
> `LLM(..., speculative_model=...)` only exists here.

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ./huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

The paths above match the script defaults (`./huggingface/<model>/`, relative to the
repo root), so the example commands below work as-is. Override them with `--target` /
`--draft`, or with the `TARGET_MODEL` / `DRAFT_MODEL` environment variables.

For speculative decoding, download the target model as well:

```bash
huggingface-cli download --resume-download Qwen/Qwen3-8B \
  --local-dir ./huggingface/Qwen3-8B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Speculative Decoding

Nano-vLLM can use a smaller draft model to propose tokens for a larger target model.
The target model verifies the proposed tokens, allowing multiple tokens to be
accepted in a single decoding step.

```bash
python example_speculative.py \
  --target ./huggingface/Qwen3-8B \
  --draft ./huggingface/Qwen3-0.6B \
  --num-spec-tokens 4
```

To run the target model without speculative decoding as a baseline:

```bash
python example_speculative.py \
  --target ./huggingface/Qwen3-8B \
  --no-spec
```

`--num-spec-tokens` controls how many tokens the draft model proposes at each
decoding step. `--cudagraph` enables CUDA Graph execution; omit it to use the
default eager backend in `example_speculative.py`.

## Benchmark

See `bench.py` for the vLLM comparison benchmark and `bench_speculative.py` for
the speculative decoding benchmark.

### Speculative Decoding Benchmark

Example command:

```bash
python bench_speculative.py \
  --target ./huggingface/Qwen3-8B \
  --draft ./huggingface/Qwen3-0.6B \
  --num-seqs 1 \
  --max-tokens 256
```

The benchmark automatically includes `k=0` as the non-speculative baseline.

**Test Configuration:**
- Platform: AutoDL
- GPU: NVIDIA RTX 4090 (24GB)
- Target model: Qwen3-8B
- Draft model: Qwen3-0.6B
- Backend: cudagraph

**Results from one run (speculative decoding off vs. on):**

All rows are one `bench_speculative.py` sweep on this repo, so `k` is the only
variable. `baseline` is `k=0` (`speculative_model=None`): plain one-token-per-step

| Config | Time (s) | Throughput (tok/s) | Speedup | Tokens / seq / step | Alpha | Blocks | Reprefill |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 4.46 | 57.4 | 1.00x | 1.00 | - | 141 | 0.07x |
| k=1 | 5.83 | 43.9 | 0.76x | 1.71 | 0.707 | 61 | 0.07x |
| k=2 | 4.48 | 57.1 | 0.99x | 2.35 | 0.764 | 61 | 0.07x |
| k=3 | 4.47 | 57.3 | 1.00x | 2.46 | 0.681 | 61 | 0.07x |
| k=4 | 3.99 | 64.1 | 1.12x | 2.91 | 0.725 | 61 | 0.07x |
| k=5 | 3.95 | 64.8 | 1.13x | 3.08 | 0.722 | 61 | 0.07x |
| k=6 | 4.30 | 59.5 | 1.04x | 3.32 | 0.734 | 61 | 0.07x |


The speculative benchmark reports:
- `Time`: total generation time.
- `Throughput`: generated tokens per second.
- `Speedup`: throughput relative to the baseline.
- `Tokens / seq / step`: average generated tokens per decoding step.
- `Alpha`: estimated draft-token acceptance rate.
- `Blocks`: allocated KV-cache blocks.
- `Reprefill`: prefill-token ratio caused by re-prefilling.


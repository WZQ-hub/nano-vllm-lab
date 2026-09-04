[English](README.md) | 简体中文

# nano-vllm-lab

一个从零实现的轻量级 vLLM，并扩展了投机采样（speculative decoding）。

本仓库 fork 自 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)，保留了上游引擎的整体结构，
在此基础上增加了基于草稿模型（draft model）的投机采样及配套 benchmark。

## 主要特性

* 🚀 **快速离线推理** —— 推理速度与 vLLM 相当
* 📖 **可读的代码** —— 精简的 Python 实现
* ⚡ **优化套件** —— 前缀缓存、张量并行、Torch 编译、CUDA Graph 等
* 🧠 **投机采样** —— 草稿模型辅助解码，投机长度可配置

## 安装

```bash
git clone https://github.com/WZQ-hub/nano-vllm-lab.git
cd nano-vllm-lab
pip install -e .
```

> 注意：本仓库是 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 的 fork。
> 直接安装上游包**不会**带有投机采样 —— `LLM(..., speculative_model=...)` 只在本仓库中存在。

## 模型下载

手动下载模型权重：

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ./huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

上面的路径与脚本默认值一致（`./huggingface/<模型名>/`，相对于仓库根目录），因此下文的示例命令可以直接运行。
也可以用 `--target` / `--draft` 参数，或 `TARGET_MODEL` / `DRAFT_MODEL` 环境变量覆盖。

使用投机采样还需要下载目标模型：

```bash
huggingface-cli download --resume-download Qwen/Qwen3-8B \
  --local-dir ./huggingface/Qwen3-8B/ \
  --local-dir-use-symlinks False
```

## 快速开始

用法见 `example.py`。接口与 vLLM 基本一致，`LLM.generate` 的返回值略有差异：

```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## 投机采样

用一个较小的草稿模型为较大的目标模型提前提议若干 token，目标模型再对这些 token 做验证，
从而在一个解码步内接受多个 token。

```bash
python example_speculative.py \
  --target ./huggingface/Qwen3-8B \
  --draft ./huggingface/Qwen3-0.6B \
  --num-spec-tokens 4
```

作为基线，也可以关掉投机采样只跑目标模型：

```bash
python example_speculative.py \
  --target ./huggingface/Qwen3-8B \
  --no-spec
```

`--num-spec-tokens` 控制草稿模型每步提议多少个 token。`--cudagraph` 启用 CUDA Graph；
不加该参数时 `example_speculative.py` 默认走 eager 后端。

## Benchmark

`bench.py` 是与 vLLM 对比的 benchmark，`bench_speculative.py` 是投机采样的 benchmark。

### 投机采样 Benchmark

示例命令：

```bash
python bench_speculative.py \
  --target ./huggingface/Qwen3-8B \
  --draft ./huggingface/Qwen3-0.6B \
  --num-seqs 1 \
  --max-tokens 256
```

脚本会自动加上 `k=0` 作为不开投机采样的基线。

**测试配置：**
- 平台：AutoDL
- GPU：NVIDIA RTX 4090 (24GB)
- 目标模型：Qwen3-8B
- 草稿模型：Qwen3-0.6B
- 后端：cudagraph

**单次运行结果（投机采样关闭 vs. 开启）：**

所有行都来自同一次 `bench_speculative.py` sweep，唯一的变量是 `k`。`baseline` 即 `k=0`
（`speculative_model=None`）的逐 token 解码 —— 跑的是本仓库的代码，而非上游 nano-vllm。

| 配置 | 时间 (s) | 吞吐 (tok/s) | 加速比 | 每序列每步 token 数 | Alpha | Blocks | Reprefill |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 4.46 | 57.4 | 1.00x | 1.00 | - | 141 | 0.07x |
| k=1 | 5.83 | 43.9 | 0.76x | 1.71 | 0.707 | 61 | 0.07x |
| k=2 | 4.48 | 57.1 | 0.99x | 2.35 | 0.764 | 61 | 0.07x |
| k=3 | 4.47 | 57.3 | 1.00x | 2.46 | 0.681 | 61 | 0.07x |
| k=4 | 3.99 | 64.1 | 1.12x | 2.91 | 0.725 | 61 | 0.07x |
| k=5 | 3.95 | 64.8 | 1.13x | 3.08 | 0.722 | 61 | 0.07x |
| k=6 | 4.30 | 59.5 | 1.04x | 3.32 | 0.734 | 61 | 0.07x |

各列含义：
- `时间`：总生成耗时。
- `吞吐`：每秒生成的 token 数。
- `加速比`：相对 baseline 的吞吐比值。
- `每序列每步 token 数`：每个解码步平均产出的 token 数。
- `Alpha`：估算出的草稿 token 接受率。
- `Blocks`：分配到的 KV cache block 数。
- `Reprefill`：因重新 prefill 产生的 prefill token 占比。

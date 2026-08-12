# Speculative Decoding 实现说明

## 已完成的部分

我已经为你创建了基于 nano-vllm 架构的 Speculative Decoding 实现：

### 1. 核心实现 ([nanovllm/speculative_decoding.py](nanovllm/speculative_decoding.py))

**SpeculativeDecoder 类**提供了完整的推测解码功能：

- **Draft 生成**：使用小模型（draft model）快速生成 K 个候选 tokens
- **并行验证**：用大模型（target model）验证这些候选 tokens
- **Accept/Reject**：接受匹配的 tokens，在第一个不匹配处停止并使用 target 的预测

**关键方法**：
- `generate()` - 主入口，执行完整的推测解码流程
- `_draft_generate()` - 用 draft model 生成候选 tokens
- `_verify_and_accept()` - 用 target model 验证并接受 tokens
- `_prefill()` - 处理初始 prompt 的 prefill 阶段

### 2. 使用示例 ([example_speculative.py](example_speculative.py))

演示了如何使用 SpeculativeDecoder。

## 复用的 nano-vllm 组件

✅ **ModelRunner** - 管理模型推理、KV cache、CUDA graph  
✅ **Sampler** - Token 采样（支持 temperature）  
✅ **Sequence** - 序列状态管理  
✅ **SamplingParams** - 采样参数配置  
✅ **Config** - 模型配置

## 还需要补充的部分

### 1. **Block Manager 集成**

当前的 `_allocate_blocks()` 使用了简化的块分配逻辑：

```python
def _allocate_blocks(self, seq: Sequence, _runner: ModelRunner):
    # 简化版：顺序分配块
    seq.block_table = list(range(num_blocks_needed))
```

**TODO**：
- 集成 `nanovllm/engine/block_manager.py` 的 BlockManager
- 处理块的分配、释放和共享
- 避免块冲突（draft 和 target 使用不同的块池）

### 2. **KV Cache 同步**

当前实现中 draft 和 target 各自维护独立的 KV cache。需要考虑：

- **回滚机制**：当 draft tokens 被拒绝时，需要回滚 KV cache
- **Cache 裁剪**：在不匹配时裁剪多余的 cache entries
- **内存优化**：避免重复存储相同的 prefix

**建议**：参考你原始代码的 `prune_cache()` 思路，但需要适配 nano-vllm 的 block-based cache。

### 3. **性能优化**

当前实现的 `_verify_and_accept()` 是逐 token 验证的：

```python
for draft_token in draft_tokens:
    temp_seq.num_scheduled_tokens = 1
    token_ids = self.target_runner.run([temp_seq], is_prefill=False)
```

**优化方向**：
- **批量验证**：一次前向传播验证所有 K 个 tokens
- **Early stopping**：第一个不匹配就停止，不继续验证后续 tokens
- **CUDA graph**：draft model 也使用 CUDA graph 加速

### 4. **统计信息**

建议添加性能统计：

```python
class SpeculativeDecoder:
    def __init__(self, ...):
        self.stats = {
            "total_draft_tokens": 0,
            "accepted_tokens": 0,
            "acceptance_rate": 0.0,
            "num_iterations": 0,
        }
```

用于分析：
- **Acceptance rate**：draft tokens 被接受的比例
- **Speedup**：相比普通解码的加速比
- **Draft efficiency**：draft model 的生成质量

### 5. **错误处理**

需要增加：
- Draft model 加载失败的处理
- 内存不足时的降级策略（退回到普通解码）
- Tokenizer 不匹配的检测
- EOS token 处理的边界情况

### 6. **测试用例**

建议创建 `tests/test_speculative_decoding.py`：

```python
def test_acceptance_logic():
    """测试 accept/reject 逻辑是否正确"""
    pass

def test_eos_handling():
    """测试 EOS token 的处理"""
    pass

def test_cache_consistency():
    """测试 KV cache 的一致性"""
    pass

def test_speedup():
    """对比普通解码，验证加速效果"""
    pass
```

## 使用方法

### 基本使用

```bash
python example_speculative.py
```

### 配置参数

```python
decoder = SpeculativeDecoder(
    target_model_runner=target_runner,
    draft_model_path="./huggingface/Qwen3-0.6B/",  # 小模型路径
    tokenizer=tokenizer,
    num_speculative_tokens=5,  # K 值：每次生成多少个候选 tokens
)
```

**num_speculative_tokens (K) 的选择**：
- K=3~5：平衡性能和准确率（推荐）
- K 太小：加速效果不明显
- K 太大：acceptance rate 降低，反而变慢

### 模型选择

**Draft Model 应该比 Target Model 小 3-10 倍**：
- Target: Qwen3-8B → Draft: Qwen3-0.6B (13x smaller) ✅
- Target: Llama-70B → Draft: Llama-7B (10x smaller) ✅
- Target: Qwen3-8B → Draft: Qwen3-1.8B (4.4x smaller) ⚠️ draft 太慢

## 预期性能

根据 DeepMind 的 Speculative Decoding 论文：

- **理想情况**（high acceptance rate）：2-3x 加速
- **实际情况**：1.5-2x 加速（取决于任务和模型匹配度）
- **最坏情况**（low acceptance rate）：可能比普通解码慢

**影响因素**：
1. Draft 和 Target 模型的相似度
2. 生成任务的难度（创意写作 vs 模板填充）
3. Temperature 设置（低 temp 更容易匹配）

## 与原始代码的对比

### 你的原始实现
```python
def speculative_decoding(prompt, max_new_tokens, gen_config):
    # 使用 transformers 的 .generate() API
    draft_outputs = draft_model.generate(...)
```

**问题**：
- 依赖 transformers 的 generate()，不适配 nano-vllm
- KV cache 结构假设不匹配
- 没有复用 nano-vllm 的优化（CUDA graph、paged attention）

### 新实现
```python
class SpeculativeDecoder:
    def _draft_generate(self, seq, num_tokens):
        # 使用 ModelRunner.run() 逐步生成
        token_ids = self.draft_runner.run([seq], is_prefill=False)
```

**优势**：
- 完全复用 nano-vllm 的基础设施
- 统一的 KV cache 管理
- 可以利用 CUDA graph 等优化

## 下一步建议

1. **运行示例**：`python example_speculative.py` 验证基本功能
2. **修复 Block Manager**：集成正确的块分配逻辑
3. **优化验证阶段**：改为批量验证而不是逐个验证
4. **添加统计**：输出 acceptance rate 等指标
5. **性能测试**：对比 `example.py` 的普通生成

## 参考资料

- [Speculative Decoding 论文](https://arxiv.org/abs/2211.17192)
- nano-vllm 的 [ModelRunner](nanovllm/engine/model_runner.py) 实现
- nano-vllm 的 [Sequence](nanovllm/engine/sequence.py) 状态管理

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
try:
    from llmcompressor.modifiers.gptq import GPTQModifier              # 新版路径
except ImportError:
    from llmcompressor.modifiers.quantization import GPTQModifier      # 旧版路径

MODEL_PATH = "/root/autodl-tmp/nano-vllm-lab/huggingface/Qwen3-8B/"
SAVE_DIR = "/root/autodl-tmp/nano-vllm-lab/huggingface/Qwen3-8B-W4A16-G128/"
NUM_SAMPLES = 512
MAX_LEN = 2048

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 1) 校准数据：取 512 条对话，转成 token
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{NUM_SAMPLES}]").shuffle(seed=42)

def preprocess(ex):
    return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}
ds = ds.map(preprocess)

def tokenize(s):
    return tokenizer(s["text"], padding=False, max_length=MAX_LEN,
                     truncation=True, add_special_tokens=False)
ds = ds.map(tokenize, remove_columns=ds.column_names)

# 2) 配方：GPTQ + W4A16
recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

# 3) 量化（这次要传 dataset）
oneshot(model=model, dataset=ds, recipe=recipe,
        max_seq_length=MAX_LEN, num_calibration_samples=NUM_SAMPLES)

# 4) 只存一次
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
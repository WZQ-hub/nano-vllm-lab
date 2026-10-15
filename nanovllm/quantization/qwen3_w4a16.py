import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


model_path = os.path.expanduser("./huggingface/Qwen3-8B/")
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

recipe = QuantizationModifier(
    target = "linear",
    scheme = "W8A16",
    ignore = "lm_head"
)
oneshot(model=model, recipe=recipe)

SAVE_DIR = "Qwen3-8B-W8A16"
model.save_pretrained(SAVE_DIR, save_compressed=True)          
tokenizer.save_pretrained(SAVE_DIR)
model.save_pretrained(SAVE_DIR + "-ref", save_compressed=False)
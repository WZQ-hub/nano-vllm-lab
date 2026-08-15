import os 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from nanovllm.spec_decoding.speculative_decoding import speculative_round

TARGET_PATH = os.environ.get(
    "TARGET_MODEL",
    os.path.expanduser("./huggingface/Qwen3-8B"),
)

DRAFT_PATH = os.environ.get(
    "DRAFT_MODEL",
    os.path.expanduser("./huggingface/Qwen3-0.6B"),
)

device = "cuda" if torch.cuda.is_available() else "cpu"
type = torch.bfloat16 if device == "cuda" else torch.float32

class HFLogitsModel:
    def __init__(self, model_path):
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=type)
            .to(device)
            .eval()
        )

    @torch.inference_mode()
    def logits(self, input_ids):
        return self.model(input_ids = input_ids, use_cache=False).logits


tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH)
target_model = HFLogitsModel(TARGET_PATH)
draft_model = HFLogitsModel(DRAFT_PATH)

prompt = "Explain speculative decoding in one short paragraph."
prompt_ids = tokenizer(
    prompt,
    return_tensors="pt",
).input_ids.to(device)

generated_ids = prompt_ids
max_rounds = 8
k = 4

for _ in range(max_rounds):
    new_tokens = speculative_round(
        prompt_ids=generated_ids,
        target_model=target_model,
        draft_model=draft_model,
        k=k,
    )

    generated_ids = torch.cat(
        [generated_ids, new_tokens],
        dim=1,
    )

    if tokenizer.eos_token_id in new_tokens[0].tolist():
        break

completion_ids = generated_ids[0, prompt_ids.shape[1]:].tolist()

print("Prompt:", prompt)
print("Completion:", tokenizer.decode(
    completion_ids,
    skip_special_tokens=True,
))
print("Generated token count:", len(completion_ids))
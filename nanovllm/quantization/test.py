import json, torch
from safetensors import safe_open

def load(d, key):
    try:
        fn = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"][key]
    except FileNotFoundError:
        fn = "model.safetensors"
    with safe_open(f"{d}/{fn}", "pt") as f:
        return f.get_tensor(key)

orig = "/root/autodl-tmp/nano-vllm-lab/huggingface/Qwen3-8B"
comp = "Qwen3-8B-W8A16"
name = "model.layers.0.self_attn.q_proj"

w = load(orig, name + ".weight").float()
packed = load(comp, name + ".weight_packed")
scale = load(comp, name + ".weight_scale").float()
print("weight_shape:", load(comp, name + ".weight_shape"))

p = packed.to(torch.int64) & 0xFFFFFFFF
u = torch.stack([(p >> (8 * i)) & 0xFF for i in range(4)], dim=-1).reshape(packed.shape[0], -1)

for tag, q in [("offset128", u - 128),
               ("int8", u.to(torch.uint8).view(torch.int8).to(torch.int64))]:
    err = ((q.float() * scale) - w).abs() / scale
    print(tag, "max err/scale =", err.max().item())
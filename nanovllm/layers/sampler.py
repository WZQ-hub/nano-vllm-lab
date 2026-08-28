import torch
from torch import nn

@torch.compile
def compute_probs(logits: torch.Tensor, temperatures: torch.Tensor):
    logits = logits.float().div_(temperatures.unsqueeze(dim=1))
    return torch.softmax(logits, dim=-1)

@torch.compile
def sample_from_probs(probs: torch.Tensor):
    q = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
    return probs.div(q).argmax(dim=-1)





class Sampler(nn.Module):


    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        return sample_from_probs(compute_probs(logits, temperatures))



    # @torch.compile
    # def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
    #     logits = logits.float().div_(temperatures.unsqueeze(dim=1))
    #     probs = torch.softmax(logits, dim=-1)
    #     sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
    #     return sample_tokens

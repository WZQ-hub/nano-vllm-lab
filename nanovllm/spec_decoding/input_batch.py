from dataclasses import dataclass
import torch





@dataclass
class InputBatch:
    
    # [num_tokens_after_padding]
    input_ids: torch.Tensor

    # [num_tokens_after_padding]
    positions: torch.Tensor

    # [num_reqs + 1]
    query_start_loc: torch.Tensor

    # [num_reqs]
    seq_lens: torch.Tensor

    # [total_num_logits]
    logits_indices: torch.Tensor

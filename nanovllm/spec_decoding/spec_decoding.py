import torch
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import sample_from_probs
class SpecDecoding:
    

    @torch.inference_mode()
    def accept(self, seqs: list[Sequence], draft_ids: torch.Tensor, draft_probs: torch.Tensor, target_probs: torch.Tensor) -> list[list[int]]:
        bs, k = draft_probs.size(0), draft_probs.size(1)
        # propose 给每条 seq 都补满了 k 个草稿，num_logits 一律是 k+1，可以直接 view
        assert all(seq.num_logits == k + 1 for seq in seqs)
        target_probs = target_probs.view(bs, k + 1, -1) # [bs * (k + 1), vocab] -> [bs, k + 1, vocab]


        # 只在草稿 token 那一个位置上，各取一个标量概率
        idx = draft_ids.unsqueeze(-1) # [bs, k, 1]
        q_sel = draft_probs.gather(dim=-1, index=idx).squeeze(-1) # [bs, k]
        p_sel = target_probs[:, :k].gather(dim=-1, index=idx).squeeze(-1) # [bs, k] 不要bonus行
        accepted = torch.rand_like(q_sel) < p_sel / q_sel.clamp_min(1e-10)
        n = accepted.to(torch.int32).cumprod(dim=-1).sum(dim=-1) # [bs]，最长 True 前缀

        # q 末尾补一行零：n == k 时 (p_k - 0)+ 归一化后正好是 bonus 分布，两种情况合并
        q_pad = torch.cat([draft_probs, torch.zeros_like(draft_probs[:, :1])], dim=1)  # [bs, k+1, vocab]
        rows = torch.arange(bs, device=n.device)
        resid = (target_probs[rows, n] - q_pad[rows, n]).clamp_min(0) # [bs, vocab]
        resid = resid / resid.sum(dim=-1, keepdim=True).clamp_min(1e-10)
        extra_token = sample_from_probs(resid) # token [bs]

        return [seq.draft_token_ids[:ni] + [e]
                for seq, ni, e in zip(seqs, n.tolist(), extra_token.tolist())]
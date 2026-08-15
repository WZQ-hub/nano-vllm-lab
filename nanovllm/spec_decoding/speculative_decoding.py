import torch

@torch.inference_mode()
def speculative_round(prompt_ids, target_model, draft_model, k):

    prompt_len = prompt_ids.shape[1]
    draft_inputs = prompt_ids
    draft_outputs = []

    for _ in range(k):
        draft_logits = draft_model.logits(draft_inputs)[:, -1, :] # [batch, prompt, vocab_size]
        draft_tokens = torch.argmax(draft_logits, dim=-1)
        draft_outputs.append(draft_tokens)
        draft_inputs = torch.cat([draft_inputs, draft_tokens[:, None]], dim=1)

    draft_outputs = torch.stack(draft_outputs, dim=1) # [batch, k]

    # target model validation
    verify_inputs = torch.cat([prompt_ids, draft_outputs], dim=1)
    logits = target_model.logits(verify_inputs)
    verify_logits = logits[:, prompt_len - 1: , :]
    output_tokens = []

    # verify draft tokens
    for i in range(k):
        draft_token = draft_outputs[:, i]
        target_token = torch.argmax(verify_logits[:, i, :], dim=-1) # [batch, prompt, vocab_size]

        if torch.equal(draft_token, target_token):
            output_tokens.append(draft_token)
        else:
            output_tokens.append(target_token)
            break
    if len(output_tokens) == k:
        bonus_token = torch.argmax(verify_logits[:, k, :], dim=-1)
        output_tokens.append(bonus_token)

    return torch.stack(output_tokens, dim=1)
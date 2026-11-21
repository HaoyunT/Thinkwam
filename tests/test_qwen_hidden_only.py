import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM
from thinkwam_p0.models.qwen import QwenConditionEncoder


@pytest.mark.parametrize('lora', [False, True])
@pytest.mark.parametrize('checkpointing', [False, True])
def test_hidden_only_preserves_hidden_states_and_gradients(monkeypatch, lora, checkpointing):
    torch.set_num_threads(1)
    def load(*args, **kwargs):
        config = Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, attention_dropout=0.1)
        return Qwen2ForCausalLM(config)
    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained', load)
    torch.manual_seed(19)
    model = QwenConditionEncoder(vjepa_dim=16, qwen_dim=32, qwen_path='unused',
                                action_tokens=4, lora=lora, lora_rank=4,
                                gradient_checkpointing=checkpointing,
                                attn_implementation='sdpa')
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'lora_B' in name:
                p.normal_(std=0.01)
    visual = torch.randn(2, 8, 16)
    ids = torch.randint(0, 128, (2, 5))
    mask = torch.tensor([[1,1,1,1,1], [1,1,1,0,0]], dtype=torch.bool)
    state_keys = tuple(model.state_dict())
    results = []
    for hidden_only in (False, True):
        model.hidden_only = hidden_only
        model.train()
        model.zero_grad(set_to_none=True)
        torch.manual_seed(83)
        output = model(visual, text_input_ids=ids, attention_mask=mask)
        output['hidden'].float().square().mean().backward()
        results.append((output['hidden'].detach().clone(),
                        {n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    assert results[0][1].keys() == results[1][1].keys()
    for name in results[0][1]:
        torch.testing.assert_close(results[0][1][name], results[1][1][name], rtol=1e-6, atol=1e-7)
    assert tuple(model.state_dict()) == state_keys
    if lora:
        assert any('lora_B' in n and p.abs().sum() > 0 for n,p in results[1][1].items())

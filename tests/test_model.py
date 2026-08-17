import torch

from da_edgeformer.config import ModelConfig
from da_edgeformer.models.edgeformer import DAEdgeFormer
from da_edgeformer.models.layers import LocalCausalSelfAttention


def small_config() -> ModelConfig:
    return ModelConfig(
        window_size=16,
        branch_width=16,
        width=16,
        attention_blocks=1,
        attention_heads=4,
        local_span=4,
        feedforward_width=32,
        adapter_bottleneck=4,
        head_hidden_width=16,
        detector_dim=8,
    )


def test_model_shapes_and_online_parameters() -> None:
    model = DAEdgeFormer(small_config())
    output = model(torch.randn(3, 16))
    assert output.power.shape == (3, 3)
    assert output.state_logits.shape == (3, 3)
    assert output.detector_features.shape == (3, 8)
    model.freeze_backbone()
    assert model.parameter_counts()["trainable"] < model.parameter_counts()["total"]
    assert all(parameter.requires_grad for parameter in model.online_parameters())


def test_manuscript_parameter_counts() -> None:
    model = DAEdgeFormer(ModelConfig())
    assert model.parameter_counts()["total"] == 636_030
    model.freeze_backbone()
    assert model.parameter_counts()["trainable"] == 73_982


def test_local_attention_is_causal() -> None:
    torch.manual_seed(2)
    attention = LocalCausalSelfAttention(width=8, heads=2, span=3).eval()
    original = torch.randn(1, 8, 8)
    changed = original.clone()
    changed[:, 5:] = torch.randn_like(changed[:, 5:]) * 100
    with torch.no_grad():
        first = attention(original)
        second = attention(changed)
    torch.testing.assert_close(first[:, :5], second[:, :5])


def test_full_encoder_is_causal() -> None:
    torch.manual_seed(3)
    model = DAEdgeFormer(small_config()).eval()
    original = torch.randn(1, 16)
    changed = original.clone()
    changed[:, 11:] += 50
    with torch.no_grad():
        first, _ = model.encode(original)
        second, _ = model.encode(changed)
    torch.testing.assert_close(first[:, :11], second[:, :11], atol=1e-5, rtol=1e-5)

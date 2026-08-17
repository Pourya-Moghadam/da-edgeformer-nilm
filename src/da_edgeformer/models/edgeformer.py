from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from da_edgeformer.config import ModelConfig
from da_edgeformer.models.layers import EncoderBlock, ResidualAdapter, TemporalBranch


@dataclass
class ModelOutput:
    power: Tensor
    state_logits: Tensor
    detector_features: Tensor


class DAEdgeFormer(nn.Module):
    """Causal convolution-attention NILM model with online-trainable adapters."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Conv1d(1, config.branch_width, kernel_size=1)
        self.branches = nn.ModuleList(
            TemporalBranch(config.branch_width, kernel, dilation)
            for kernel, dilation in zip(config.conv_kernels, config.conv_dilations, strict=True)
        )
        self.fusion = nn.Conv1d(
            config.branch_width * len(config.conv_kernels), config.width, kernel_size=1
        )
        self.blocks = nn.ModuleList(
            EncoderBlock(
                config.width,
                config.attention_heads,
                config.local_span,
                config.feedforward_width,
                config.dropout,
            )
            for _ in range(config.attention_blocks)
        )
        self.adapters = nn.ModuleList(
            ResidualAdapter(config.width, config.adapter_bottleneck)
            for _ in range(config.attention_blocks + 1)
        )
        self.output_norm = nn.LayerNorm(config.width)
        self.regression_head = self._head(config)
        self.classification_head = self._head(config)
        self.detector_projection = nn.Linear(config.width, config.detector_dim, bias=False)
        generator = torch.Generator().manual_seed(config.detector_projection_seed)
        random_matrix = torch.randn(config.width, config.detector_dim, generator=generator)
        orthogonal, _ = torch.linalg.qr(random_matrix, mode="reduced")
        with torch.no_grad():
            self.detector_projection.weight.copy_(orthogonal.T)
        self.detector_projection.weight.requires_grad_(False)

    @staticmethod
    def _head(config: ModelConfig) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(config.width, config.head_hidden_width),
            nn.GELU(),
            nn.Linear(config.head_hidden_width, len(config.appliances)),
        )

    def encode(self, aggregate: Tensor) -> tuple[Tensor, Tensor]:
        if aggregate.ndim == 2:
            aggregate = aggregate.unsqueeze(1)
        if aggregate.ndim != 3 or aggregate.shape[1] != 1:
            raise ValueError("aggregate must have shape [batch, window] or [batch, 1, window]")
        hidden = self.input_projection(aggregate)
        hidden = self.fusion(torch.cat([branch(hidden) for branch in self.branches], dim=1))
        hidden = hidden.transpose(1, 2)

        # The detector observes a frozen pre-adapter representation.
        detector_sequence = self.detector_projection(hidden).detach()
        hidden = self.adapters[0](hidden)
        for block, adapter in zip(self.blocks, self.adapters[1:], strict=True):
            hidden = adapter(block(hidden))

        return hidden, detector_sequence

    def forward(self, aggregate: Tensor) -> ModelOutput:
        hidden, detector_sequence = self.encode(aggregate)
        final = self.output_norm(hidden[:, -1])
        power = torch.relu(self.regression_head(final))
        state_logits = self.classification_head(final)
        return ModelOutput(power, state_logits, detector_sequence[:, -1])

    def freeze_backbone(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for module in [
            self.adapters,
            self.output_norm,
            self.regression_head,
            self.classification_head,
        ]:
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    def unfreeze_all(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        self.detector_projection.weight.requires_grad_(False)

    def online_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def parameter_counts(self) -> dict[str, int]:
        return {
            "total": sum(parameter.numel() for parameter in self.parameters()),
            "trainable": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
        }

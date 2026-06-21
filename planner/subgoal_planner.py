from typing import Any

import torch
from torch import nn


class SubgoalPlanner(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.depth = depth
        self.dropout = dropout

        layers: list[nn.Module] = []
        layer_input_dim = 2 * input_dim
        for _ in range(depth):
            layers.extend(
                [
                    nn.LayerNorm(layer_input_dim),
                    nn.Linear(layer_input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            layer_input_dim = hidden_dim
        layers.extend(
            [nn.LayerNorm(layer_input_dim), nn.Linear(layer_input_dim, output_dim)]
        )
        self.mlp = nn.Sequential(*layers)

    def forward(
        self, current_emb: torch.Tensor, goal_emb: torch.Tensor
    ) -> torch.Tensor:
        if goal_emb.shape != current_emb.shape or current_emb.size(1) != 1:
            raise ValueError(
                "current and goal embeddings must have matching shapes "
                f"(batch, 1, embedding_dim), got {tuple(current_emb.shape)} and "
                f"{tuple(goal_emb.shape)}"
            )

        return self.mlp(torch.concat([current_emb, goal_emb], dim=-1))


class HierarchicalWM(nn.Module):
    def __init__(self, base_model: Any, planner: SubgoalPlanner) -> None:
        super().__init__()
        self.base_model = base_model
        self.planner = planner
        self.base_model.requires_grad_(False)
        self.base_model.eval()

    def train(self, mode: bool = True) -> "HierarchicalWM":
        super().train(mode)
        self.base_model.eval()
        return self

    def encode(self, batch: dict) -> dict:
        with torch.no_grad():
            return self.base_model.encode({"pixels": batch["pixels"]})

    def forward(
        self, current_emb: torch.Tensor, goal_emb: torch.Tensor
    ) -> torch.Tensor:
        return self.planner.forward(current_emb, goal_emb)

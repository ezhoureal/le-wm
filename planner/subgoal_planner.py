from typing import Any

import torch
from torch import nn

from module import Transformer


class SubgoalPlanner(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float,
        emb_dropout: float,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.emb_dropout = emb_dropout

        self.dropout_layer = nn.Dropout(emb_dropout)
        self.goal_proj = nn.Linear(input_dim, input_dim)
        self.transformer = Transformer(
            input_dim=2 * input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

    def forward(
        self, current_emb: torch.Tensor, goal_emb: torch.Tensor
    ) -> torch.Tensor:
        if goal_emb.shape != current_emb.shape or current_emb.size(1) != 1:
            raise ValueError(
                "current and goal embeddings must have matching shapes "
                f"(batch, 1, embedding_dim), got {tuple(current_emb.shape)} and "
                f"{tuple(goal_emb.shape)}"
            )

        goal = self.goal_proj(goal_emb)
        x = torch.concat([current_emb, goal], dim=-1)
        return self.transformer(self.dropout_layer(x))


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

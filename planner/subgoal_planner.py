from typing import Any

import torch
from torch import nn

from module import Transformer


class SubgoalPlanner(nn.Module):
    def __init__(
        self,
        *,
        num_frames: int,
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
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.emb_dropout = emb_dropout

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout_layer = nn.Dropout(emb_dropout)
        self.goal_proj = nn.Linear(input_dim, input_dim)
        self.transformer = Transformer(
            input_dim=3 * input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

    def forward(
        self, context_emb: torch.Tensor, goal_emb: torch.Tensor
    ) -> torch.Tensor:
        if goal_emb.shape != context_emb.shape:
            raise ValueError(
                "goal embeddings must match context embeddings, got "
                f"{tuple(goal_emb.shape)}"
            )
        if context_emb.size(1) > self.num_frames:
            raise ValueError(
                f"context length {context_emb.size(1)} exceeds predictor window "
                f"{self.num_frames}"
            )

        goal = self.goal_proj(goal_emb)
        pos_embedding = self.pos_embedding[:, : context_emb.size(1)].expand(
            context_emb.size(0), -1, -1
        )
        x = torch.concat([context_emb, goal, pos_embedding], dim=-1)
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
        self, context_emb: torch.Tensor, goal_emb: torch.Tensor
    ) -> torch.Tensor:
        return self.planner.forward(context_emb, goal_emb)

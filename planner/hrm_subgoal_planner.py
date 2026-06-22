"""HRM architecture adapted for continuous subgoal prediction.

The recurrent hierarchy and ACT follow sapientinc/HRM's Apache-2.0
implementation. Token embeddings and classification outputs are replaced with
continuous latent projections for the planner task.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class HRMTrainingOutput:
    prediction: torch.Tensor
    prediction_loss: torch.Tensor
    q_loss: torch.Tensor
    correct: torch.Tensor
    steps: torch.Tensor
    q_halt_accuracy: torch.Tensor


def _truncated_normal_(tensor: torch.Tensor, std: float) -> torch.Tensor:
    lower = -2.0
    upper = 2.0
    with torch.no_grad():
        sqrt_two = math.sqrt(2)
        lower_cdf = math.erf(lower / sqrt_two)
        upper_cdf = math.erf(upper / sqrt_two)
        normalizer = (upper_cdf - lower_cdf) / 2
        density = (2 * math.pi) ** -0.5
        upper_density = density * math.exp(-0.5 * lower**2)
        lower_density = density * math.exp(-0.5 * upper**2)
        corrected_std = std / math.sqrt(
            1
            - (upper * upper_density - lower * lower_density) / normalizer
            - ((upper_density - lower_density) / normalizer) ** 2
        )
        tensor.uniform_(lower_cdf, upper_cdf)
        tensor.erfinv_()
        tensor.mul_(sqrt_two * corrected_std)
        tensor.clip_(lower * corrected_std, upper * corrected_std)
    return tensor


def _rms_norm(hidden_states: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = hidden_states.dtype
    normalized = hidden_states.float()
    variance = normalized.square().mean(dim=-1, keepdim=True)
    return (normalized * torch.rsqrt(variance + eps)).to(dtype)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(self, head_dim: int, sequence_length: int, theta: float) -> None:
        super().__init__()
        frequencies = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(sequence_length, dtype=torch.float32)
        angles = torch.outer(positions, frequencies)
        angles = torch.cat((angles, angles), dim=-1)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos, self.sin


class Attention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        query, key, value = self.qkv(hidden_states).chunk(3, dim=-1)
        query = query.view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = value.view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = cos_sin
        cos = cos[:sequence_length].view(1, 1, sequence_length, self.head_dim)
        sin = sin[:sequence_length].view(1, 1, sequence_length, self.head_dim)
        query_dtype = query.dtype
        query = (query * cos + _rotate_half(query) * sin).to(query_dtype)
        key = (key * cos + _rotate_half(key) * sin).to(query_dtype)

        attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(
            batch_size, sequence_length, hidden_dim
        )
        return self.output(attended)


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim: int, expansion: float) -> None:
        super().__init__()
        intermediate_dim = math.ceil(round(expansion * hidden_dim * 2 / 3) / 256) * 256
        self.gate_and_up = nn.Linear(hidden_dim, 2 * intermediate_dim, bias=False)
        self.down = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_and_up(hidden_states).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class HRMBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        expansion: float,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.attention = Attention(hidden_dim, num_heads)
        self.mlp = SwiGLU(hidden_dim, expansion)
        self.rms_norm_eps = rms_norm_eps

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = _rms_norm(
            hidden_states + self.attention(hidden_states, cos_sin),
            self.rms_norm_eps,
        )
        return _rms_norm(hidden_states + self.mlp(hidden_states), self.rms_norm_eps)


class ReasoningLevel(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        expansion: float,
        rms_norm_eps: float,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                HRMBlock(hidden_dim, num_heads, expansion, rms_norm_eps)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos_sin)
        return hidden_states


class HRMSubgoalPlanner(nn.Module):
    """Predict a subgoal latent using HRM's coupled high/low reasoning levels."""

    high_init: torch.Tensor
    low_init: torch.Tensor

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        high_cycles: int,
        low_cycles: int,
        high_layers: int,
        low_layers: int,
        halt_max_steps: int,
        halt_exploration_prob: float,
        correctness_threshold: float,
        num_heads: int,
        expansion: float,
        rms_norm_eps: float,
        rope_theta: float,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if (hidden_dim // num_heads) % 2 != 0:
            raise ValueError("attention head dimension must be even")
        if (
            min(
                high_cycles,
                low_cycles,
                high_layers,
                low_layers,
                halt_max_steps,
            )
            < 1
        ):
            raise ValueError("HRM cycles, layers, and halt steps must be positive")
        if not 0 <= halt_exploration_prob <= 1:
            raise ValueError("halt_exploration_prob must be between zero and one")
        if correctness_threshold <= 0:
            raise ValueError("correctness_threshold must be positive")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.high_cycles = high_cycles
        self.low_cycles = low_cycles
        self.halt_max_steps = halt_max_steps
        self.halt_exploration_prob = halt_exploration_prob
        self.correctness_threshold = correctness_threshold

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        self.q_head = nn.Linear(hidden_dim, 2)
        self.rotary_embedding = RotaryEmbedding(
            hidden_dim // num_heads, sequence_length=2, theta=rope_theta
        )
        self.high_level = ReasoningLevel(
            hidden_dim, num_heads, expansion, rms_norm_eps, high_layers
        )
        self.low_level = ReasoningLevel(
            hidden_dim, num_heads, expansion, rms_norm_eps, low_layers
        )
        self.register_buffer(
            "high_init", _truncated_normal_(torch.empty(hidden_dim), std=1.0)
        )
        self.register_buffer(
            "low_init", _truncated_normal_(torch.empty(hidden_dim), std=1.0)
        )
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _reason(
        self,
        high_state: torch.Tensor,
        low_state: torch.Tensor,
        input_embeddings: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            for high_step in range(self.high_cycles):
                for low_step in range(self.low_cycles):
                    is_final = (
                        high_step == self.high_cycles - 1
                        and low_step == self.low_cycles - 1
                    )
                    if not is_final:
                        low_state = self.low_level(
                            low_state, high_state + input_embeddings, cos_sin
                        )
                if high_step != self.high_cycles - 1:
                    high_state = self.high_level(high_state, low_state, cos_sin)

        low_state = self.low_level(low_state, high_state + input_embeddings, cos_sin)
        high_state = self.high_level(high_state, low_state, cos_sin)
        return high_state, low_state

    def _input_embeddings(
        self, current_emb: torch.Tensor, goal_emb: torch.Tensor
    ) -> torch.Tensor:
        if goal_emb.shape != current_emb.shape or current_emb.ndim != 3:
            raise ValueError(
                "current and goal embeddings must have matching shapes "
                f"(batch, 1, embedding_dim), got {tuple(current_emb.shape)} and "
                f"{tuple(goal_emb.shape)}"
            )
        if current_emb.size(1) != 1 or current_emb.size(2) != self.input_dim:
            raise ValueError(
                f"expected embeddings shaped (batch, 1, {self.input_dim}), "
                f"got {tuple(current_emb.shape)}"
            )

        return self.input_projection(torch.cat((current_emb, goal_emb), dim=1))

    def _initial_states(
        self, input_embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.high_init.expand_as(input_embeddings),
            self.low_init.expand_as(input_embeddings),
        )

    def forward(
        self, current_emb: torch.Tensor, goal_emb: torch.Tensor
    ) -> torch.Tensor:
        input_embeddings = self._input_embeddings(current_emb, goal_emb)
        high_state, low_state = self._initial_states(input_embeddings)
        cos_sin = self.rotary_embedding()
        batch_size = current_emb.size(0)
        halted = torch.zeros(batch_size, dtype=torch.bool, device=current_emb.device)
        final_prediction = input_embeddings.new_zeros(batch_size, 1, self.output_dim)

        for step in range(1, self.halt_max_steps + 1):
            high_state, low_state = self._reason(
                high_state.detach(),
                low_state.detach(),
                input_embeddings,
                cos_sin,
            )
            prediction = self.output_projection(high_state[:, :1])
            q_halt, q_continue = self.q_head(high_state[:, 0]).float().unbind(-1)
            should_halt = q_halt > q_continue
            if step == self.halt_max_steps:
                should_halt = torch.ones_like(should_halt)
            newly_halted = ~halted & should_halt
            final_prediction = torch.where(
                newly_halted.view(-1, 1, 1), prediction, final_prediction
            )
            halted = halted | should_halt
            if halted.all():
                break

        return final_prediction

    def training_forward(
        self,
        current_emb: torch.Tensor,
        goal_emb: torch.Tensor,
        target_emb: torch.Tensor,
    ) -> HRMTrainingOutput:
        input_embeddings = self._input_embeddings(current_emb, goal_emb)
        expected_target_shape = (current_emb.size(0), 1, self.output_dim)
        if target_emb.shape != expected_target_shape:
            raise ValueError(
                f"expected target embedding shape {expected_target_shape}, "
                f"got {tuple(target_emb.shape)}"
            )

        high_state, low_state = self._initial_states(input_embeddings)
        cos_sin = self.rotary_embedding()
        batch_size = current_emb.size(0)
        halted = torch.zeros(batch_size, dtype=torch.bool, device=current_emb.device)
        final_prediction = input_embeddings.new_zeros(batch_size, 1, self.output_dim)
        final_correct = torch.zeros(
            batch_size, dtype=torch.bool, device=current_emb.device
        )
        final_q_halt = torch.zeros(batch_size, device=current_emb.device)
        steps = torch.zeros(batch_size, dtype=torch.int32, device=current_emb.device)

        minimum_steps = torch.zeros_like(steps)
        explore = (
            torch.rand(batch_size, device=current_emb.device)
            < self.halt_exploration_prob
        )
        if self.halt_max_steps > 1:
            random_steps = torch.randint(
                2,
                self.halt_max_steps + 1,
                (batch_size,),
                device=current_emb.device,
                dtype=steps.dtype,
            )
            minimum_steps = torch.where(explore, random_steps, minimum_steps)

        per_sample_losses: list[torch.Tensor] = []
        q_halt_logits: list[torch.Tensor] = []
        q_continue_logits: list[torch.Tensor] = []
        active_masks: list[torch.Tensor] = []
        correctness: list[torch.Tensor] = []

        for step in range(1, self.halt_max_steps + 1):
            high_state, low_state = self._reason(
                high_state.detach(),
                low_state.detach(),
                input_embeddings,
                cos_sin,
            )
            prediction = self.output_projection(high_state[:, :1])
            sample_loss = (prediction.float() - target_emb.float()).pow(2).mean((1, 2))
            is_correct = sample_loss < self.correctness_threshold
            q_halt, q_continue = self.q_head(high_state[:, 0]).float().unbind(-1)
            active = ~halted

            per_sample_losses.append(sample_loss)
            q_halt_logits.append(q_halt)
            q_continue_logits.append(q_continue)
            active_masks.append(active)
            correctness.append(is_correct)

            should_halt = (q_halt > q_continue) & (step >= minimum_steps)
            if step == self.halt_max_steps:
                should_halt = torch.ones_like(should_halt)
            newly_halted = active & should_halt
            final_prediction = torch.where(
                newly_halted.view(-1, 1, 1), prediction, final_prediction
            )
            final_correct = torch.where(newly_halted, is_correct, final_correct)
            final_q_halt = torch.where(newly_halted, q_halt, final_q_halt)
            steps = torch.where(newly_halted, step, steps)
            halted = halted | should_halt

        prediction_loss = torch.cat(
            [
                loss[mask]
                for loss, mask in zip(per_sample_losses, active_masks, strict=True)
            ]
        ).mean()
        halt_loss = F.binary_cross_entropy_with_logits(
            torch.cat(
                [
                    logits[mask]
                    for logits, mask in zip(q_halt_logits, active_masks, strict=True)
                ]
            ),
            torch.cat(
                [
                    correct[mask].float()
                    for correct, mask in zip(correctness, active_masks, strict=True)
                ]
            ),
        )
        continue_loss = q_continue_logits[0].new_zeros(())
        if self.halt_max_steps > 1:
            continue_logits = torch.cat(
                [
                    q_continue_logits[index][active_masks[index]]
                    for index in range(self.halt_max_steps - 1)
                ]
            )
            with torch.no_grad():
                continue_targets = torch.cat(
                    [
                        torch.sigmoid(
                            torch.maximum(
                                q_halt_logits[index + 1],
                                q_continue_logits[index + 1],
                            )
                        )[active_masks[index]]
                        for index in range(self.halt_max_steps - 1)
                    ]
                )
            continue_loss = F.binary_cross_entropy_with_logits(
                continue_logits, continue_targets
            )

        return HRMTrainingOutput(
            prediction=final_prediction,
            prediction_loss=prediction_loss,
            q_loss=halt_loss + continue_loss,
            correct=final_correct.float().mean(),
            steps=steps.float().mean(),
            q_halt_accuracy=((final_q_halt >= 0) == final_correct).float().mean(),
        )

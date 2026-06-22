import pytest
import torch

from planner.hrm_subgoal_planner import HRMSubgoalPlanner


def make_planner() -> HRMSubgoalPlanner:
    return HRMSubgoalPlanner(
        input_dim=12,
        hidden_dim=16,
        output_dim=12,
        high_cycles=2,
        low_cycles=2,
        high_layers=1,
        low_layers=1,
        reasoning_steps=2,
        num_heads=2,
        expansion=2.0,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )


def test_hrm_subgoal_shape_and_gradients() -> None:
    planner = make_planner()
    current = torch.randn(3, 1, 12)
    goal = torch.randn(3, 1, 12)

    prediction = planner(current, goal)
    prediction.square().mean().backward()
    parameters = dict(planner.named_parameters())

    assert prediction.shape == (3, 1, 12)
    assert planner.input_projection.weight.grad is not None
    assert parameters["high_level.layers.0.attention.qkv.weight"].grad is not None
    assert parameters["low_level.layers.0.attention.qkv.weight"].grad is not None


def test_hrm_subgoal_rejects_wrong_embedding_shape() -> None:
    planner = make_planner()

    with pytest.raises(ValueError, match="expected embeddings shaped"):
        planner(torch.randn(2, 2, 12), torch.randn(2, 2, 12))


def test_hrm_subgoal_supports_bfloat16_autocast() -> None:
    planner = make_planner()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        prediction = planner(torch.randn(2, 1, 12), torch.randn(2, 1, 12))

    assert prediction.dtype == torch.bfloat16

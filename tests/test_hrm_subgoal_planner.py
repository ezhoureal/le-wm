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
        halt_max_steps=2,
        halt_exploration_prob=0.1,
        correctness_threshold=0.08,
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

    assert prediction.shape == (3, 1, 12)
    assert planner.input_projection.weight.grad is not None
    assert planner.high_level.layers[0].attention.qkv.weight.grad is not None
    assert planner.low_level.layers[0].attention.qkv.weight.grad is not None


def test_hrm_act_q_head_receives_gradients() -> None:
    planner = make_planner()
    current = torch.randn(3, 1, 12)
    goal = torch.randn(3, 1, 12)
    target = torch.randn(3, 1, 12)

    output = planner.training_forward(current, goal, target)
    (output.prediction_loss + 0.5 * output.q_loss).backward()

    assert output.prediction.shape == target.shape
    assert 1 <= output.steps <= planner.halt_max_steps
    assert planner.q_head.weight.grad is not None


def test_hrm_subgoal_rejects_wrong_embedding_shape() -> None:
    planner = make_planner()

    with pytest.raises(ValueError, match="expected embeddings shaped"):
        planner(torch.randn(2, 2, 12), torch.randn(2, 2, 12))


def test_hrm_subgoal_supports_bfloat16_autocast() -> None:
    planner = make_planner()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        prediction = planner(torch.randn(2, 1, 12), torch.randn(2, 1, 12))

    assert prediction.dtype == torch.bfloat16

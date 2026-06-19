from __future__ import annotations

import base64
import io
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import numpy as np
import stable_worldmodel as swm
import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam
    from openai.types.shared import ReasoningEffort

ReasoningEffortConfig = Literal["", "none", "minimal", "low", "medium", "high", "xhigh"]


class BlockPoseDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dx: float
    dy: float
    dtheta: float


class NextSubgoal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pusher_xy: tuple[float, float]
    block_pose_delta: BlockPoseDelta


class SubgoalSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase: str = Field(description="approach, contact, push, align, or fine_align")
    object_relation: str
    desired_contact_side: str
    next_subgoal: NextSubgoal
    rationale: str


@dataclass(frozen=True)
class SubgoalRequest:
    current_state: np.ndarray
    current_image: np.ndarray
    final_goal_pose: np.ndarray
    final_goal_image: np.ndarray
    horizon_steps: int
    env_index: int
    current_step: int
    steps_remaining: int


@dataclass(frozen=True)
class VLMProfile:
    model: str
    api_key_env: str
    base_url: str
    enable_thinking: bool
    reasoning_effort: ReasoningEffortConfig


class SubgoalProposer(Protocol):
    def propose(self, request: SubgoalRequest) -> SubgoalSpec: ...


class SubgoalRenderer(Protocol):
    def render_subgoal(
        self, spec: SubgoalSpec, request: SubgoalRequest
    ) -> Mapping[str, Any]: ...


class PushTSubgoalRenderer:
    def __init__(self, resolution: int = 224) -> None:
        from stable_worldmodel.envs.pusht.env import PushT

        self.env = PushT(resolution=resolution, render_mode="rgb_array")
        self.env._setup()

    def render_subgoal(
        self, spec: SubgoalSpec, request: SubgoalRequest
    ) -> Mapping[str, Any]:
        subgoal_state = _pusht_subgoal_state(request.current_state, spec)
        self.env.goal_pose = request.final_goal_pose
        self.env._set_state(subgoal_state)
        return {"goal": np.asarray(self.env.render())}


class OpenAISubgoalProposer:
    def __init__(
        self,
        model: str,
        system_prompt: str,
        api_key_env: str,
        base_url: str,
        enable_thinking: bool,
        reasoning_effort: ReasoningEffortConfig,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort

    def propose(self, request: SubgoalRequest) -> SubgoalSpec:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.getenv(self.api_key_env),
            base_url=self.base_url,
        )
        messages = cast(
            "list[ChatCompletionMessageParam]",
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": _subgoal_chat_content(request)},
            ],
        )
        print(f'vlm query = {messages}')
        completion_args: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if self.reasoning_effort:
            completion_args["reasoning_effort"] = cast(
                "ReasoningEffort", self.reasoning_effort
            )
        if self.enable_thinking:
            completion_args["extra_body"] = {"enable_thinking": True}

        completion = client.chat.completions.create(**completion_args)
        content_parts: list[str] = []
        for chunk in completion:
            if len(chunk.choices) == 0:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content_parts.append(delta.content)
        return _parse_subgoal_spec("".join(content_parts))


class GuidedSolver(swm.solver.ICEMSolver):
    def __init__(
        self,
        model: Any,
        subgoal_renderer: SubgoalRenderer,
        model_choice: str,
        vlm: Mapping[str, Any],
        subgoal_proposer: SubgoalProposer | None = None,
        system_prompt: str = (
            "You decompose PushT planning into one physically plausible, "
            "near-term symbolic subgoal. Return only valid JSON matching the "
            "requested schema, with no markdown fences or extra prose. "
            "Prefer subgoals reachable within the supplied horizon and useful "
            "for contact-rich progress toward the final target."
        ),
        batch_size: int = 1,
        num_samples: int = 300,
        var_scale: float = 1.0,
        n_steps: int = 30,
        topk: int = 30,
        noise_beta: float = 2.0,
        alpha: float = 0.1,
        n_elite_keep: int = 5,
        return_mean: bool = True,
        device: str | torch.device = "cpu",
        seed: int = 1234,
        eval_budget: int = 0,
        callbacks: list[Any] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            batch_size=batch_size,
            num_samples=num_samples,
            var_scale=var_scale,
            n_steps=n_steps,
            topk=topk,
            noise_beta=noise_beta,
            alpha=alpha,
            n_elite_keep=n_elite_keep,
            return_mean=return_mean,
            device=device,
            seed=seed,
            callbacks=callbacks,
        )
        self.subgoal_renderer = subgoal_renderer
        vlm_profile = _resolve_vlm_profile(model_choice, vlm)

        self.subgoal_proposer = subgoal_proposer or OpenAISubgoalProposer(
            model=vlm_profile.model,
            system_prompt=system_prompt,
            api_key_env=vlm_profile.api_key_env,
            base_url=vlm_profile.base_url,
            enable_thinking=vlm_profile.enable_thinking,
            reasoning_effort=vlm_profile.reasoning_effort,
        )
        self._final_goal: dict[str, Any] = {}
        self._has_final_goal = False
        self._solve_step = 0
        self._eval_budget = eval_budget

    def query_vlm(self, info_dict: Mapping[str, Any]) -> dict[str, Any]:
        _require_keys(
            info_dict,
            ["pos_agent", "block_pose", "vel_agent", "pixels", "goal", "goal_pose"],
        )
        total_envs = int(len(info_dict["pos_agent"]))
        rendered_items: list[Mapping[str, Any]] = []
        horizon_steps = int(self.horizon) * int(self._config.action_block)
        eval_budget = self._eval_budget or horizon_steps
        steps_remaining = max(eval_budget - self._solve_step, 0)
        for env_index in range(total_envs):
            request = SubgoalRequest(
                current_state=_raw_pusht_state(info_dict, env_index),
                current_image=_as_image_array(
                    _single_env_value(info_dict["pixels"], env_index),
                    "pixels",
                ),
                final_goal_pose=_as_goal_pose(
                    _single_env_value(info_dict["goal_pose"], env_index)
                ),
                final_goal_image=_as_image_array(
                    _single_env_value(self._final_goal["goal"], env_index),
                    "goal",
                ),
                horizon_steps=horizon_steps,
                env_index=env_index,
                current_step=self._solve_step,
                steps_remaining=steps_remaining,
            )
            spec = self.subgoal_proposer.propose(request)
            rendered = self.subgoal_renderer.render_subgoal(spec, request)
            if "goal" not in rendered:
                raise ValueError(
                    "Subgoal renderer must return a mapping with a 'goal' entry."
                )
            rendered_items.append(rendered)

        return _format_subgoal_info(_stack_rendered_items(rendered_items), info_dict)

    @torch.inference_mode()
    def solve(
        self, info_dict: dict[str, Any], init_action: torch.Tensor | None = None
    ) -> dict[str, Any]:
        if not self._has_final_goal:
            self._final_goal = _final_goal_context(info_dict)
            self._has_final_goal = True

        subgoal_info = self.query_vlm(info_dict)
        planned_info = dict(info_dict)
        planned_info.update(subgoal_info)
        planned_info.pop("goal_emb", None)
        outputs = cast(dict[str, Any], super().solve(planned_info, init_action))
        self._solve_step += int(self._config.receding_horizon) * int(
            self._config.action_block
        )
        return outputs


def _resolve_vlm_profile(model_choice: str, vlm: Mapping[str, Any]) -> VLMProfile:
    profile = cast(Mapping[str, Any], vlm["profiles"][model_choice])
    reasoning_effort = str(vlm.get("reasoning_effort", ""))

    return VLMProfile(
        model=str(profile["model"]),
        api_key_env=str(profile["api_key_env"]),
        base_url=str(profile["base_url"]),
        enable_thinking=bool(vlm.get("enable_thinking", False)),
        reasoning_effort=cast(ReasoningEffortConfig, reasoning_effort),
    )


def _final_goal_context(info_dict: Mapping[str, Any]) -> dict[str, Any]:
    _require_keys(info_dict, ["goal"])
    return {"goal": info_dict["goal"]}


def _require_keys(info_dict: Mapping[str, Any], keys: Sequence[str]) -> None:
    missing = [key for key in keys if key not in info_dict]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"LLM-guided PushT planning requires: {joined}.")


def _single_env_value(value: Any, env_index: int) -> Any:
    if torch.is_tensor(value):
        return value[env_index].detach().cpu()
    if isinstance(value, np.ndarray):
        return value[env_index]
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value[env_index]
    return value


def _stack_rendered_items(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = set(items[0])
    for item in items:
        if set(item) != keys:
            raise ValueError("Every rendered subgoal must return the same keys.")

    stacked: dict[str, Any] = {}
    for key in keys:
        values = [item[key] for item in items]
        first_value = values[0]
        if torch.is_tensor(first_value):
            stacked[key] = torch.stack([torch.as_tensor(value) for value in values])
        elif isinstance(first_value, np.ndarray):
            stacked[key] = np.stack([np.asarray(value) for value in values])
        else:
            stacked[key] = values
    return stacked


def _format_subgoal_info(
    subgoal_info: Mapping[str, Any], info_dict: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "goal": _format_rendered_goal_like_template(
            np.asarray(subgoal_info["goal"]), info_dict["goal"]
        )
    }


def _format_rendered_goal_like_template(
    images: np.ndarray, template: Any
) -> torch.Tensor:
    goal = torch.as_tensor(images)
    if goal.ndim != 4:
        raise ValueError(
            f"Expected rendered goals with shape (B, H, W, C), got {goal.shape}."
        )
    if goal.shape[-1] not in {1, 3, 4}:
        raise ValueError("Expected rendered goals to be channel-last RGB images.")

    template_tensor = torch.as_tensor(template)
    if template_tensor.ndim != 5:
        raise ValueError(
            f"Expected goal template with shape (B, T, C, H, W), got {template_tensor.shape}."
        )

    goal = goal[..., :3].permute(0, 3, 1, 2).float() / 255.0
    if goal.shape[-2:] != template_tensor.shape[-2:]:
        resized = []
        for image in goal:
            pil_image = Image.fromarray(
                (image.permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(
                    np.uint8
                )
            )
            pil_image = pil_image.resize(
                (int(template_tensor.shape[-1]), int(template_tensor.shape[-2])),
                Image.Resampling.BILINEAR,
            )
            resized.append(torch.as_tensor(np.asarray(pil_image)).permute(2, 0, 1))
        goal = torch.stack(resized).float() / 255.0

    if float(template_tensor.min()) < 0.0 or float(template_tensor.max()) > 1.0:
        mean = goal.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = goal.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        goal = (goal - mean) / std

    goal = goal.to(dtype=template_tensor.dtype)
    time_steps = int(template_tensor.shape[1])
    return goal.unsqueeze(1).expand(-1, time_steps, -1, -1, -1).clone()


def _raw_pusht_state(info_dict: Mapping[str, Any], env_index: int) -> np.ndarray:
    pos_agent = _flat_array(_single_env_value(info_dict["pos_agent"], env_index))
    block_pose = _flat_array(_single_env_value(info_dict["block_pose"], env_index))
    vel_agent = _flat_array(_single_env_value(info_dict["vel_agent"], env_index))
    return np.concatenate([pos_agent[:2], block_pose[:3], vel_agent[:2]]).astype(
        np.float64
    )


def _as_goal_pose(value: Any) -> np.ndarray:
    pose = _flat_array(value).astype(np.float64)
    if pose.shape[0] != 3:
        raise ValueError(
            f"Expected 'goal_pose' to contain 3 values, got {pose.shape[0]}."
        )
    return pose


def _flat_array(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value).reshape(-1)


def _as_image_array(value: Any, name: str) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    image = np.asarray(value)
    if image.ndim == 4:
        image = image[-1]
    if image.ndim == 3 and image.shape[0] in {1, 3, 4}:
        image = np.moveaxis(image, 0, -1)
    if image.ndim not in {2, 3}:
        raise ValueError(f"Expected '{name}' to be an image, got shape {image.shape}.")
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.ndim == 3 and image.shape[-1] not in {3, 4}:
        raise ValueError(f"Expected '{name}' to have 1, 3, or 4 channels.")
    return _to_uint8_image(image)


def _to_uint8_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    float_image = image.astype(np.float32, copy=False)
    min_value = float(float_image.min())
    max_value = float(float_image.max())
    if min_value < 0.0 or max_value > 1.0:
        denom = max(max_value - min_value, 1e-6)
        float_image = (float_image - min_value) / denom
    return np.clip(float_image * 255.0, 0.0, 255.0).astype(np.uint8)


def _pusht_subgoal_state(current_state: np.ndarray, spec: SubgoalSpec) -> np.ndarray:
    subgoal_state = current_state.copy()
    delta = spec.next_subgoal.block_pose_delta
    pusher_xy = np.asarray(spec.next_subgoal.pusher_xy, dtype=np.float64)
    subgoal_state[:2] = np.clip(pusher_xy, 0.0, 512.0)
    subgoal_state[2] = np.clip(subgoal_state[2] + delta.dx, 0.0, 512.0)
    subgoal_state[3] = np.clip(subgoal_state[3] + delta.dy, 0.0, 512.0)
    subgoal_state[4] = (subgoal_state[4] + delta.dtheta) % (2.0 * np.pi)
    subgoal_state[-2:] = 0.0
    return subgoal_state


def _subgoal_chat_content(request: SubgoalRequest) -> list[dict[str, Any]]:
    return [
        {"type": "text", "text": _subgoal_prompt(request)},
        {
            "type": "image_url",
            "image_url": {
                "url": _image_data_url(request.current_image),
                "detail": "auto",
            },
        },
        {
            "type": "image_url",
            "image_url": {
                "url": _image_data_url(request.final_goal_image),
                "detail": "auto",
            },
        },
    ]


def _subgoal_prompt(request: SubgoalRequest) -> str:
    payload = {
        "task": "Choose the next rendered subgoal for long-horizon PushT MPC.",
        "response_rules": [
            "Return exactly one JSON object and no other text.",
            "Use the top-level fields shown in response_json_example.",
            "Do not wrap the answer in a schema, contract, output_contract, or metadata key.",
            "Use PushT pixel/state coordinates for pusher_xy, not normalized coordinates.",
        ],
        "env_index": request.env_index,
        "current_step": request.current_step,
        "steps_remaining": request.steps_remaining,
        "reachable_horizon_steps": request.horizon_steps,
        "current_state": _compact_array(request.current_state),
        "current_image": {
            "role": "current observation image",
            "shape": list(request.current_image.shape),
        },
        "final_goal_pose": _compact_array(request.final_goal_pose),
        "final_goal_image": {
            "role": "final goal image",
            "shape": list(request.final_goal_image.shape),
        },
        "response_json_example": {
            "phase": "approach",
            "object_relation": "short spatial relation to the block",
            "desired_contact_side": "edge or side of the block to contact",
            "next_subgoal": {
                "pusher_xy": [215.0, 130.0],
                "block_pose_delta": {"dx": -10.0, "dy": 5.0, "dtheta": -0.15},
            },
            "rationale": "brief reason for this waypoint",
        },
    }
    return json.dumps(payload, sort_keys=True)


def _image_data_url(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _parse_subgoal_spec(content: str) -> SubgoalSpec:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1])
    print(f'vlm guidance = {stripped}')
    return SubgoalSpec.model_validate_json(stripped)


def _compact_array(value: np.ndarray) -> dict[str, Any]:
    flat = value.astype(np.float32, copy=False).reshape(-1)
    if flat.size <= 16:
        return {
            "shape": list(value.shape),
            "values": flat.tolist(),
        }
    return {
        "shape": list(value.shape),
        "mean": float(flat.mean()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "sample": flat[:16].tolist(),
    }

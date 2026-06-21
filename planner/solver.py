from typing import Any, Protocol, cast

import stable_worldmodel as swm
import torch

from planner.subgoal_planner import SubgoalPlanner


class EncodableWorldModel(Protocol):
    def encode(self, info: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: ...


class LearnedSubgoalSolver(swm.solver.CEMSolver):
    def __init__(
        self,
        model: Any,
        planner: SubgoalPlanner,
        batch_size: int,
        num_samples: int,
        var_scale: float,
        n_steps: int,
        topk: int,
        device: str,
        seed: int,
    ) -> None:
        super().__init__(
            model=model,
            batch_size=batch_size,
            num_samples=num_samples,
            var_scale=var_scale,
            n_steps=n_steps,
            topk=topk,
            device=device,
            seed=seed,
        )
        self.planner = planner

    @torch.inference_mode()
    def solve(
        self, info_dict: dict[str, Any], init_action: torch.Tensor | None = None
    ) -> dict[str, Any]:
        pixels = cast(torch.Tensor, info_dict["pixels"]).to(self.device)[:, -1:]
        goal = cast(torch.Tensor, info_dict["goal"]).to(self.device)
        model = cast(EncodableWorldModel, self.model)

        current_emb = model.encode({"pixels": pixels})["emb"]
        goal_emb = model.encode({"pixels": goal})["emb"][:, -1:]
        subgoal_emb = self.planner(current_emb, goal_emb)[:, -1]

        planned_info = dict(info_dict)
        planned_info["pixels"] = pixels
        planned_info["goal_emb"] = subgoal_emb.unsqueeze(1)
        return super().solve(planned_info, init_action)

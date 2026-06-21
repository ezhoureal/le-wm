from collections import deque
from typing import Any

import numpy as np
import stable_worldmodel as swm


class HierarchicalWMPolicy(swm.policy.WorldModelPolicy):
    def __init__(self, history_size: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.history_size = history_size
        self._pixel_history: list[deque[np.ndarray]] = []

    def set_env(self, env: Any) -> None:
        super().set_env(env)
        self._pixel_history = [
            deque(maxlen=self.history_size) for _ in range(env.num_envs)
        ]

    def get_action(self, info_dict: dict[str, Any], **kwargs: Any) -> np.ndarray:
        pixels = np.asarray(info_dict["pixels"])
        needs_flush = info_dict.get("_needs_flush")
        histories = []
        for env_index, history in enumerate(self._pixel_history):
            if needs_flush is not None and needs_flush[env_index]:
                history.clear()
            history.append(pixels[env_index, -1].copy())
            histories.append(np.stack(history))

        planned_info = dict(info_dict)
        planned_info["pixels"] = np.stack(histories)
        return super().get_action(planned_info, **kwargs)

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa
import torch

from stable_worldmodel.data import LanceDataset


class SparseLanceDataset(LanceDataset):
    def __init__(
        self,
        path: str | Path,
        num_steps: int,
        history_size: int,
        goal_steps_ahead: tuple[int, int],
        subgoal_steps_ahead: int,
        keys_to_load: list[str],
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None,
    ) -> None:
        goal_min, goal_max = goal_steps_ahead
        if history_size < 1:
            raise ValueError("history size must be positive")
        if not 0 <= goal_min <= goal_max:
            raise ValueError("goal steps must be a non-negative ordered range")
        if subgoal_steps_ahead < 0:
            raise ValueError("subgoal steps must be non-negative")
        if goal_max + history_size > num_steps:
            raise ValueError("goal frames must fit within num_steps")

        self.history_size = history_size
        self.goal_steps_ahead = goal_steps_ahead
        self.subgoal_steps_ahead = subgoal_steps_ahead
        super().__init__(
            path=path,
            frameskip=1,
            num_steps=num_steps,
            transform=transform,
            keys_to_load=keys_to_load,
        )

    def _sample_frame_offsets(self) -> tuple[int, ...]:
        goal_min, goal_max = self.goal_steps_ahead
        goal_steps_ahead = int(torch.randint(goal_min, goal_max + 1, ()).item())
        subgoal_steps_ahead = min(goal_steps_ahead, self.subgoal_steps_ahead)
        return (
            *range(self.history_size),
            *range(subgoal_steps_ahead, subgoal_steps_ahead + self.history_size),
            *range(goal_steps_ahead, goal_steps_ahead + self.history_size),
        )

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict[str, Any]:
        del end
        global_start = int(self.offsets[ep_idx] + start)
        rows = [global_start + offset for offset in self._sample_frame_offsets()]
        batch = self._fetch_rows(rows)
        steps = self._process_batch(ep_idx, global_start, batch)
        return self.transform(steps) if self.transform else steps

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        all_rows: list[int] = []
        sample_starts: list[tuple[int, int]] = []
        for index in indices:
            ep_idx, start = self.clip_indices[index]
            global_start = int(self.offsets[ep_idx] + start)
            frame_offsets = self._sample_frame_offsets()
            all_rows.extend(global_start + offset for offset in frame_offsets)
            sample_starts.append((ep_idx, global_start))

        batch = None
        if self._fetch_columns and all_rows:
            self._ensure_open()
            assert self._perm is not None
            unique_rows = sorted(set(all_rows))
            unique_batch = self._perm.__getitems__(unique_rows)
            row_lookup = {row: index for index, row in enumerate(unique_rows)}
            gather = pa.array([row_lookup[row] for row in all_rows], type=pa.int64())
            batch = unique_batch.take(gather)

        frame_count = 3 * self.history_size
        results: list[dict[str, Any]] = []
        for index, (ep_idx, global_start) in enumerate(sample_starts):
            sample_batch = (
                batch.slice(index * frame_count, frame_count)
                if batch is not None
                else None
            )
            steps = self._process_batch(ep_idx, global_start, sample_batch)
            if self.transform:
                steps = self.transform(steps)
            results.append(steps)

        return results

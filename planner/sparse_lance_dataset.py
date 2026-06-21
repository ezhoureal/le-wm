from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa

from stable_worldmodel.data import LanceDataset


class SparseLanceDataset(LanceDataset):
    def __init__(
        self,
        path: str | Path,
        num_steps: int,
        frame_offsets: tuple[int, ...],
        keys_to_load: list[str],
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None,
    ) -> None:
        if not frame_offsets or tuple(sorted(set(frame_offsets))) != frame_offsets:
            raise ValueError("frame offsets must be sorted and unique")
        if frame_offsets[-1] >= num_steps:
            raise ValueError("frame offsets must fit within num_steps")

        self.frame_offsets = frame_offsets
        super().__init__(
            path=path,
            frameskip=1,
            num_steps=num_steps,
            transform=transform,
            keys_to_load=keys_to_load,
        )

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict[str, Any]:
        del end
        global_start = int(self.offsets[ep_idx] + start)
        rows = [global_start + offset for offset in self.frame_offsets]
        batch = self._fetch_rows(rows)
        steps = self._process_batch(ep_idx, global_start, batch)
        return self.transform(steps) if self.transform else steps

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        all_rows: list[int] = []
        sample_starts: list[tuple[int, int]] = []
        for index in indices:
            ep_idx, start = self.clip_indices[index]
            global_start = int(self.offsets[ep_idx] + start)
            all_rows.extend(global_start + offset for offset in self.frame_offsets)
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

        frame_count = len(self.frame_offsets)
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

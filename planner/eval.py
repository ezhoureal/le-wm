import os

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

from planner.policy import HierarchicalWMPolicy
from planner.subgoal_planner import HierarchicalWM

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def img_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.Resize(size=img_size),
        ]
    )


def get_episode_lengths(dataset: Any, episode_ids: np.ndarray) -> np.ndarray:
    episode_key = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    dataset_episode_ids = dataset.get_col_data(episode_key)
    step_ids = dataset.get_col_data("step_idx")
    return np.array(
        [
            step_ids[dataset_episode_ids == episode_id].max() + 1
            for episode_id in episode_ids
        ]
    )


def get_dataset(cfg: DictConfig) -> Any:
    cache_dir = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        cfg.eval.dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=cache_dir,
    )


def get_processors(dataset: Any, keys: list[str]) -> dict[str, Any]:
    processors = {}
    for key in keys:
        if key == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        values = dataset.get_col_data(key)
        processor.fit(values[~np.isnan(values).any(axis=1)])
        processors[key] = processor
        if key != "action":
            processors[f"goal_{key}"] = processor
    return processors


def sample_starts(
    dataset: Any, num_eval: int, goal_offset: int, seed: int
) -> tuple[list[int], list[int]]:
    episode_key = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_ids = np.unique(dataset.get_col_data(episode_key))
    max_start_steps = get_episode_lengths(dataset, episode_ids) - goal_offset - 1
    max_start_by_episode = dict(zip(episode_ids, max_start_steps, strict=True))
    max_start_per_row = np.array(
        [max_start_by_episode[value] for value in dataset.get_col_data(episode_key)]
    )
    valid_indices = np.flatnonzero(
        dataset.get_col_data("step_idx") <= max_start_per_row
    )
    if len(valid_indices) < num_eval:
        raise ValueError(
            f"evaluation needs {num_eval} valid starting points, got {len(valid_indices)}"
        )

    generator = np.random.default_rng(seed)
    rows = np.sort(generator.choice(valid_indices, size=num_eval, replace=False))
    row_data = dataset.get_row_data([int(row) for row in rows])
    return row_data[episode_key].tolist(), row_data["step_idx"].tolist()


@hydra.main(version_base=None, config_path="../config/eval", config_name="planner")
def run(cfg: DictConfig) -> None:
    if cfg.plan_config.horizon * cfg.plan_config.action_block > cfg.eval.eval_budget:
        raise ValueError("planning horizon cannot exceed the evaluation budget")

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(cfg.eval.img_size, cfg.eval.img_size))
    dataset = get_dataset(cfg)

    model: HierarchicalWM = swm.wm.utils.load_pretrained(
        cfg.planner_checkpoint, cache_dir=cfg.planner_cache_dir
    )
    model.to(cfg.device).eval().requires_grad_(False)
    model.base_model.interpolate_pos_encoding = True

    solver = hydra.utils.instantiate(
        cfg.solver, model=model.base_model, planner=model.planner
    )
    policy = HierarchicalWMPolicy(
        history_size=model.planner.num_frames,
        solver=solver,
        config=swm.PlanConfig(**cfg.plan_config),
        process=get_processors(dataset, list(cfg.dataset.keys_to_cache)),
        transform={
            "pixels": img_transform(cfg.eval.img_size),
            "goal": img_transform(cfg.eval.img_size),
        },
    )
    world.set_policy(policy)

    episodes, start_steps = sample_starts(
        dataset,
        num_eval=cfg.eval.num_eval,
        goal_offset=cfg.eval.goal_offset_steps,
        seed=cfg.seed,
    )
    callables_container = OmegaConf.to_container(cfg.eval.callables, resolve=True)
    if not isinstance(callables_container, list):
        raise TypeError("eval.callables must resolve to a list")

    output_dir = Path(cfg.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=start_steps,
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=episodes,
        callables=callables_container,
        video=output_dir,
    )
    elapsed = time.time() - started_at
    print(metrics)

    with (output_dir / cfg.output.filename).open("a") as output_file:
        output_file.write("\n==== CONFIG ====\n")
        output_file.write(OmegaConf.to_yaml(cfg))
        output_file.write("\n==== RESULTS ====\n")
        output_file.write(f"metrics: {metrics}\n")
        output_file.write(f"evaluation_time: {elapsed} seconds\n")


if __name__ == "__main__":
    run()

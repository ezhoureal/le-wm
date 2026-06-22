import os
from functools import partial
from pathlib import Path
from typing import Any, cast

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from planner.hrm_subgoal_planner import HRMSubgoalPlanner
from planner.sparse_lance_dataset import SparseLanceDataset
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


def model_forward(self, batch, stage, cfg):
    """Encode observations and train the subgoal predictor."""
    output = self.model.encode(batch)
    # emb shape: (B, T, D).
    # T contains [current frame, subgoal frame, goal frame].
    emb = output["emb"]
    if emb.size(1) != 3:
        raise ValueError(f"subgoal training needs 3 sparse frames, got {emb.size(1)}")

    current_emb = emb[:, :1]
    target_emb = emb[:, 1:2]
    goal_emb = emb[:, 2:]
    if cfg.act.enabled:
        planner = cast(HRMSubgoalPlanner, self.model.planner)
        result = planner.training_forward(current_emb, goal_emb, target_emb)
        output["pred_loss"] = result.prediction_loss
        output["q_loss"] = result.q_loss
        output["final_pred_loss"] = (result.prediction - target_emb).pow(2).mean()
        output["correct"] = result.correct
        output["reasoning_steps"] = result.steps
        output["q_halt_accuracy"] = result.q_halt_accuracy
        output["loss"] = result.prediction_loss + cfg.act.q_loss_weight * result.q_loss
    else:
        pred_emb = self.model(current_emb, goal_emb)
        output["loss"] = output["pred_loss"] = (pred_emb - target_emb).pow(2).mean()

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    metrics_dict = {
        f"{stage}/{key}": output[key].detach()
        for key in ("correct", "reasoning_steps", "q_halt_accuracy")
        if key in output
    }
    self.log_dict(losses_dict | metrics_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path="../config/train", config_name="subgoal")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_path = Path(cfg.data.dataset.name)
    if not dataset_path.is_absolute():
        cache_dir = os.environ.get("LOCAL_DATASET_DIR")
        cache_root = Path(cache_dir) if cache_dir else None
        dataset_path = (
            swm.data.utils.get_cache_dir(cache_root, sub_folder="datasets")
            / dataset_path
        )
    dataset = SparseLanceDataset(
        path=dataset_path,
        num_steps=cfg.data.dataset.num_steps,
        goal_steps_ahead=tuple(cfg.goal_steps_ahead),
        subgoal_steps_ahead=cfg.subgoal_steps_ahead,
        keys_to_load=list(cfg.data.dataset.keys_to_load),
        transform=None,
    )
    transforms: list[Any] = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size),
    ]

    for col in cfg.data.dataset.keys_to_load:
        if col.startswith("pixels"):
            continue
        normalizer = get_column_normalizer(dataset, col, col)
        transforms.append(normalizer)

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    episode_ids = torch.randperm(len(dataset.lengths), generator=rnd_gen)
    num_train_episodes = int(len(episode_ids) * cfg.train_split)
    train_episode_ids = set(episode_ids[:num_train_episodes].tolist())
    train_indices = [
        index
        for index, (episode_id, _) in enumerate(dataset.clip_indices)
        if episode_id in train_episode_ids
    ]
    val_indices = [
        index
        for index, (episode_id, _) in enumerate(dataset.clip_indices)
        if episode_id not in train_episode_ids
    ]
    torch_dataset = cast(torch.utils.data.Dataset[dict[str, Any]], dataset)
    train_set, val_set = (
        torch.utils.data.Subset(torch_dataset, train_indices),
        torch.utils.data.Subset(torch_dataset, val_indices),
    )

    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        shuffle=True,
        drop_last=True,
        generator=rnd_gen,
    )
    val = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False
    )

    ##############################
    ##       model / optim      ##
    ##############################

    pretrained_model = swm.wm.utils.load_pretrained(
        cfg.pretrained_model,
        cache_dir=cfg.pretrained_cache_dir,
    )
    world_model = hydra.utils.instantiate(cfg.model)
    world_model.base_model.load_state_dict(pretrained_model.state_dict())
    optimizers = {
        "model_opt": {
            "modules": r"model\.planner",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        forward=partial(model_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(cast(dict[str, Any], OmegaConf.to_container(cfg)))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg.model,
        epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()

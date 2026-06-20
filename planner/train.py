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

from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


def model_forward(self, batch, stage, cfg):
    """Encode observations and train the subgoal predictor."""

    ctx_len = cfg.history_size
    pred_frame = cfg.subgoal_steps_ahead
    goal_frame = cfg.goal_steps_ahead

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    needed_frames = max(pred_frame, goal_frame) + ctx_len
    if emb.size(1) < needed_frames:
        raise ValueError(
            f"subgoal training needs at least {needed_frames} frames, got "
            f"{emb.size(1)}. Set data.dataset.num_steps >= "
            "max(subgoal_steps_ahead, goal_steps_ahead) + history_size."
        )

    ctx_emb = emb[:, :ctx_len]
    tgt_emb = emb[:, pred_frame : pred_frame + ctx_len]
    goal_emb = emb[:, goal_frame : goal_frame + ctx_len]
    pred_emb = self.model(ctx_emb, goal_emb)

    output["loss"] = output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path="../config/train", config_name="subgoal")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = cast(
        dict[str, Any], OmegaConf.to_container(cfg.data.dataset, resolve=True)
    )
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR")
    dataset_kwargs = {"transform": None, **dataset_cfg}
    if cache_dir is not None:
        dataset_kwargs["cache_dir"] = cache_dir
    dataset = swm.data.load_dataset(dataset_name, **dataset_kwargs)
    transforms: list[Any] = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    ]

    for col in cfg.data.dataset.keys_to_load:
        if col.startswith("pixels"):
            continue
        normalizer = get_column_normalizer(dataset, col, col)
        transforms.append(normalizer)

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
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

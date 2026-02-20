import os
import gc
import math
import argparse
import pprint
import warnings
from pathlib import Path
from distutils.util import strtobool

import cv2
import numpy as np
from loguru import logger as loguru_logger

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from src.loftr import LoFTR
from src.loftr.utils.supervision import compute_supervision_coarse, compute_supervision_fine
from src.losses.loftr_loss import LoFTRLoss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.metrics import (
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    aggregate_metrics,
)
from src.utils.comm import gather
from src.utils.misc import lower_config, flattenList, get_rank_zero_only_logger
from src.utils.profiler import PassThroughProfiler, build_profiler
from src.config.default import get_cfg_defaults

warnings.filterwarnings("ignore")

# =========================
# Global configs / paths
# =========================
MAIN_CFG_PATH = "configs/loftr_ds_dense.py"
SMARTPHONE_DATA_ROOT = "dataset/"

TRAIN_BATCH_SIZE = 36
DEVICES = [0,1,2]  # e.g. [0] or [0,1,2,3]
MAX_EPOCHS = 30
TRUE_BATCH_SIZE = TRAIN_BATCH_SIZE * len(DEVICES)


# =========================
# Lightweight weight-only saving callback
# =========================
class SaveWeightsOnlyCallback(pl.Callback):
    """
    Save ONLY the matcher weights (state_dict) to minimize disk usage.
    This avoids Lightning full checkpoints (optimizer/scheduler/trainer states).
    """
    def __init__(self, save_dir: str, save_every_n_epochs: int = 1):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_epochs = int(save_every_n_epochs)

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = int(trainer.current_epoch)
        if (epoch + 1) % self.save_every_n_epochs != 0:
            return

        save_path = self.save_dir / f"epoch{epoch:03d}_weights.pt"

        # Save ONLY LoFTR matcher weights
        state_dict = pl_module.matcher.state_dict()
        torch.save(state_dict, save_path)

        print(f"[weights-only] Saved → {save_path}")


# =========================
# Lightning module
# =========================
class PL_LoFTR(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        super().__init__()
        self.config = config
        _config = lower_config(self.config)
        self.loftr_cfg = lower_config(_config["loftr"])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = 1

        # Matcher and loss
        self.matcher = LoFTR(config=_config["loftr"])
        self.loss = LoFTRLoss(_config)

        # Optional pretrained weights
        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location="cuda")["state_dict"]
            self.matcher.load_state_dict(state_dict, strict=True)
            loguru_logger.info(f"Loaded pretrained checkpoint: {pretrained_ckpt}")

        self.dump_dir = dump_dir

    def configure_optimizers(self):
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]

    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_idx,
        optimizer_closure, on_tpu, using_native_amp, using_lbfgs
    ):
        # LR warmup (kept)
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == "linear":
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + (self.trainer.global_step / warmup_step) * abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
            elif self.config.TRAINER.WARMUP_TYPE == "constant":
                pass
            else:
                raise ValueError(f"Unknown warm-up type: {self.config.TRAINER.WARMUP_TYPE}")

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def _trainval_inference(self, batch):
        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, self.config)

        with self.profiler.profile("LoFTR"):
            self.matcher(batch)

        with self.profiler.profile("Compute fine supervision"):
            compute_supervision_fine(batch, self.config)

        with self.profiler.profile("Compute losses"):
            self.loss(batch)

    def _trainval_inference_no_supervision(self, batch):
        with self.profiler.profile("LoFTR"):
            self.matcher(batch)
        with self.profiler.profile("Compute losses"):
            self.loss(batch)

    def training_step(self, batch, batch_idx):
        # Your original code uses _trainval_inference1 (no supervision calls)
        self._trainval_inference_no_supervision(batch)

        # Logging (rank0 only)
        if self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            for k, v in batch["loss_scalars"].items():
                self.logger.experiment.add_scalar(f"train/{k}", v, self.global_step)

        gc.collect()
        torch.cuda.empty_cache()
        return {"loss": batch["loss"]}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar("train/avg_loss_on_epoch", avg_loss, global_step=self.current_epoch)
        gc.collect()
        torch.cuda.empty_cache()

    def _compute_metrics(self, batch):
        with self.profiler.profile("Compute metrics"):
            compute_symmetrical_epipolar_errors(batch)
            compute_pose_errors(batch, self.config)

            rel_pair_names = list(zip(*batch["pair_names"]))
            bs = batch["image0"].size(0)
            metrics = {
                "identifiers": ["#".join(rel_pair_names[b]) for b in range(bs)],
                "epi_errs": [batch["epi_errs"][batch["m_bids"] == b].cpu().numpy() for b in range(bs)],
                "R_errs": batch["R_errs"],
                "t_errs": batch["t_errs"],
                "inliers": batch["inliers"],
            }
            ret_dict = {"metrics": metrics}
        return ret_dict, rel_pair_names

    def test_step(self, batch, batch_idx):
        with self.profiler.profile("LoFTR"):
            self.matcher(batch)

        ret_dict, rel_pair_names = self._compute_metrics(batch)
        return ret_dict

    def test_epoch_end(self, outputs):
        _metrics = [o["metrics"] for o in outputs]
        metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}

        if self.trainer.global_rank == 0:
            print(self.profiler.summary())
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)
            loguru_logger.info("\n" + pprint.pformat(val_metrics_4tb))


# =========================
# Dataset / DataModule
# =========================
class SmartphoneDataset(Dataset):
    def __init__(self, mode="train", **kwargs):
        super().__init__()
        self.root = SMARTPHONE_DATA_ROOT
        self.left_dir = os.path.join(self.root, "Left_fig")
        self.right_dir = os.path.join(self.root, "Right_fig")
        self.coarse_dir = os.path.join(self.root, "Label")

        self.img_names = sorted([f for f in os.listdir(self.left_dir) if f.endswith(".jpg")])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        name = self.img_names[idx]

        img1 = cv2.imread(os.path.join(self.left_dir, name), 0)
        img2 = cv2.imread(os.path.join(self.right_dir, name), 0)

        img1 = cv2.resize(img1, (480, 640))
        img2 = cv2.resize(img2, (480, 640))

        img1 = torch.from_numpy(img1)[None].float() / 255.0
        img2 = torch.from_numpy(img2)[None].float() / 255.0

        conf = cv2.imread(os.path.join(self.coarse_dir, name), 0)
        if conf is None:
            conf = np.zeros((300, 300), dtype=np.uint8)
        else:
            conf = conf.copy()
            conf[conf < 150] = 0
            conf[conf >= 150] = 1

        conf = torch.from_numpy(conf)[None].long()

        return {
            "image0": img1,
            "image1": img2,
            "gtconf_matrix": conf,
        }


class MultiSceneDataModule(pl.LightningDataModule):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.train_dataset = SmartphoneDataset(mode="train")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=8,
            pin_memory=False,
            drop_last=True,
        )


# =========================
# CLI args
# =========================
loguru_logger = get_rank_zero_only_logger(loguru_logger)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("main_cfg_path", type=str, help="main config path")
    parser.add_argument("--exp_name", type=str, default="default_exp_name")
    parser.add_argument("--batch_size", type=int, default=8, help="batch_size per gpu")
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument(
        "--pin_memory",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        default=False,
        help="whether to use pinned memory",
    )

    parser.add_argument("--disable_ckpt", action="store_true", help="disable saving weights (debug only)")
    parser.add_argument("--profiler_name", type=str, default=None, help="options: [inference, pytorch]")

    # Add PL Trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # NOTE: Keep your original fixed argument string behavior
    return parser.parse_args(
        f"{MAIN_CFG_PATH} --exp_name test --num_nodes 1 --accelerator gpu "
        "--check_val_every_n_epoch 1 --log_every_n_steps 1 "
        "--limit_val_batches 1 --num_sanity_val_steps 10 "
        f"--benchmark True --max_epochs {MAX_EPOCHS}".split()
    )


# =========================
# Train entry
# =========================
def train():
    args = parse_args()
    rank_zero_only(pprint.pprint)(vars(args))

    # Load config
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)

    pl.seed_everything(config.TRAINER.SEED)

    # LR scaling (kept)
    config.TRAINER.TRUE_BATCH_SIZE = TRUE_BATCH_SIZE
    scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.SCALING = scaling
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * scaling
    config.TRAINER.WARMUP_STEP = math.floor(config.TRAINER.WARMUP_STEP / scaling)

    # Module / Data
    profiler = build_profiler(args.profiler_name)
    model = PL_LoFTR(config, profiler=profiler)
    loguru_logger.info("LoFTR LightningModule initialized!")

    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info("LoFTR DataModule initialized!")

    # Logger
    tb_logger = TensorBoardLogger(save_dir="logs/tb_logs", name=args.exp_name, default_hp_metric=False)

    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [lr_monitor]

    # IMPORTANT: Save weights only (no big .ckpt)
    if not args.disable_ckpt:
        weights_dir = Path(tb_logger.log_dir) / "weights"
        callbacks.append(SaveWeightsOnlyCallback(save_dir=str(weights_dir), save_every_n_epochs=10))

    # Trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        # Disable Lightning checkpointing to avoid default heavy .ckpt
        enable_checkpointing=False,
        logger=tb_logger,
        gradient_clip_val=1.0,
        strategy="ddp",
        accelerator="gpu",
        devices=DEVICES,
        callbacks=callbacks,
        profiler=profiler,
    )

    loguru_logger.info("Trainer initialized!")
    loguru_logger.info("Start training!")
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    train()

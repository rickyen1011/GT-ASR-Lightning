import pathlib
import argparse
import json5
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from utils.utils import initialize_config, load_config

CKPT_DIR = "checkpoints"
torch.set_float32_matmul_precision("medium")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate a model using PyTorch Lightning."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=pathlib.Path,
        help="onfiguration file (JSON).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=pathlib.Path,
        help="Path to a pre-trained model checkpoint file.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint.",
    )
    parser.add_argument(
        "--gpus",
        default=1,
        type=int,
        help="Number of GPUs to use for training. (Default: 1)",
    )
    parser.add_argument(
        "--seed", default=1109, type=int, help="Random seed. (Default: 1109)"
    )
    parser.add_argument(
        "--gradient-clip-val",
        default=1.0,
        type=float,
        help="Gradient clipping threshold. (Default: 1.0)",
    )
    return parser.parse_args()


def setup_experiment_dirs(config, config_path, checkpoint_path):
    """Set up experiment directories and save configuration."""
    model_name = config_path.stem
    method_name = config_path.parents[1].stem
    exp_dir = (
        pathlib.Path("Experiments")
        / config["exp_dir"]
        / config["unit"]
        / method_name
        / model_name
    )
    exp_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "config": config,
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "exp_dir": str(exp_dir),
    }

    config_path = exp_dir / "config.json"
    with open(config_path, "w") as f:
        json5.dump(config, f, indent=2, sort_keys=False)

    return exp_dir


def get_trainer(exp_dir, trainer_config, gpus, gradient_clip_val):
    """Configure and return a PyTorch Lightning Trainer instance."""
    checkpoint_dir = exp_dir / CKPT_DIR
    val_acc_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="val/acc",
        mode="max",
        filename="best",
        save_top_k=1,
        verbose=True,
        save_weights_only=True,
    )
    val_ter_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="val/ter",
        mode="min",
        save_top_k=3,
        verbose=True,
        save_last=True,
    )
    early_stop = EarlyStopping(
        monitor=trainer_config["early_stop"]["monitor"],
        mode=trainer_config["early_stop"]["mode"],
        patience=trainer_config["early_stop"]["patience"]
    )
    callbacks = [val_acc_checkpoint, val_ter_checkpoint, early_stop]
    return Trainer(
        default_root_dir=exp_dir,
        strategy="ddp",
        accelerator="gpu",
        devices=gpus,
        callbacks=callbacks,
        gradient_clip_val=gradient_clip_val,
        max_epochs=trainer_config["max_epochs"],
        val_check_interval=trainer_config["val_check_interval"],
        accumulate_grad_batches=trainer_config["accumulate_grad_batches"],
    )


def main():
    """Main function."""
    args = parse_args()
    if args.checkpoint_path and args.resume:
        raise ValueError(
            "Resume conflict with preloaded model. Please use one of them."
        )

    seed_everything(seed=args.seed)

    config = load_config(args.config)
    exp_dir = setup_experiment_dirs(
        config,
        args.config,
        args.checkpoint_path,
    )

    train_dataset, val_dataset = (
        initialize_config(config["train_dataset"]),
        initialize_config(config["validation_dataset"]),
    )
    lightning_module = initialize_config(
        config["lightning_module"], pass_args=False
    )(config)

    trainer = get_trainer(
        exp_dir, config["trainer"], args.gpus, args.gradient_clip_val
    )
    ckpt_path = (
        (exp_dir / CKPT_DIR / "last.ckpt") if args.resume else args.checkpoint_path
    )

    train_dataloader = lightning_module.get_train_dataloader(train_dataset)
    val_dataloader = lightning_module.get_val_dataloader(val_dataset)

    trainer.fit(
        lightning_module,
        train_dataloader,
        val_dataloader,
        ckpt_path=str(ckpt_path) if ckpt_path else None,
    )


if __name__ == "__main__":
    main()

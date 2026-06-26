import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import ParcelDataset, collate_fn
from losses import RTMDetLitModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str,
                        default="/Volumes/playground/it_cl_terminal/ilo/data/blender_generated/images")
    parser.add_argument("--labels", type=str,
                        default="/Volumes/playground/it_cl_terminal/ilo/data/blender_generated/labels")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num_queries", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    pl.seed_everything(42)

    full = ParcelDataset(args.images, args.labels, img_size=args.img_size,
                         max_parcels=args.num_queries)
    n_val = max(1, int(len(full) * args.val_split))
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(full, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn,
                            persistent_workers=args.num_workers > 0)

    model = RTMDetLitModule(lr=args.lr)

    callbacks = [
        ModelCheckpoint(
            dirpath=args.ckpt_dir,
            filename="parcel3d",
            monitor="val/loss", mode="min", save_top_k=3,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(monitor="val/loss", patience=30, mode="min"),
    ]

    logger = TensorBoardLogger("tb_logs", name="parcel3d_detr")

    # Auto pick best accelerator (mps on Apple Silicon, cuda otherwise, fallback cpu)
    if torch.cuda.is_available():
        accelerator, devices, precision = "gpu", 1, "16-mixed"
    elif torch.backends.mps.is_available():
        accelerator, devices, precision = "mps", 1, "32-true"
    else:
        accelerator, devices, precision = "cpu", 1, "32-true"

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=0.5,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
import argparse
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")
from torch.utils.data import ConcatDataset, DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from dataset import ParcelDataset, RealPackDataset, collate_fn
from losses import RTMDetLitModule
from model import FX_REF


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, nargs="+", default=["TASLER_MULTI", "TASLER"],
                        help="one or more TASLER-style root folders (pooled together) "
                             "containing shard_XXXXXXXX/ subfolders")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_queries", type=int, default=30,
                        help="cap on labelled objects per image (max_parcels)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--n_val", type=int, default=600,
                        help="number of held-out validation images (seed-42 permutation; "
                             "evaluate.py must use the same value)")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--run_name", type=str, default="parcel3d",
                        help="used for checkpoint filename prefix and TB logger name")
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--num_head_convs", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=128,
                        help="FPN / head width. 128 is ~1.15x faster than 256 at "
                             "equal accuracy on this single-class problem.")
    parser.add_argument("--depth_canon_range", type=float, nargs=2, default=(0.4, 28.0),
                        help="sigmoid output range of the CANONICAL depth head "
                             "(metres at fx=FX_REF). Must cover depth*FX_REF/fx over "
                             "the dataset AFTER scale augmentation: the data spans "
                             "0.71-11.31, and a zoom by s divides it by s, so the "
                             "default s in [0.5, 1.25] needs 0.57-22.6.")
    parser.add_argument("--fx_ref", type=float, default=FX_REF,
                        help="reference focal length defining canonical depth; "
                             "must match at inference (stored in the checkpoint)")
    parser.add_argument("--no_augment", action="store_true",
                        help="disable photometric + quarter-turn + scale augmentation")
    parser.add_argument("--no_scale_augment", action="store_true",
                        help="keep photometric + quarter-turn but disable the "
                             "random zoom (dataset.scale_target)")
    parser.add_argument("--scale_range", type=float, nargs=2, default=(0.5, 1.25),
                        help="random zoom range; Cube R-CNN's [0.50, 1.25]")
    parser.add_argument("--real_dir", type=str, default=None,
                        help="folder of REAL frames with 2D rotated-box labels only "
                             "(packstuecke-style *.jpg + *.txt). Added to the TRAINING "
                             "set with partial supervision (detection, box, angle, "
                             "type; no metric size or depth). Never added to "
                             "validation, so the synthetic val split - and val/loss - "
                             "stay comparable with runs that did not use it. Keep the "
                             "real test frames in a separate folder.")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--resume", type=str, default=None,
                        help="resume an INTERRUPTED run from a checkpoint, restoring "
                             "model + optimizer + LR-schedule + epoch counter so "
                             "training continues exactly where it stopped. Model "
                             "hyperparameters are taken from the checkpoint, so "
                             "architecture flags on the command line are ignored. "
                             "Keep --epochs the same as the original run, or the "
                             "cosine schedule will not line up.")
    parser.add_argument("--init_ckpt", type=str, default=None,
                        help="warm-start model weights from this checkpoint "
                             "(fresh optimizer + a new warmup/cosine schedule "
                             "sized to --epochs/--lr) — for continuing training "
                             "past a finished run, not resuming an interrupted one")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    pl.seed_everything(42)

    # Two views of the same file list: the training subset gets augmentation,
    # the validation subset must not. A single dataset + random_split cannot do
    # this (the augment flag lives on the dataset, not the split), which is why
    # augmentation was silently never enabled before.
    aug = not args.no_augment
    ds_kwargs = dict(img_size=args.img_size, max_parcels=args.num_queries)
    train_full = ParcelDataset(args.data_dir, augment=aug, geom_augment=aug,
                               scale_augment=aug and not args.no_scale_augment,
                               scale_range=tuple(args.scale_range), **ds_kwargs)
    val_full = ParcelDataset(args.data_dir, augment=False, geom_augment=False, **ds_kwargs)

    n_total = len(train_full)
    n_val = max(1, min(args.n_val, n_total - 1))
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(42)).tolist()
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx)
    if args.real_dir:
        real_ds = RealPackDataset(args.real_dir, img_size=args.img_size,
                                  augment=aug, geom_augment=aug,
                                  scale_augment=aug and not args.no_scale_augment,
                                  scale_range=tuple(args.scale_range))
        train_ds = ConcatDataset([train_ds, real_ds])
        print(f"[train] + {len(real_ds)} real 2D-labelled frames from {args.real_dir} "
              f"(training only)")
    print(f"[train] {len(train_ds)} train / {len(val_ds)} val images, "
          f"augment={'on' if aug else 'off'}, "
          f"scale_augment={'on' if train_full.scale_augment else 'off'}")

    loader_kwargs = dict(num_workers=args.num_workers, collate_fn=collate_fn,
                         persistent_workers=args.num_workers > 0,
                         pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            **loader_kwargs)

    model_kwargs = dict(lr=args.lr, weight_decay=args.weight_decay,
                        backbone=args.backbone, num_head_convs=args.num_head_convs,
                        d_model=args.d_model,
                        depth_canon_range=tuple(args.depth_canon_range),
                        fx_ref=args.fx_ref,
                        img_size=args.img_size, epochs=args.epochs)
    if args.resume:
        # Rebuild with the checkpoint's own hparams: Trainer(ckpt_path=...) restores
        # weights into whatever module it is handed, so constructing a differently
        # shaped one from CLI defaults would fail (or silently mismatch).
        model = RTMDetLitModule.load_from_checkpoint(args.resume, map_location="cpu")
        print(f"[train] resuming from {args.resume} "
              f"(epoch {torch.load(args.resume, map_location='cpu', weights_only=False)['epoch']})")
    elif args.init_ckpt:
        model = RTMDetLitModule.load_from_checkpoint(args.init_ckpt, map_location="cpu", **model_kwargs)
    else:
        model = RTMDetLitModule(**model_kwargs)

    callbacks = [
        ModelCheckpoint(
            dirpath=args.ckpt_dir,
            filename=args.run_name + "-{epoch:02d}",
            monitor="val/loss", mode="min", save_top_k=3,
            save_last=True,          # last.ckpt -> exact resume point after any stop
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(monitor="val/loss", patience=args.patience, mode="min"),
    ]

    # TensorBoard is optional here — fall back to CSV rather than killing a
    # multi-hour run at setup time over a missing viewer dependency.
    try:
        logger = TensorBoardLogger("tb_logs", name=args.run_name)
    except ModuleNotFoundError:
        print("[train] tensorboard not installed; logging to csv_logs/ instead "
              "(pip install tensorboard to restore TB logging)")
        logger = CSVLogger("csv_logs", name=args.run_name)

    # Auto pick best accelerator (mps on Apple Silicon, cuda otherwise, fallback cpu)
    if torch.cuda.is_available():
        accelerator, devices, precision = "gpu", 1, "16-mixed"
        # channels_last suits the NHWC tensor-core kernels AMP dispatches to;
        # worth ~10% end-to-end here and free.
        model.model = model.model.to(memory_format=torch.channels_last)
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
        log_every_n_steps=25,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)

    ckpt_cb = callbacks[0]
    print(f"BEST_CKPT: {ckpt_cb.best_model_path}")
    print(f"BEST_SCORE: {ckpt_cb.best_model_score}")


if __name__ == "__main__":
    main()

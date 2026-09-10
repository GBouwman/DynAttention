import torch
import pytorch_lightning as pl
from pathlib import Path

from dynvision.models import DyRCNNx4
from dynvision import project_paths
from dynvision.data.ffcv_dataloader import get_ffcv_dataloader

# 1. Ensure you have generated the FFCV .beton files first!
# Run this via snakemake (from the dynvision/workflow directory):
# snakemake /home/glenn/DynVision_Working/data/processed/cifar10/train_all/train.beton

# Create model
model = DyRCNNx4(
    n_classes=10,  # Number of output classes
    input_dims=(20, 3, 32, 32),  # (timesteps, channels, height, width) Note: CIFAR10 is 32x32 natively
    recurrence_type="full",  # Type of recurrent connectivity
    dt=2.0,  # Integration time step (ms)
    tau=5.0  # Neural time constant (ms)
)

# Set path to the processed FFCV beton files
data_processed = project_paths.project_paths.data.processed / "cifar10"
train_beton = data_processed / "train_all" / "train.beton"
test_beton = data_processed / "train_all" / "val.beton" # validation set split from training set

# 2. Create temporal data loaders using FFCV
# This replaces `StimulusDurationDataLoader` by applying `ExtendDataTimeFFCV` inside the FFCV pipeline natively.
train_loader = get_ffcv_dataloader(
    path=train_beton,
    batch_size=32,
    data_timesteps=20, # This extends the images to 20 timesteps!
    resolution=32,
    num_workers=4,
    train=True,
    normalize=([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]), # CIFAR10 normalization
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

test_loader = get_ffcv_dataloader(
    path=test_beton,
    batch_size=32,
    data_timesteps=20,
    resolution=32,
    num_workers=4,
    train=False,
    normalize=([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

# Configure trainer
trainer = pl.Trainer(
    max_epochs=10,
    accelerator='auto',
    devices=1,
    logger=pl.loggers.TensorBoardLogger('logs/'),
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            monitor='val_accuracy',
            filename='custom-rcnn-{epoch:02d}-{val_accuracy:.2f}',
            save_top_k=3,
            mode='max'
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    ]
)

# Train the model
if __name__ == "__main__":
    trainer.fit(model, train_loader, test_loader)
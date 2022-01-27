from argparse import ArgumentParser
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


import wandb
from data import CustomDataModule
from my_model import CNN2RNNModel
from utils import initialize, get_valid_transforms, get_train_transforms
from train import split_data

from hyperparams import *

def sweep_iteration():
    train_data, val_data = split_data(seed=1234, mode='train')
    # set up W&B logger
    wandb.init()    # required to have access to `wandb.config`
    config = wandb.config
    wandb_logger = WandbLogger(project="lair-challenge")

    # setup data
    csv_feature_dict, label_encoder, label_decoder = initialize()

    train_transforms = get_train_transforms(
        IMAGE_HEIGHT, IMAGE_WIDTH)
    val_transforms = get_valid_transforms(
        IMAGE_HEIGHT, IMAGE_WIDTH)
    data_module = CustomDataModule(
        train=train_data,
        val=val_data,
        csv_feature_dict=csv_feature_dict,
        label_encoder=label_encoder,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
    )

    # setup model - note how we refer to sweep parameters with wandb.config
    model = CNN2RNNModel(
        max_len=MAX_LEN,
        embedding_dim=EMBEDDING_DIM,
        num_features=NUM_FEATURES,
        class_n=CLASS_N,
        rate=DROPOUT_RATE,
        learning_rate=config.LEARNING_RATE,
    )

    ckpt_path = f'./weights/{MODEL_NAME}/'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    checkpoint = ModelCheckpoint(
        monitor='score',
        dirpath=ckpt_path,
        filename='{epoch}-{score:.3f}',
        save_top_k=3,
        mode='max',
        save_weights_only=True
    )

    early_stop = EarlyStopping(
        monitor='score',
        patience=3,
        verbose=False,
        mode='max'
    )

    # setup Trainer
    trainer = pl.Trainer(
        max_epochs=config.EPOCHS,
        gpus=2,
        precision=16,
        callbacks=[checkpoint, early_stop],
        strategy="ddp",
        log_every_n_steps=5,
        logger=wandb_logger,
        limit_train_batches=0.5
    )

    # train
    trainer.fit(model, data_module)


if __name__ == "__main__":
    # parser = ArgumentParser()
    # parser.add_argument('--sweep_id', type=str)
    # args = parser.parse_args()

    sweep_config = {
        "method": "bayes",   # Random search
        "metric": {           # We want to maximize val_acc
            "name": "score",
            "goal": "maximize"
        },
        "parameters": {
            "EPOCHS": {
                # Choose from pre-defined values
                "distribution": "uniform",
                "min": 10,
                "max": 30
            },
            "LEARNING_RATE": {
                # log uniform distribution between exp(min) and exp(max)
                "distribution": "log_uniform",
                "min": -9.21,   # exp(-9.21) = 1e-4
                "max": -4.61    # exp(-4.61) = 1e-2
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="lair_challenge")
    wandb.agent(sweep_id, function=sweep_iteration)

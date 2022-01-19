import os
from argparse import ArgumentParser
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils.data import Dataset
from torchvision import models
from tqdm import tqdm

import wandb
from data import LAIRDataModule, csv_feature_dict, label_decoder, label_encoder
from hyperparams import (batch_size, dropout_rate, embedding_dim, epochs,
                         learning_rate, max_len, vision_pretrain)
from my_model import LitModel


def main(config):
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    # parser = ArgumentParser()
    # parser.add_argument('--batch_size', default=32, type=int)
    # parser = pl.Trainer.add_argparse_args(parser)
    # parser = CNN2RNN.add_model_specific_args(parser)
    # args = parser.parse_args()

    # neptune_logger = NeptuneLogger(
    #     api_key=os.environ.get("NEPTUNE_API_TOKEN"),  # replace with your own
    #     project="caplab/Dacon-lair",  # format "<WORKSPACE/PROJECT>"
    # )
    lair_data = LAIRDataModule(batch_size)
    model = LitModel(max_len=max_len, embedding_dim=embedding_dim, num_features=len(
        csv_feature_dict), class_n=len(label_encoder), rate=dropout_rate, lr=config.learning_rate)

    wandb_logger = WandbLogger(project="lair-challenge")
    wandb_logger.watch(model.cnn)
    # wandb_logger.watch(model.rnn)
    
    # model.load_state_dict(torch.load(save_path, map_location=device))


    checkpoint_callback = ModelCheckpoint(
        monitor='score',
        # dirpath='.checkpoint',
        filename='{epoch}-{val_score:.3f}',
        save_top_k=3,
        mode='max',
        save_weights_only=True,
    )

    early_stop_callback = EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    verbose=False,
                    mode='min'
                    )
    # limit train_batches=20
    trainer = Trainer(max_epochs=epochs, gpus=2, precision=16,
                      default_root_dir=os.getcwd(), 
                      log_every_n_steps=5,
                      callbacks=[checkpoint_callback], logger=wandb_logger, 
                      strategy='ddp')

    # trainer.tune(model)
    trainer.fit(model, datamodule=lair_data)

 
# Sweep parameters


if __name__ == "__main__":
    hyperparameter_defaults = dict(
    learning_rate = learning_rate
)
    wandb.init(config=hyperparameter_defaults)
    config = wandb.config   
    main(config)

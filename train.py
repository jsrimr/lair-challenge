import json
import os
from glob import glob

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models

import wandb
from data import CustomDataModule
from my_model import CNN2RNNModel
from utils import initialize

ROOT_DIR = 'data'


def get_train_transforms(height, width):
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        # A.OneOf([
        #                   A.HorizontalFlip(p=1),
        #                   A.RandomRotate90(p=1),
        #                   A.VerticalFlip(p=1) ], p=0.75),
        ToTensorV2(),
    ])


def get_valid_transforms(height, width):
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2()
    ])



def split_data(split_rate=0.2, seed=42, mode='train'):
    """
    Use for model trained image and time series.
    """
    if mode == 'train':
        train = sorted(glob(f'{ROOT_DIR}/train/*'))

        labelsss = pd.read_csv(f'{ROOT_DIR}/train.csv')['label']
        train, val = train_test_split(
            train, test_size=split_rate, random_state=seed, stratify=labelsss)

        return train, val
    elif mode == 'test':
        test = sorted(glob(f'{ROOT_DIR}/test/*'))

        return test



        
def run(config, train_idx, val_idx):
    """
    Use for model trained image and time series.
    """
    # train_data, val_data = split_data(seed=1234, mode='train')
    
    train_data, val_data = train_path_list[train_idx], train_path_list[val_idx]

    train_transforms = get_train_transforms(
        config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    val_transforms = get_valid_transforms(
        config.IMAGE_HEIGHT, config.IMAGE_WIDTH)

    data_module = CustomDataModule(
        train=train_data,
        val=val_data,
        csv_feature_dict=csv_feature_dict,
        label_encoder=label_encoder,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        num_workers=config.NUM_WORKERS,
        batch_size=config.BATCH_SIZE,
    )

    model = CNN2RNNModel(
        max_len=config.MAX_LEN,
        embedding_dim=config.EMBEDDING_DIM,
        num_features=config.NUM_FEATURES,
        class_n=config.CLASS_N,
        rate=config.DROPOUT_RATE,
        learning_rate=config.LEARNING_RATE,
    )
    wandb_logger = WandbLogger(project="lair-challenge")

    wandb_logger.watch(model.cnn)
    wandb_logger.watch(model.rnn)
    pl.seed_everything(1234)

    ckpt_path = f'./weights/{config.MODEL_NAME}/'
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

    trainer = pl.Trainer(
        # max_epochs=config.EPOCHS,
        gpus=2,
        precision=16,
        callbacks=[checkpoint, early_stop],
        strategy="ddp",
        log_every_n_steps=5,
        logger=wandb_logger
    )

    trainer.fit(model, data_module)


# Sweep parameters
from hyperparams import (BATCH_SIZE, CLASS_N, DROPOUT_RATE, EMBEDDING_DIM,
                         EPOCHS, IMAGE_HEIGHT, IMAGE_WIDTH, LEARNING_RATE,
                         MAX_LEN, NUM_FEATURES, NUM_WORKERS, SEED)

from attrdict import AttrDict

if __name__ == "__main__":
    hyperparameter_defaults = dict(
        SEED=SEED,
        IMAGE_WIDTH=IMAGE_WIDTH,
        IMAGE_HEIGHT=IMAGE_HEIGHT,
        BATCH_SIZE=BATCH_SIZE,
        CLASS_N=CLASS_N,
        LEARNING_RATE=LEARNING_RATE,
        EMBEDDING_DIM=EMBEDDING_DIM,
        NUM_FEATURES=NUM_FEATURES,
        MAX_LEN=MAX_LEN,
        DROPOUT_RATE=DROPOUT_RATE,
        EPOCHS=EPOCHS,
        NUM_WORKERS=NUM_WORKERS,
        MODEL_NAME='ConvNeXt-B-22k'
    )

    csv_feature_dict, label_encoder, label_decoder = initialize()
    train_path_list = np.array(sorted(glob(f'{ROOT_DIR}/train/*')))

    wandb.init(config=hyperparameter_defaults)  # 이거 없으면 wandb 에서 프로젝트를 자꾸 새로 만들더라.. 왜 그러지..
    # wandb.define_metric("score", summary="max") # 이거하니까 logging 이 안됨
    config = wandb.config
    # config = AttrDict(hyperparameter_defaults)

    kf_stratified = StratifiedKFold(shuffle=True)
    train = sorted(glob(f'{ROOT_DIR}/train/*'))
    labelsss = pd.read_csv(f'{ROOT_DIR}/train.csv')['label']


    # ref : https://www.kaggle.com/haqishen/ranzcr-1st-place-soluiton-cls-model-small-ver#Train-&-Validation-Function
    for train_idx, val_idx in kf_stratified.split(train, labelsss):
        run(config, train_idx, val_idx)

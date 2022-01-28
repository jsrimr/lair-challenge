from argparse import ArgumentParser
from datetime import datetime
import json
import os
from glob import glob
from tabnanny import check
import cv2
import numpy as np

import pytorch_lightning as pl
import torch

from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.metrics import f1_score

from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models

import wandb
from data import CustomDataModule
from my_model import CNN2RNNModel
from utils import initialize


from utils import get_train_transforms, get_valid_transforms, split_data, ROOT_DIR

        
def run(config, train_idx=None, val_idx=None, full_train=False):
    """
    Use for model trained image and time series.
    """
    if train_idx and val_idx:
        # from utils import ROOT_DIR
        train_path_list = np.array(sorted(glob(f'{ROOT_DIR}/train/*')))
        train_data, val_data = train_path_list[train_idx], train_path_list[val_idx]
    else:
        train_data, val_data = split_data(seed=1234, mode='train')
    

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

    if full_train:
        # 모든 데이터 포함함
        train_data = np.array(sorted(glob(f'{ROOT_DIR}/train/*')))
        callbacks = [checkpoint]
    else:
        callbacks = [checkpoint, early_stop]
        

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
        cnn_model_name=config.MODEL_NAME,
        img_size=config.IMAGE_HEIGHT
    )
    wandb_logger = WandbLogger(project="lair-challenge")

    wandb_logger.watch(model.cnn)
    wandb_logger.watch(model.rnn)
    pl.seed_everything(1234)

    trainer = pl.Trainer(
        max_epochs=config.EPOCHS,
        gpus=2,
        precision=16,
        callbacks=callbacks,
        strategy="ddp",
        log_every_n_steps=5,
        logger=wandb_logger
    )

    trainer.fit(model, data_module)


# Sweep parameters
from hyperparams import (BATCH_SIZE, CLASS_N, DROPOUT_RATE, EMBEDDING_DIM,
                         EPOCHS, IMAGE_HEIGHT, IMAGE_WIDTH, LEARNING_RATE,
                         MAX_LEN, NUM_FEATURES, NUM_WORKERS, SEED, MODEL_NAME)

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
        MODEL_NAME=MODEL_NAME
    )

    csv_feature_dict, label_encoder, label_decoder = initialize()
    

    wandb.init(config=hyperparameter_defaults)  # 이거 없으면 wandb 에서 프로젝트를 자꾸 새로 만들더라.. 왜 그러지..
    # wandb.define_metric("score", summary="max") # 이거하니까 logging 이 안됨
    config = wandb.config
    wandb.run.name = f'{IMAGE_WIDTH}_{MODEL_NAME}_{datetime.now().strftime("%m%d%H%M")}'
    # config = AttrDict(hyperparameter_defaults)


    # ===========k-fold cv===================
    # kf_stratified = StratifiedKFold(shuffle=True)
    # train = sorted(glob(f'{ROOT_DIR}/train/*'))
    # labelsss = pd.read_csv(f'{ROOT_DIR}/train.csv')['label']
    # ref : https://www.kaggle.com/haqishen/ranzcr-1st-place-soluiton-cls-model-small-ver#Train-&-Validation-Function
    # for train_idx, val_idx in kf_stratified.split(train, labelsss):
    #     run(config, train_idx, val_idx)
    # =========================================

    parser = ArgumentParser()
    parser.add_argument('--full_train', action='store_true')
    args = parser.parse_args()

    run(config, full_train=args.full_train)
